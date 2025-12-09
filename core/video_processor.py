import cv2
import numpy as np
import supervision as sv
import os
from datetime import datetime
from config.config import UNKNOWN_FACES_DIR

class VideoProcessor:
    def __init__(self, face_handler):
        self.face_handler = face_handler
        
        # Initialize ByteTrack
        self.tracker = sv.ByteTrack(
            track_thresh=0.5,       
            track_buffer=30,        
            match_thresh=0.8,       
            frame_rate=30
        )
        
        # Initialize Annotator
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5,
            text_padding=5
        )
        
        # Track recognized faces (The Cache)
        self.tracker_id_to_person = {}
        
        # Track logged unknown faces to prevent duplicate logging
        self.logged_unknown_ids = set()
        
        # Ensure unknown faces directory exists
        if not os.path.exists(UNKNOWN_FACES_DIR):
            os.makedirs(UNKNOWN_FACES_DIR)
    
    def clear_cache(self):
        """Forces the processor to forget currently tracked faces"""
        self.tracker_id_to_person = {}
        self.logged_unknown_ids = set()

    def process_frame(self, frame, mark_attendance_callback=None, unknown_person_callback=None):
        """
        Process a single frame for Face Recognition and Tracking.
        Returns: (detections, labels, faces, messages)
        """
        from config.config import PROCESS_EVERY_N_FRAMES, RESIZE_FACTOR
        
        # Initialize frame counter if not exists
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
            self.last_detections = sv.Detections.empty()
            self.last_faces = []
        
        self.frame_count += 1
        
        # --- PROCESS EVERY Nth FRAME ---
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            
            # 1. Resize for faster inference
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            
            # 2. Detect faces on small frame
            faces = self.face_handler.detect_faces(small_frame)
            
            # 3. Scale back coordinates to original size
            for face in faces:
                face.bbox = face.bbox / RESIZE_FACTOR
                if hasattr(face, 'kps') and face.kps is not None:
                    face.kps = face.kps / RESIZE_FACTOR
            
            self.last_faces = faces
            
            # 4. Format detections for ByteTrack
            if len(faces) > 0:
                xyxy = np.array([face.bbox for face in faces])
                confidence = np.array([face.det_score for face in faces])
                class_id = np.zeros(len(faces), dtype=int)
                
                detections = sv.Detections(
                    xyxy=xyxy, 
                    confidence=confidence, 
                    class_id=class_id
                )
            else:
                detections = sv.Detections.empty()
            
            # 5. Update tracker
            tracked_detections = self.tracker.update_with_detections(detections)
            self.last_detections = tracked_detections
            
        else:
            # --- SKIP FRAME: USE LAST KNOWN DETECTIONS ---
            faces = self.last_faces
            tracked_detections = self.last_detections

        if tracked_detections.tracker_id is not None:
            tracked_detections.class_id = tracked_detections.tracker_id.astype(int)
        
        labels = []
        messages = []
        
        # Loop through tracked detections
        for i in range(len(tracked_detections)):
            if tracked_detections.tracker_id is None: continue
            
            tracker_id = tracked_detections.tracker_id[i]
            current_bbox = tracked_detections.xyxy[i]
            
            # --- CASE 1: EXISTING TRACK (We already know who this is) ---
            if tracker_id in self.tracker_id_to_person:
                person_id, person_name = self.tracker_id_to_person[tracker_id]
                label = f"{person_name} ({person_id})"
                
                # Update attendance (Only on processed frames to save DB calls)
                if self.frame_count % PROCESS_EVERY_N_FRAMES == 0 and mark_attendance_callback:
                    success, message = mark_attendance_callback(person_id, person_name)
                    if success and message and "Tracking" not in message:
                        messages.append(message)

            # --- CASE 2: NEW TRACK (We need to recognize the face) ---
            else:
                # Only run recognition on the processed frame to save resources
                if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                     label = f"Tracking #{tracker_id}"
                else:
                    best_face = None
                    max_iou = 0.0
                    
                    # Find matching face detection
                    for face in faces:
                        iou = self.calculate_iou(current_bbox, face.bbox)
                        if iou > 0.5 and iou > max_iou:
                            max_iou = iou
                            best_face = face

                    if best_face:
                        # Check against the pickle file
                        person_id, person_name, similarity = self.face_handler.recognize_face(best_face.embedding)
                        
                        if person_id:
                            self.tracker_id_to_person[tracker_id] = (person_id, person_name)
                            label = f"{person_name} ({person_id})"
                            
                            if mark_attendance_callback:
                                success, message = mark_attendance_callback(person_id, person_name)
                                if success and message:
                                    messages.append(message)
                        else:
                            label = f"Unknown #{tracker_id}"
                            
                            # --- CASE 3: UNKNOWN PERSON LOGGING ---
                            if unknown_person_callback and tracker_id not in self.logged_unknown_ids:
                                # 1. Save Snapshot
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                                filename = f"unknown_{timestamp}.jpg"
                                filepath = os.path.join(UNKNOWN_FACES_DIR, filename)
                                
                                # Save the face crop
                                x1, y1, x2, y2 = map(int, current_bbox)
                                h, w, _ = frame.shape
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                face_crop = frame[y1:y2, x1:x2]
                                
                                if face_crop.size > 0:
                                    cv2.imwrite(filepath, face_crop)
                                    
                                    # 2. Log to DB
                                    unknown_person_callback(filepath, best_face.embedding)
                                    self.logged_unknown_ids.add(tracker_id)
                                    messages.append(f"Logged Unknown Person #{tracker_id}")
                    else:
                        label = f"Tracking #{tracker_id}"
            
            labels.append(label)
        
        return tracked_detections, labels, faces, messages

    def annotate_frame(self, frame, detections, labels, faces):
        """
        Draw bounding boxes, labels, and landmarks on the frame.
        This allows the UI to draw on the *latest* frame using the *latest known* data.
        """
        # Annotate boxes and labels
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
            labels=labels
        )
        
        # Draw landmarks
        annotated_frame = self.draw_landmarks(annotated_frame, faces)
        
        return annotated_frame
    
    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def draw_landmarks(self, frame, faces):
        for face in faces:
            if hasattr(face, 'kps') and face.kps is not None:
                for point in face.kps:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        return frame
    
    def draw_info_panel(self, frame, info_dict):
        panel_height = 80
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        y_offset = 25
        x_offset = 20
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(panel, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        return np.vstack([panel, frame])
    
    def add_fps_counter(self, frame, fps):
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
        
    def get_detection_count(self, tracked_detections):
        if tracked_detections.tracker_id is not None:
            return len(tracked_detections.tracker_id)
        return 0