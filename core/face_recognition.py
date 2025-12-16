import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis

# ADD THIS LINE at the top to import configuration variables
from config.config import (
    SIMILARITY_THRESHOLD, FACE_DETECTION_MODEL, DETECTION_SIZE,
    FACE_DETECTION_BACKEND, DNN_PROTO_PATH, DNN_MODEL_PATH, DNN_CONFIDENCE_THRESHOLD
)

class Face:
    """Generic Face object to standardize results between backends"""
    def __init__(self, bbox, det_score, embedding=None, kps=None):
        self.bbox = bbox # [x1, y1, x2, y2]
        self.det_score = det_score
        self.embedding = embedding
        self.kps = kps

class FaceRecognitionHandler:
    def __init__(self, db_manager): 
        # Initialize InsightFace (Always needed for Recognition/Registration)
        self.app = FaceAnalysis(
            name=FACE_DETECTION_MODEL, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
        
        # Initialize OpenCV DNN if backend selected
        self.backend = FACE_DETECTION_BACKEND
        self.net = None
        if self.backend == 'opencv_dnn':
            try:
                print(f"Loading OpenCV DNN Model from {DNN_MODEL_PATH}...")
                self.net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
                print("OpenCV DNN loaded successfully.")
            except Exception as e:
                print(f"Error loading OpenCV DNN: {e}. Fallback to InsightFace.")
                self.backend = 'insightface'

        self.db_manager = db_manager
        self.similarity_threshold = SIMILARITY_THRESHOLD 
        self.registered_faces = self.load_face_encodings()




    
    def detect_faces(self, frame):
        """Detect faces in a frame using selected backend"""
        if self.backend == 'opencv_dnn' and self.net:
            return self._detect_faces_opencv(frame)
        else:
            return self.app.get(frame)

    def _detect_faces_opencv(self, frame):
        """Internal method for OpenCV DNN detection"""
        (h, w) = frame.shape[:2]
        # Resize to 640x640 (standard SD resolution) for better small/multi face detection
        # The model is trained on 300x300 but works better at higher res for small faces
        TARGET_SIZE = (640, 640) 
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, TARGET_SIZE), 1.0,
            TARGET_SIZE, (104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Lower default threshold slightly to catch more faces (filtered later by track thresh if needed)
            if confidence > 0.4: 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure within bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Create Face object (Standardized)
                f = Face(bbox=np.array([startX, startY, endX, endY]), det_score=confidence)
                faces.append(f)
                
        return faces
    
    def extract_face_encoding(self, frame):
        """
        Extract face encoding from a frame. 
        Uses InsightFace pipeline regardless of detection backend 
        because we need the embedding model.
        """
        # Note: app.get() does detection + recognition
        faces = self.app.get(frame)
        
        if len(faces) == 0:
            return None, "No face detected"
        
        if len(faces) > 1:
            return None, "Multiple faces detected. Please ensure only one person is in frame"
        
        return faces[0].embedding, "Face encoding extracted successfully"
    
    def load_face_encodings(self):
        """Load face encodings from database"""
        return self.db_manager.get_all_face_encodings()
    
    def add_face_encoding(self, person_id, name, face_encoding):
        """Add a face encoding to the in-memory database"""
        self.registered_faces[person_id] = {
            'name': name,
            'encoding': face_encoding
        }
        return True
    
    def remove_face_encoding(self, person_id):
        """Remove a face encoding from the database"""
        if person_id in self.registered_faces:
            del self.registered_faces[person_id]
            return True
        return False
    
    def calculate_similarity(self, encoding1, encoding2):
        """Calculate cosine similarity between two face encodings"""
        similarity = np.dot(encoding1, encoding2) / (
            np.linalg.norm(encoding1) * np.linalg.norm(encoding2)
        )
        return similarity
    
    def recognize_face(self, face_encoding):
        """Recognize a face by comparing with registered faces"""
        if len(self.registered_faces) == 0:
            return None, None, 0.0
        
        max_similarity = 0.0
        recognized_id = None
        recognized_name = None
        
        for person_id, data in self.registered_faces.items():
            similarity = self.calculate_similarity(face_encoding, data['encoding'])
            
            if similarity > max_similarity and similarity > self.similarity_threshold:
                max_similarity = similarity
                recognized_id = person_id
                recognized_name = data['name']
        
        return recognized_id, recognized_name, max_similarity
    
    def recognize_multiple_faces(self, faces):
        """Recognize multiple faces in a frame"""
        recognized_faces = []
        
        for face in faces:
            person_id, person_name, similarity = self.recognize_face(face.embedding)
            
            recognized_faces.append({
                'bbox': face.bbox,
                'person_id': person_id,
                'person_name': person_name,
                'similarity': similarity,
                'det_score': face.det_score,
                'landmarks': face.kps if hasattr(face, 'kps') else None
            })
        
        return recognized_faces
    
    def verify_face(self, person_id, face_encoding):
        """Verify if a face encoding matches a specific person"""
        if person_id not in self.registered_faces:
            return False, 0.0
        
        similarity = self.calculate_similarity(
            face_encoding, 
            self.registered_faces[person_id]['encoding']
        )
        
        is_match = similarity > self.similarity_threshold
        return is_match, similarity
    
    def get_registered_count(self):
        """Get the number of registered faces"""
        return len(self.registered_faces)
    
    def get_all_registered_ids(self):
        """Get all registered person IDs"""
        return list(self.registered_faces.keys())
    
    def reload_face_encodings(self):
        """Reload face encodings from database"""
        self.registered_faces = self.load_face_encodings()
        return len(self.registered_faces)
    
    def update_similarity_threshold(self, new_threshold):
        """Update the similarity threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.similarity_threshold = new_threshold
            return True
        return False
