import numpy as np
import pickle
import os
from insightface.app import FaceAnalysis

# ADD THIS LINE at the top to import configuration variables
from config.config import SIMILARITY_THRESHOLD, FACE_DETECTION_MODEL, DETECTION_SIZE 
# You should also update SIMILARITY_THRESHOLD here if you are using db_manager

class FaceRecognitionHandler:
    # You can remove similarity_threshold=0.6 from the arguments and use the config value
    def __init__(self, db_manager): 
        # Initialize InsightFace
        self.app = FaceAnalysis(
            # Ensure it uses the variable from config.py
            name=FACE_DETECTION_MODEL, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # CHANGE THIS LINE: Use the DETECTION_SIZE variable from config.py
        self.app.prepare(ctx_id=0, det_size=DETECTION_SIZE)
        
        self.db_manager = db_manager
        # Use the config threshold for consistency
        self.similarity_threshold = SIMILARITY_THRESHOLD 
        self.registered_faces = self.load_face_encodings()




    
    def detect_faces(self, frame):
        """Detect faces in a frame"""
        faces = self.app.get(frame)
        return faces
    
    def extract_face_encoding(self, frame):
        """Extract face encoding from a frame"""
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
