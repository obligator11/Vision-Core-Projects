"""
SafetyAI - Core Inference Pipeline Wrapper
Encapsulates MediaPipe Pose to extract structured tracking landmarks safely.
"""
import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self, static_mode=False, model_complexity=1, smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_frame(self, frame: np.ndarray):
        """
        Processes incoming RGB image data and returns structured tracking frames.
        """
        if frame is None:
            return None, None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None, None
            
        return results.pose_landmarks, results
        
    def extract_key_vectors(self, landmarks, frame_width: int, frame_height: int) -> dict:
        """
        Translates raw normalized tracking points into absolute coordinate spaces.
        Catches indexing faults and missing target frames safely.
        """
        # --- CRITICAL PROTECTION LINE FIX ---
        if landmarks is None:
            return {}  # Return an empty dictionary smoothly if person is absent
            
        extracted_points = {}
        try:
            indices = {
                'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
                'left_hip': 23, 'right_hip': 24, 'left_ankle': 27, 'right_ankle': 28
            }
            
            for name, idx in indices.items():
                lm = landmarks.landmark[idx]
                if lm.visibility < 0.5:
                    return {}
                extracted_points[name] = np.array([
                    int(lm.x * frame_width),
                    int(lm.y * frame_height),
                    lm.z
                ])
                
        except (IndexError, AttributeError):
            return {}
            
        return extracted_points