import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from config import SystemConfig

@dataclass(frozen=True)
class FaceLandmarksContract:
    """Immutable data transfer object structure binding relevant normalized array locations elements."""
    left_eye: np.ndarray
    right_eye: np.ndarray
    inner_lips: np.ndarray
    pose_3d_points: np.ndarray
    all_mesh_points: np.ndarray

class FaceDetector:
    """Wraps MediaPipe Face Mesh processes execution pipelines context mapping frameworks structures."""
    
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        # Initialize inference structural setups targeting localized properties parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.6
        )

    def process_frame(self, frame: cv2.Mat) -> FaceLandmarksContract | None:
        """
        Extracts structural face matrices features parameters mappings from input frames.
        Returns mapped vector elements contract structs or None if tracking drops.
        """
        h, w, _ = frame.shape
        # Map BGR to RGB context layer space
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        all_pts = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.float32)
        
        # Populate functional subarrays tracking structural attributes components maps
        left_eye = all_pts[SystemConfig.LEFT_EYE_LANDMARKS]
        right_eye = all_pts[SystemConfig.RIGHT_EYE_LANDMARKS]
        inner_lips = all_pts[SystemConfig.INNER_LIPS_LANDMARKS]
        
        # Map 3D mathematical alignment structures dependencies points array
        pose_indices = [
            SystemConfig.NOSE_TIP_IDX, SystemConfig.CHIN_IDX,
            SystemConfig.LEFT_EYE_CORNER_IDX, SystemConfig.RIGHT_EYE_CORNER_IDX,
            SystemConfig.LEFT_MOUTH_CORNER_IDX, SystemConfig.RIGHT_MOUTH_CORNER_IDX
        ]
        pose_3d_points = all_pts[pose_indices]
        
        return FaceLandmarksContract(
            left_eye=left_eye,
            right_eye=right_eye,
            inner_lips=inner_lips,
            pose_3d_points=pose_3d_points,
            all_mesh_points=all_pts
        )
        
    def close(self) -> None:
        """Releases underlying model graphs context configurations processing frames allocation maps."""
        self.face_mesh.close()