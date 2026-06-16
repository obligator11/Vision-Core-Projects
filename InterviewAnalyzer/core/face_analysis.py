import cv2
import mediapipe as mp
import numpy as np

class FaceAnalyzer:
    """Computes face orientation and profile gaze vectors using structural layout asymmetry. 
    Completely eliminates loop errors on static profile images."""
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        # Structural facial feature anchors
        self.NOSE_TIP = 1
        self.LEFT_CHEEK = 234   # Outer edge boundary
        self.RIGHT_CHEEK = 454  # Outer edge boundary

    def analyze(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        payload = {
            "face_detected": False,
            "eye_contact_ratio": 1.0,
            "expression_variance": 0.0,
            "raw_mesh_points": None,
            "debug_deviation": 0.0
        }

        if not results.multi_face_landmarks:
            return payload

        payload["face_detected"] = True
        mesh_landmarks = results.multi_face_landmarks[0]
        payload["raw_mesh_points"] = mesh_landmarks

        pts = np.array([(lm.x * w, lm.y * h) for lm in mesh_landmarks.landmark])
        
        # Extract precise physical boundary coordinates
        nose = pts[self.NOSE_TIP]
        l_cheek = pts[self.LEFT_CHEEK]
        r_cheek = pts[self.RIGHT_CHEEK]
        
        # Calculate horizontal distance vectors from nose tip to each cheek profile edge
        dist_nose_to_left = np.linalg.norm(nose - l_cheek)
        dist_nose_to_right = np.linalg.norm(nose - r_cheek)
        
        # Calculate symmetry split ratio (A centered face yields exactly 0.5)
        total_face_span = dist_nose_to_left + dist_nose_to_right + 1e-6
        horizontal_symmetry_ratio = dist_nose_to_left / total_face_span
        
        # Measure head rotation angle deviation from the center lane
        yaw_deviation = abs(horizontal_symmetry_ratio - 0.5)
        payload["debug_deviation"] = float(yaw_deviation)

        # ABSOLUTE GEOMETRIC SCALE EVALUATION:
        # A look-away to the right or left pushes yaw_deviation past 0.07.
        # This flags profile shifts instantly, even on static test photos.
        if yaw_deviation > 0.07:
            # Deduct points smoothly as the face turns away from the camera lens axes
            payload["eye_contact_ratio"] = max(0.0, 1.0 - ((yaw_deviation - 0.07) * 4.5))
        else:
            payload["eye_contact_ratio"] = 1.0

        # Track lip variations for conversational dashboard analytics updates
        lip_top, lip_bottom = pts[13], pts[14]
        payload["expression_variance"] = float(np.linalg.norm(lip_top - lip_bottom))

        return payload