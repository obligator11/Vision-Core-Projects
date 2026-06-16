import numpy as np

class MathUtils:
    """Mathematical utility class for high-performance geometric matrix vectors evaluations."""
    
    @staticmethod
    def calculate_euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Computes the linear scalar distance between two N-dimensional geometric points."""
        return float(np.linalg.norm(point_a - point_b))

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        """
        Calculates the Eye Aspect Ratio (EAR) metric given 6 vertical/horizontal landmarks.
        Formula: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Vertical distances
        p2_p6 = MathUtils.calculate_euclidean_distance(eye_points[1], eye_points[5])
        p3_p5 = MathUtils.calculate_euclidean_distance(eye_points[2], eye_points[4])
        # Horizontal distance
        p1_p4 = MathUtils.calculate_euclidean_distance(eye_points[0], eye_points[3])
        
        if p1_p4 == 0.0:
            return 0.0
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    @staticmethod
    def calculate_mar(mouth_points: np.ndarray) -> float:
        """
        Calculates Mouth Aspect Ratio (MAR) based on inner perimeter vertical and horizontal spaces.
        Formula tailored around spatial opening bounds.
        """
        # Average vertical coordinates gap calculations
        v1 = MathUtils.calculate_euclidean_distance(mouth_points[1], mouth_points[7])
        v2 = MathUtils.calculate_euclidean_distance(mouth_points[2], mouth_points[6])
        v3 = MathUtils.calculate_euclidean_distance(mouth_points[3], mouth_points[5])
        # Horizontal base coordinates length
        h = MathUtils.calculate_euclidean_distance(mouth_points[0], mouth_points[4])
        
        if h == 0.0:
            return 0.0
        return (v1 + v2 + v3) / (3.0 * h)

    @staticmethod
    def estimate_head_pose(landmarks_3d: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float]:
        """
        Applies mathematical projection parsing to resolve 3D rotational matrices.
        Returns pitch, yaw, and roll scalars using structural face landfalls.
        """
        # Generic 3D model point matrices mappings
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)

        # Approximate focal lengths to establish projection parameters
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1)) # Assuming minimal lens distortion parameters
        
        import cv2
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, landmarks_3d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0

        # Decompose rotation vector using Rodrigues calculation
        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])
        
        return pitch, yaw, roll