import cv2
import numpy as np

class MotionTracker:
    """
    Tracks localized spatial keypoint changes inside the camera viewport 
    via sparse pyramidal Lucas-Kanade optical flow arrays.
    """
    def __init__(self, max_points_to_track=40):
        self.max_points = max_points_to_track
        self.feature_params = dict(maxCorners=self.max_points,
                                   qualityLevel=0.05,
                                   minDistance=15,
                                   blockSize=9)
        self.prev_gray = None
        self.current_keypoints = None

    def _extract_features(self, gray_frame):
        pts = cv2.goodFeaturesToTrack(gray_frame, **self.feature_params)
        if pts is not None:
            return pts.reshape(-1, 2)
        return np.empty((0, 2), dtype=np.float32)

    def update(self, bgr_frame) -> np.ndarray:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.current_keypoints is None or len(self.current_keypoints) < 8:
            self.prev_gray = gray
            self.current_keypoints = self._extract_features(gray)
            return self.current_keypoints

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, 
            self.current_keypoints.astype(np.float32), None,
            winSize=(21, 21), maxLevel=3
        )

        if next_pts is not None and status is not None:
            valid_idx = status.reshape(-1) == 1
            self.current_keypoints = next_pts[valid_idx]
        else:
            self.current_keypoints = np.empty((0, 2), dtype=np.float32)

        if len(self.current_keypoints) < 8:
            self.current_keypoints = self._extract_features(gray)

        self.prev_gray = gray
        return self.current_keypoints