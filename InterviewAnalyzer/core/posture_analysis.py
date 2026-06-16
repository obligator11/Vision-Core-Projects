import cv2
import mediapipe as mp
import numpy as np
import time

class PostureAnalyzer:
    """Measures spatial stability, skeletal tilt offsets, and tracks 
    unconscious physical adjustments or fidgeting over time."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.history_limit = 30
        self.coordinate_history = []
        self.last_timestamp = time.time()

    def analyze(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        payload = {
            "pose_detected": False,
            "alignment_score": 100.0,
            "stability_score": 100.0,
            "raw_pose_landmarks": None
        }

        if not results.pose_landmarks:
            return payload

        payload["pose_detected"] = True
        landmarks = results.pose_landmarks
        payload["raw_pose_landmarks"] = landmarks

        l_sh = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        l_sh_v = np.array([l_sh.x * w, l_sh.y * h])
        r_sh_v = np.array([r_sh.x * w, r_sh.y * h])

        shoulder_delta_y = abs(l_sh_v[1] - r_sh_v[1])
        shoulder_span = np.linalg.norm(l_sh_v - r_sh_v) + 1e-6
        tilt_ratio = shoulder_delta_y / shoulder_span
        
        alignment_score = max(0.0, 100.0 - (tilt_ratio * 350.0))
        payload["alignment_score"] = float(alignment_score)

        current_time = time.time()
        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time

        current_centroid = (l_sh_v + r_sh_v) / 2.0
        self.coordinate_history.append(current_centroid)

        if len(self.coordinate_history) > self.history_limit:
            self.coordinate_history.pop(0)

        if len(self.coordinate_history) >= 2 and dt > 0:
            recent_movement = np.linalg.norm(self.coordinate_history[-1] - self.coordinate_history[-2])
            frame_velocity = recent_movement / dt
            stability_score = max(0.0, 100.0 - (frame_velocity * 0.12))
            payload["stability_score"] = float(stability_score)
        else:
            payload["stability_score"] = 100.0

        return payload