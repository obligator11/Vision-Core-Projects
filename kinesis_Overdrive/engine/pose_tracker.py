"""Wraps MediaPipe Pose to expose just what the game needs: wrist positions."""

import cv2
import mediapipe as mp


class PoseTracker:
    LEFT_WRIST_IDX = 15
    RIGHT_WRIST_IDX = 16

    def __init__(self, model_complexity=1, min_detection_confidence=0.7,
                 min_tracking_confidence=0.7):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """Returns {'left_wrist': (x, y), 'right_wrist': (x, y)} or None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        left = lm[self.LEFT_WRIST_IDX]
        right = lm[self.RIGHT_WRIST_IDX]
        return {
            "left_wrist": (int(left.x * w), int(left.y * h)),
            "right_wrist": (int(right.x * w), int(right.y * h)),
        }

    def close(self):
        self._pose.close()
