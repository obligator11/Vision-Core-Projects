

import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.results = None

    def find_pose(self, frame, draw=True):
        """Run pose estimation on a BGR frame. Optionally draw the skeleton."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.pose.process(rgb)

        if draw and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        return frame

    def get_landmarks(self, frame):
        """
        Return a dict {landmark_name: (x_px, y_px, visibility)} for every
        detected body landmark, in pixel coordinates for the given frame.
        Returns an empty dict if nothing was detected.
        """
        landmarks = {}
        if self.results and self.results.pose_landmarks:
            h, w = frame.shape[:2]
            for lm_enum in self.mp_pose.PoseLandmark:
                lm = self.results.pose_landmarks.landmark[lm_enum.value]
                landmarks[lm_enum.name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
        return landmarks

    def close(self):
        self.pose.close()