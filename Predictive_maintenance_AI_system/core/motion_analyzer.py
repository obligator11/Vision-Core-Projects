import cv2
import numpy as np
import config

class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None
        
    def calculate_vibration_intensity(self, current_frame: np.ndarray) -> float:
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (config.GAUSSIAN_BLUR_KERNAL, config.GAUSSIAN_BLUR_KERNAL), 0)

        if self.prev_gray is None:
            self.prev_gray = gray_blurred
            return 0.0

        frame_delta = cv2.absdiff(self.prev_gray, gray_blurred)

        _, threshold_mask = cv2.threshold(frame_delta, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        total_pixels = threshold_mask.size
        active_motion_pixels = cv2.countNonZero(threshold_mask)

        motion_score = (active_motion_pixels / total_pixels) * 100.0

        self.prev_gray = gray_blurred

        return min(motion_score * 5.0 , 100.0)