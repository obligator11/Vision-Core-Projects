import cv2
import time
import threading
import numpy as np
from typing import Tuple, Optional

class ThreadedPoseTracker:
    """
    Manages hardware webcam frame capture interfaces inside an isolated thread
    and extracts a primary point of tracking (e.g., center of motion index)
    to establish zero-lag raw stream buffering.
    """
    def __init__(self, src: int = 0):
        self.stream = cv2.VideoCapture(src)
        # Configure initial frame sizing parameters
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        # Simplified tracker state: background subtraction center for tracking point
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

    def start(self) -> 'ThreadedPoseTracker':
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self) -> None:
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def extract_target_landmark(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts the center-of-mass coordinate point of major motion variations.
        Acts as a functional mock for skeletal point/hand tracking matrices.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        # Apply smoothing operations to eliminate environmental visual noise
        fg_mask = cv2.GaussianBlur(fg_mask, (11, 11), 0)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 800:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    return np.array([cX, cY], dtype=float)
        return None

    def stop(self) -> None:
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()