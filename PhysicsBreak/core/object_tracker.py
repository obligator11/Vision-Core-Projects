import cv2
import numpy as np
import time
from collections import deque

class ObjectTracker:
    """
    Handles real-time color tracking in the HSV space and updates a thread-safe 
    historical buffer containing object positions and explicit timestamps.
    """
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.history = deque(maxlen=buffer_size)
        
        # Default tuning targets high-visibility neon greens/oranges
        self.lower_hsv = np.array([29, 86, 6])
        self.upper_hsv = np.array([64, 255, 255])
        
    def track_frame(self, frame: np.ndarray) -> tuple:
        """
        Extracts spatial center-of-mass centroids from an frame image matrix.
        Returns: (centroid_vector_or_None, binary_tracking_mask)
        """
        current_time = time.monotonic()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Clean background specular noise out of mask space
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroid = None
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 250:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = np.array([cx, cy], dtype=float)
                    self.history.append((centroid, current_time))
                    
        return centroid, mask