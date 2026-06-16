import numpy as np
import time
from collections import deque
from typing import Dict, Tuple, Optional

class KinematicPredictor:
    """
    Tracks coordinate data variables over specific historical timelines
    to calculate instantly updated velocity patterns.
    """
    def __init__(self, buffer_size: int = 15, prediction_horizon_sec: float = 0.25):
        self.buffer_size = buffer_size
        self.prediction_horizon = prediction_horizon_sec
        self.history: Dict[int, deque] = {}
        
    def update_and_predict(self, landmark_id: int, current_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates tracking points, runs velocity calculations, and 
        extrapolates where the path coordinates will land next.
        """
        now = time.monotonic()
        if landmark_id not in self.history:
            self.history[landmark_id] = deque(maxlen=self.buffer_size)
            
        self.history[landmark_id].append((now, current_pos.astype(float)))
        
        if len(self.history[landmark_id]) < 2:
            return current_pos, current_pos
            
        # Extract last two timestamps and positional matrices
        t_curr, pos_curr = self.history[landmark_id][-1]
        t_prev, pos_prev = self.history[landmark_id][-2]
        
        dt = t_curr - t_prev
        if dt <= 0:
            dt = 1e-5 # Safeguard against divide by zero errors
            
        # Instantaneous velocity matrix extrapolation
        velocity = (pos_curr - pos_prev) / dt
        
        # Extrapolate ghost coordinate vector point
        predicted_pos = pos_curr + (velocity * self.prediction_horizon)
        
        # Apply smoothing limits to keep predictions inside visual boundaries
        predicted_pos = np.clip(predicted_pos, [0, 0], [1920, 1080])
        
        return pos_curr, predicted_pos