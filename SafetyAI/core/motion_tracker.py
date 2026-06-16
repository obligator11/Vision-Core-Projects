"""
SafetyAI - Spatial Velocity & Jitter Suppression Engine
Maintains an array-backed circular buffer for keypoint variance analysis.
"""
import collections
import time
import numpy as np

class MotionTracker:
    def __init__(self, buffer_window_size=30):
        self.window_size = buffer_window_size
        # Buffers tracking history of specific key joints
        self.centroid_history = collections.deque(maxlen=buffer_window_size)
        self.landmark_history = collections.deque(maxlen=buffer_window_size)
        self.last_timestamp = time.time()
        
    def update(self, keypoints: dict) -> dict:
        """
        Ingests fresh tracking coordinates, updates structural time-series metrics,
        and outputs accurate physical velocities and spatial variances.
        """
        current_time = time.time()
        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        if dt <= 0:
            dt = 0.033  # Standard 30 FPS baseline normalization fallback

        metrics = {
            'centroid_velocity_y': 0.0,
            'spatial_variance': 100.0,
            'is_moving': True
        }

        if not keypoints:
            return metrics

        # Compute current Frame Center of Mass (Torso Core Core Anchor)
        try:
            mid_hip_x = (keypoints['left_hip'][0] + keypoints['right_hip'][0]) / 2.0
            mid_hip_y = (keypoints['left_hip'][1] + keypoints['right_hip'][1]) / 2.0
            current_centroid = np.array([mid_hip_x, mid_hip_y])
        except KeyError:
            return metrics

        # Compute Downward Vertical Acceleration Velocity 
        if len(self.centroid_history) > 0:
            prev_centroid = self.centroid_history[-1]
            # Absolute pixel displacement delta scaled over delta time
            velocity_y = (current_centroid[1] - prev_centroid[1]) / dt
            metrics['centroid_velocity_y'] = max(0.0, velocity_y)

        self.centroid_history.append(current_centroid)

        # Build Rolling Structural Mesh Layer to analyze micro-movements
        stable_anchors = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        frame_sample = []
        for anchor in stable_anchors:
            if anchor in keypoints:
                frame_sample.extend([keypoints[anchor][0], keypoints[anchor][1]])

        if len(frame_sample) == 8:
            self.landmark_history.append(frame_sample)

        # Evaluate Dynamic Time-Series Overlapping Variance
        if len(self.landmark_history) == self.window_size:
            history_matrix = np.array(self.landmark_history) # Shape: [W, 8]
            variances = np.var(history_matrix, axis=0)
            mean_variance = float(np.mean(variances))
            metrics['spatial_variance'] = mean_variance
            
            # If tracking jitter variance drops below an engineered noise floor, mark worker as completely static
            if mean_variance < 1.8:
                metrics['is_moving'] = False
        else:
            metrics['is_moving'] = True

        return metrics

    def reset(self):
        """
        Flushes the time-series sliding memory rings completely.
        """
        self.centroid_history.clear()
        self.landmark_history.clear()
        self.last_timestamp = time.time()