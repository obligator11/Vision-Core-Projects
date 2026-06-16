import numpy as np
import time

class RewindEngine:
    """
    Manages bounded time tracking paths via decoupled dictionary structures.
    Safeguards the pipeline against dynamic shape changes when keypoints drop or appear.
    """
    def __init__(self, fps=30, buffer_seconds=4):
        self.fps = fps
        self.max_size = int(fps * buffer_seconds)
        
        # Track histories via distinct trajectory series maps
        self.point_tracks = {}
        self.max_tracked_features = 0
        
        self.is_rewinding = False
        self.rewind_start_time = None
        self.rewind_speed = 1.0
        self.buffer_length = 0

    def record_state(self, keypoints: np.ndarray):
        """
        Records points to independent chronological coordinate lists instead of stacked arrays.
        """
        if self.is_rewinding:
            return

        num_points = len(keypoints)
        self.max_tracked_features = max(self.max_tracked_features, num_points)

        for idx in range(self.max_tracked_features):
            if idx not in self.point_tracks:
                self.point_tracks[idx] = []
                
            if idx < num_points:
                self.point_tracks[idx].append(np.copy(keypoints[idx]))
            else:
                if len(self.point_tracks[idx]) > 0:
                    self.point_tracks[idx].append(self.point_tracks[idx][-1])
                else:
                    self.point_tracks[idx].append(np.array([0.0, 0.0], dtype=np.float32))

            if len(self.point_tracks[idx]) > self.max_size:
                self.point_tracks[idx].pop(0)

        if self.max_tracked_features > 0:
            self.buffer_length = len(self.point_tracks[0])

    def set_rewind_mode(self, active: bool, speed=1.0):
        """Toggles state between record and reverse execution patterns."""
        self.is_rewinding = active
        self.rewind_speed = speed
        if active:
            self.rewind_start_time = time.time()
        else:
            self.rewind_start_time = None

    def compute_temporal_tracks(self):
        """
        Safely decouples velocity derivatives from shape mutations.
        Returns a list of structured coordinate maps and matching trajectory vectors.
        """
        if self.buffer_length == 0 or self.max_tracked_features == 0:
            return [], []

        history_frames = [[] for _ in range(self.buffer_length)]
        velocity_frames = [[] for _ in range(self.buffer_length)]

        for p_idx in range(self.max_tracked_features):
            pts = self.point_tracks.get(p_idx, [])
            if len(pts) < self.buffer_length:
                continue
                
            for f_idx in range(self.buffer_length):
                history_frames[f_idx].append(pts[f_idx])
                
                if f_idx == 0:
                    velocity_frames[f_idx].append(np.array([0.0, 0.0], dtype=np.float32))
                else:
                    v = (pts[f_idx] - pts[f_idx - 1]) * self.fps
                    velocity_frames[f_idx].append(v)

        history = [np.array(f, dtype=np.float32) for f in history_frames]
        velocities = [np.array(v, dtype=np.float32) for v in velocity_frames]

        if not self.is_rewinding:
            return history, velocities

        elapsed_time = time.time() - self.rewind_start_time
        target_index = len(history) - 1 - int(elapsed_time * self.fps * self.rewind_speed)

        if target_index <= 0:
            self.rewind_start_time = time.time()
            target_index = len(history) - 1

        rewound_history = history[target_index:][::-1]
        rewound_velocities = [-v for v in velocities[target_index:]][::-1]

        return rewound_history, rewound_velocities