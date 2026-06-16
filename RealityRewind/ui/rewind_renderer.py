import cv2
import numpy as np
import time
from utils.geometry import extrapolate_reverse_vector

class RewindRenderer:
    """
    Applies vector math matrices to transform frames, drawing ghost trail 
    paths, telemetry dashboards, and time indicators onto adjustable viewports.
    """
    def __init__(self):
        self.hud_color_active = (0, 238, 255)       # High-vis neon yellow
        self.hud_color_rewind = (0, 0, 255)         # High-vis crimson red
        self.trail_color = (255, 140, 0)            # Deep neon blue

    def render(self, canvas: np.ndarray, tracks: list, velocities: list, is_rewind: bool, speed: float):
        h, w = canvas.shape[:2]
        overlay = canvas.copy()

        # Render Spatio-Temporal Tracking Map
        if len(tracks) > 1:
            num_frames = len(tracks)
            for step in range(num_frames):
                # Scale alpha tracking gradients progressively backward in time
                alpha = float(step) / num_frames
                current_pts = tracks[step]
                current_vel = velocities[step] if step < len(velocities) else [None]*len(current_pts)

                for i, pt in enumerate(current_pts):
                    pt_x, pt_y = int(pt[0]), int(pt[1])
                    if not (0 <= pt_x < w and 0 <= pt_y < h):
                        continue

                    # Draw keypoint particles
                    cv2.circle(overlay, (pt_x, pt_y), 4, self.trail_color, -1)

                    # Extrapolate cinematic kinematics
                    if is_rewind and i < len(current_vel):
                        vel = current_vel[i]
                        if np.linalg.norm(vel) > 30: # Threshold filter noise
                            past_pt, head_angle = extrapolate_reverse_vector(pt, vel, delta_t=0.15)
                            if 0 <= past_pt[0] < w and 0 <= past_pt[1] < h:
                                cv2.line(overlay, (pt_x, pt_y), (past_pt[0], past_pt[1]), (200, 255, 0), 1)

            # Fuse graphics overlay arrays via linear interpolation alpha blend
            cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        # Draw Interface HUD Metrics Dashboard
        hud_color = self.hud_color_rewind if is_rewind else self.hud_color_active
        mode_text = f"MODE: REALTIME RECORD" if not is_rewind else f"MODE: REALITY REWIND ({speed:0.1f}X)"
        
        # Border frame telemetry
        cv2.rectangle(canvas, (15, 15), (410, 85), (20, 20, 20), -1)
        cv2.rectangle(canvas, (15, 15), (410, 85), hud_color, 1)
        cv2.putText(canvas, mode_text, (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
        
        # Draw running status ticks
        status_char = "<<" if is_rewind else "REC"
        if is_rewind and int(time.time() * 4) % 2 == 0:
            status_char = "    " # Pulse text notification flash layer
        cv2.putText(canvas, status_char, (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bottom screen structural interaction rules
        cv2.putText(canvas, "[SPACE]: Hold to Rewind  |  [1, 2, 4]: Speed Adjust  |  [Q]: Exit", 
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)