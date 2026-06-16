import cv2
import numpy as np

class HUDVisualizer:
    """
    Renders telemetry metadata, predictive dashed tracks, observed absolute steps, 
    and multi-color error offset vectors onto incoming frames.
    """
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_dashed_line(self, img: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 1, gap: int = 8):
        """Calculates discrete sub-segments along a linear line to render dashed components."""
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0: return
        pts = np.linspace(pt1, pt2, int(dist / gap) if int(dist / gap) > 1 else 2)
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i+1].astype(int)), color, thickness, lineType=cv2.LINE_AA)

    def render_hud(self, frame: np.ndarray, metrics: dict, tracker_history: list) -> np.ndarray:
        h, w, _ = frame.shape
        
        # 1. Overlay Telemetry Panel Layout Block
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (380, 175), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (15, 15), (380, 175), (80, 80, 80), 1)

        status = metrics.get("status", "INITIALIZING")
        status_colors = {
            "NOMINAL": (0, 255, 0),    # Emerald
            "DEVIATION": (0, 255, 255), # Amber
            "ANOMALY": (0, 140, 255),  # Orange Glow
            "CRITICAL": (0, 0, 255)     # Deep Crimson Red
        }
        hud_color = status_colors.get(status, (200, 200, 200))

        # Render Core Real-time Telemetry Data Panels
        cv2.putText(frame, f"SYS STATUS: {status}", (25, 45), self.font, 0.65, hud_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"ERROR MAG: {metrics['error_magnitude']:.2f} px", (25, 75), self.font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        
        vel = metrics.get("velocity", np.zeros(2))
        acc = metrics.get("acceleration", np.zeros(2))
        cv2.putText(frame, f"VELOCITY : [{vel[0]:.1f}, {vel[1]:.1f}] px/s", (25, 105), self.font, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"ACCEL    : [{acc[0]:.1f}, {acc[1]:.1f}] px/s2", (25, 135), self.font, 0.5, (190, 190, 190), 1, cv2.LINE_AA)

        # 2. Render Motion Trajectories & Paths
        if len(tracker_history) >= 2:
            # Reconstruct and plot observed continuous motion via a solid trace line
            for i in range(len(tracker_history) - 1):
                p1 = tuple(tracker_history[i][0].astype(int))
                p2 = tuple(tracker_history[i+1][0].astype(int))
                cv2.line(frame, p1, p2, (255, 200, 0), 2, lineType=cv2.LINE_AA)

        # 3. Draw Predicted Next Position Target (Dashed Reference Layer)
        centroid = metrics.get("centroid")
        predicted = metrics.get("predicted")
        if centroid is not None and predicted is not None:
            c_tuple = tuple(centroid.astype(int))
            p_tuple = tuple(predicted.astype(int))
            
            # Predictive motion track segment plotted via dashed overlay lines
            self.draw_dashed_line(frame, c_tuple, p_tuple, (200, 200, 200), thickness=2)
            cv2.circle(frame, p_tuple, 6, (150, 150, 150), 1, lineType=cv2.LINE_AA)
            cv2.circle(frame, c_tuple, 5, hud_color, -1, lineType=cv2.LINE_AA)

            # 4. Render Dynamic Localization Vector Arrows
            if status in ["ANOMALY", "CRITICAL"]:
                cv2.arrowedLine(frame, p_tuple, c_tuple, (0, 0, 255), 3, tipLength=0.3, lineType=cv2.LINE_AA)
                
                # 5. Full-Frame structural flash overlays for high violations
                alert_box = np.zeros_like(frame)
                cv2.rectangle(alert_box, (0, 0), (w, h), (0, 0, 255), 4)
                cv2.putText(alert_box, "WARNING: PHYSICS BREAK DETECTED", (w // 2 - 220, h - 50), self.font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.addWeighted(alert_box, 0.4, frame, 1.0, 0, frame)

        return frame