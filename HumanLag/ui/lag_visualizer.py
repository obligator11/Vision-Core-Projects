import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional

class InterfaceDashboard:
    """
    Renders high-fidelity telemetry profiles, target data tracking paths,
    and dynamically resizable canvas interface scales.
    """
    def __init__(self, window_title: str = "Human Lag Detector (Resizable HUD)"):
        self.window_title = window_title
        cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
        # Automatically resize the initial viewport canvas boundary window
        cv2.resizeWindow(self.window_title, 1024, 768)

    def draw_hud(self, canvas: np.ndarray, actual: Optional[np.ndarray], 
                 predicted: Optional[np.ndarray], lag_ms: float, 
                 distance_error: float, lag_history: deque) -> np.ndarray:
        
        # 1. Render Ghost Trajectory Point Vectors
        if actual is not None and predicted is not None:
            pt_actual = (int(actual[0]), int(actual[1]))
            pt_predicted = (int(predicted[0]), int(predicted[1]))
            
            # Distance error bounding radius mapping
            cv2.line(canvas, pt_actual, pt_predicted, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, pt_actual, 10, (0, 255, 0), -1, cv2.LINE_AA)       # Actual Target: Green
            cv2.circle(canvas, pt_predicted, 12, (0, 0, 255), 2, cv2.LINE_AA)    # Ghost Prediction: Red

        # 2. Draw Top Glassmorphic Telemetry Board Overlay Panel
        cv2.rectangle(canvas, (10, 10), (320, 130), (30, 30, 30), -1)
        cv2.rectangle(canvas, (10, 10), (320, 130), (70, 70, 70), 1)
        
        cv2.putText(canvas, f"LAG TELEMETRY ENGINE", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Est. Delay: {lag_ms:.1f} ms", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Spatial Error: {distance_error:.1f} px", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # 3. Render the Horizontal Live Lag Meter Gauge
        meter_y = 115
        cv2.rectangle(canvas, (20, meter_y), (310, meter_y + 8), (50, 50, 50), -1)
        fill_width = int(np.interp(lag_ms, [0, 400], [0, 290]))
        # Switch meter bar color profile relative to delay threshold conditions
        bar_color = (0, 255, 0) if lag_ms < 120 else ((0, 165, 255) if lag_ms < 250 else (0, 0, 255))
        cv2.rectangle(canvas, (20, meter_y), (20 + fill_width, meter_y + 8), bar_color, -1)

        # 4. Render Bottom Reaction Delay Graph Window Segment
        graph_w, graph_h = 280, 100
        h, w, _ = canvas.shape
        g_x, g_y = w - graph_w - 20, h - graph_h - 20
        
        # Ensure graph layout variables remain valid inside dynamic canvas bounds
        if g_x > 0 and g_y > 0:
            cv2.rectangle(canvas, (g_x, g_y), (g_x + graph_w, g_y + graph_h), (20, 20, 20), -1)
            cv2.rectangle(canvas, (g_x, g_y), (g_x + graph_w, g_y + graph_h), (80, 80, 80), 1)
            cv2.putText(canvas, "REACTION DELAY HISTOGRAM", (g_x + 5, g_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
            
            if len(lag_history) > 1:
                pts = []
                for idx, val in enumerate(lag_history):
                    x_pos = g_x + int(idx * (graph_w / (len(lag_history) - 1)))
                    y_pos = g_y + graph_h - int(np.interp(val, [0, 450], [0, graph_h - 10]))
                    pts.append([x_pos, y_pos])
                cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (0, 215, 255), 1, cv2.LINE_AA)

        return canvas

    def render_output(self, canvas: np.ndarray) -> int:
        cv2.imshow(self.window_title, canvas)
        return cv2.waitKey(1) & 0xFF

    def close_panels(self) -> None:
        cv2.destroyAllWindows()