import cv2
import numpy as np
from config import SystemConfig
from utils.drawing_utils import DrawingUtils

class UIOverlay:
    """Renders data dashboards analytics components directly over captured raw input displays frames."""
    
    def __init__(self) -> None:
        pass

    def render_hud(self, frame: cv2.Mat, fatigue_score: float, attention_score: float, risk_status: str, landmarks_contract) -> cv2.Mat:
        """Appends interactive analytics metrics displays overlays directly above base image arrays configurations."""
        # Determine Color Status dynamic assignments boundaries definitions mappings systems
        if risk_status == "DANGER":
            status_color = SystemConfig.COLOR_DANGER
        elif risk_status == "WARNING":
            status_color = SystemConfig.COLOR_WARNING
        else:
            status_color = SystemConfig.COLOR_SAFE

        # 1. Overlay Geometric Mesh Coordinates Lines Vectors Formats
        if landmarks_contract is not None:
            # Draw facial landmarks mesh
            for pt in landmarks_contract.all_mesh_points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, SystemConfig.COLOR_MESH, -1)
                
            # Highlight structural eye parameters traces contours boundaries shapes
            DrawingUtils.draw_polyline(frame, landmarks_contract.left_eye, True, SystemConfig.COLOR_SAFE, 1)
            DrawingUtils.draw_polyline(frame, landmarks_contract.right_eye, True, SystemConfig.COLOR_SAFE, 1)
            DrawingUtils.draw_polyline(frame, landmarks_contract.inner_lips, True, SystemConfig.COLOR_WARNING, 1)

        # 2. Draw Dashboard Panel Backgrounds Boxes Areas Widgets
        cv2.rectangle(frame, (10, 10), (280, 140), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (280, 140), status_color, 1)

        # 3. Add Metric Strings Annotations Labels Vectors Text Fields
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"STATUS: {risk_status}", (20, 35), font, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Fatigue Score: {int(fatigue_score)}/100", (20, 65), font, 0.55, SystemConfig.COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Attention Score: {int(attention_score)}/100", (20, 95), font, 0.55, SystemConfig.COLOR_TEXT, 1, cv2.LINE_AA)

        # 4. Visualized Risk Bar Matrix Gauges Controls Components Designs
        # Fatigue Level Meter Bar
        cv2.rectangle(frame, (20, 115), (260, 125), (50, 50, 50), -1)
        fill_w = int((fatigue_score / 100.0) * 240)
        cv2.rectangle(frame, (20, 115), (20 + fill_w, 125), status_color, -1)

        return frame