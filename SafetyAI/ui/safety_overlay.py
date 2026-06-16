"""
SafetyAI - Dynamic Graphic Presentation Interface Suite
Overlays tracking telemetry UI overlays using high-contrast color arrays.
"""
import cv2
import numpy as np

class SafetyOverlay:
    def __init__(self):
        # Color Palettes (BGR Scheme compliance rules layout mapping)
        self.COLOR_SAFE = (76, 175, 80)     # Neon Green
        self.COLOR_WARN = (0, 193, 255)     # Amber Orange
        self.COLOR_DANGER = (0, 0, 244)     # Critical Red
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_PANEL_BG = (22, 22, 22)  # Dark Charcoal Matte Window
        
    def render_hud(self, frame: np.ndarray, score: int, status: str, anomaly: str, motion_data: dict, landmarks):
        """
        Main paint orchestration execution loop logic pipeline window layout.
        """
        h, w, _ = frame.shape
        
        # 1. Establish Active Dynamic Contextual State Highlight Borders
        if status == "SAFE":
            current_theme = self.COLOR_SAFE
        elif status == "WARNING":
            current_theme = self.COLOR_WARN
        else:
            current_theme = self.COLOR_DANGER
            
        cv2.rectangle(frame, (0, 0), (w, h), current_theme, 6)
        
        # 2. Draw Top Control Status HUD Panel Background
        panel_h = 105
        overlay_layer = frame.copy()
        cv2.rectangle(overlay_layer, (0, 0), (w, panel_h), self.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay_layer, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, panel_h), (w, panel_h), current_theme, 2)
        
        # 3. Print Telemetry Metrics Elements Inside Panel
        cv2.putText(frame, f"CRITICAL SAFETY SCORE: {score}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.COLOR_WHITE, 2)
        
        cv2.putText(frame, f"STATUS: {status}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, current_theme, 2)
                    
        cv2.putText(frame, f"ANOMALY: {anomaly}", (20, 93), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_WHITE, 1)
                    
        # Render Micro Jitter Activity Index Metrics
        var_val = motion_data.get('spatial_variance', 0.0)
        cv2.putText(frame, f"VAR: {var_val:.2f}", (w - 160, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_WHITE, 1)
                    
        is_mov = "ACTIVE" if motion_data.get('is_moving', True) else "STATIC LOCK"
        cv2.putText(frame, f"MOTION: {is_mov}", (w - 160, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, 
                    (self.COLOR_SAFE if motion_data.get('is_moving', True) else self.COLOR_WARN), 2)

        # 4. Draw Structural Skeleton Mesh Linkage Layers
        if landmarks:
            self._draw_custom_skeleton(frame, landmarks, w, h, current_theme)
            
        return frame

    def _draw_custom_skeleton(self, frame, landmarks, w, h, draw_color):
        """
        Draws vector skeletons over standard MediaPipe output streams.
        """
        # Map structural bone indices links safely
        connections = [
            (11, 12), # Shoulder line Vector Anchor
            (11, 23), (12, 24), # Torso lateral core outlines
            (23, 24), # Pelvis base vector line
            (23, 27), (24, 28)  # Leg vertical path spans
        ]
        
        # Draw Bone Segments
        for start_idx, end_idx in connections:
            try:
                lm_s = landmarks.landmark[start_idx]
                lm_e = landmarks.landmark[end_idx]
                
                if lm_s.visibility > 0.5 and lm_e.visibility > 0.5:
                    pt_s = (int(lm_s.x * w), int(lm_s.y * h))
                    pt_e = (int(lm_e.x * w), int(lm_e.y * h))
                    cv2.line(frame, pt_s, pt_e, draw_color, 2, cv2.LINE_AA)
            except IndexError:
                continue

        # Draw Node Nodes Anchors
        tracked_joints = [0, 11, 12, 23, 24, 27, 28]
        for idx in tracked_joints:
            try:
                lm = landmarks.landmark[idx]
                if lm.visibility > 0.5:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, self.COLOR_WHITE, -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 6, draw_color, 1, cv2.LINE_AA)
            except IndexError:
                continue