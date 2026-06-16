import cv2
import numpy as np

class CrowdDashboard:
    """
    Advanced display engine that handles multi-layer overlay blending, dynamic 
    bounding box rendering, and responsive real-time dashboard UI scaling.
    """
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_safe = (46, 204, 113)      # Emerald Green
        self.color_warning = (52, 152, 219)   # Amber Orange
        self.color_text_light = (245, 247, 250)

    def build_heatmap_overlay(self, canvas, boxes):
        h, w, _ = canvas.shape
        density_matrix = np.zeros((h, w), dtype=np.float32)
        
        for (x, y, bw, bh) in boxes:
            cx = int(x + (bw / 2))
            cy = int(y + (bh / 2))
            
            # Scale heatmap radius based on bounding box height to handle perspective compression
            radius = max(int(bh * 1.2), 30)
            x1, x2 = max(0, cx - radius), min(w, cx + radius)
            y1, y2 = max(0, cy - radius), min(h, cy + radius)
            
            # Avoid mathematical calculation exceptions if a bounding box slips out of frame bounds
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue
                
            ax = np.arange(x1, x2) - cx
            ay = np.arange(y1, y2) - cy
            xm, ym = np.meshgrid(ax, ay)
            
            kernel = np.exp(-(xm**2 + ym**2) / (2 * (radius / 2.5)**2))
            density_matrix[y1:y2, x1:x2] += kernel * 25
            
        density_matrix = np.clip(density_matrix, 0, 255).astype(np.uint8)
        heatmap_colorized = cv2.applyColorMap(density_matrix, cv2.COLORMAP_JET)
        return cv2.addWeighted(canvas, 0.75, heatmap_colorized, 0.35, 0)

    def draw_telemetry_hud(self, canvas, telemetry, boxes, engine_label):
        """
        Renders an advanced visual HUD onto a dynamically scaled video frame buffer canvas.
        """
        h, w, _ = canvas.shape
        canvas = self.build_heatmap_overlay(canvas, boxes)
        
        level = telemetry["congestion_level"]
        if level == "HIGH CONGESTION":
            status_color = (0, 0, 242)      # Crimson Red Alert
            status_txt = "CRITICAL BOTTLE-NECK"
        elif level == "MEDIUM":
            status_color = (0, 140, 255)    # Warning Orange
            status_txt = "WARNING: SURGING DENSITY"
        else:
            status_color = (0, 220, 100)    # Safe Green
            status_txt = "OPTIMAL EFFICIENCY"

        centroids = telemetry["centroids"]
        labels = telemetry["cluster_labels"]
        
        # Render Bounding Boxes and Proximity Clustered Connections
        for idx, (x, y, bw, bh) in enumerate(boxes):
            is_clustered = idx < len(labels) and labels[idx] != -1
            box_color = status_color if is_clustered else (220, 220, 220)
            thickness = 2 if is_clustered else 1
            
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, thickness)
            cv2.putText(canvas, f"ID:{idx:02d}", (x, max(y - 6, 12)), self.font, 0.35, box_color, 1, cv2.LINE_AA)

        if len(centroids) > 0 and len(labels) > 0:
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    if labels[i] == labels[j] and labels[i] != -1:
                        cv2.line(canvas, tuple(centroids[i]), tuple(centroids[j]), (255, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(canvas, tuple(centroids[i]), 4, (0, 255, 255), -1, cv2.LINE_AA)

        # Dynamic Instrument Panel (Adapts context dimension rules relative to scaled resolution sizes)
        panel_w = min(340, int(w * 0.4))
        panel_h = min(240, int(h * 0.5))
        
        overlay_panel = canvas.copy()
        cv2.rectangle(overlay_panel, (0, 0), (panel_w, panel_h), (20, 24, 33), -1)
        cv2.addWeighted(overlay_panel, 0.85, canvas, 0.15, 0, dst=canvas)
        cv2.rectangle(canvas, (0, 0), (6, panel_h), status_color, -1)

        # Scale layout text scaling parameters to match custom user window scale factor variables
        text_scale = max(0.4, panel_w / 600.0)
        
        cv2.putText(canvas, "CROWD MONITOR SYSTEM", (15, int(panel_h * 0.12)), self.font, text_scale * 1.3, self.color_text_light, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Engine: {engine_label}", (15, int(panel_h * 0.22)), self.font, text_scale, (140, 150, 170), 1, cv2.LINE_AA)
        cv2.putText(canvas, status_txt, (15, int(panel_h * 0.42)), self.font, text_scale * 1.1, status_color, 2, cv2.LINE_AA)
        
        cv2.putText(canvas, f"Total Count : {len(boxes)} persons", (15, int(panel_h * 0.58)), self.font, text_scale, self.color_text_light, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Density     : {telemetry['density_score']}%", (15, int(panel_h * 0.68)), self.font, text_scale, self.color_text_light, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Flow Rating : {telemetry['flow_efficiency']}%", (15, int(panel_h * 0.78)), self.font, text_scale, self.color_text_light, 1, cv2.LINE_AA)
        
        # Flow Rating Progress Bar Track Bar Widget Overlay
        bar_x = 15
        bar_y = int(panel_h * 0.88)
        bar_w = panel_w - 30
        bar_h = 5
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 75), -1)
        fill_w = int((telemetry['flow_efficiency'] / 100.0) * bar_w)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (242, 156, 12), -1)

        return canvas