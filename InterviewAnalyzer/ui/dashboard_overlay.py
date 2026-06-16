import cv2
import mediapipe as mp

class DashboardOverlay:
    """Draws multi-modal performance dashboards, structural coordinate grids, 
    and face metrics that adjust seamlessly to resizable application canvases."""
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
        self.mesh_style = self.mp_draw.DrawingSpec(color=(0, 235, 235), thickness=1, circle_radius=1)
        self.pose_style = self.mp_draw.DrawingSpec(color=(240, 160, 10), thickness=2, circle_radius=2)

    def draw(self, frame, face_data, posture_data, audio_data, score_payload):
        h, w, _ = frame.shape

        # 1. Render Face Mesh Contours and Skeletal Connections
        if face_data["face_detected"] and face_data["raw_mesh_points"]:
            self.mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_data["raw_mesh_points"],
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mesh_style
            )

        if posture_data["pose_detected"] and posture_data["raw_pose_landmarks"]:
            self.mp_draw.draw_landmarks(
                image=frame,
                landmark_list=posture_data["raw_pose_landmarks"],
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_style,
                connection_drawing_spec=self.pose_style
            )

        # 2. Left HUD Overlay Panel Configuration (Responsive Scaling)
        panel_w = min(460, int(w * 0.45))
        panel_h = min(280, int(h * 0.55))
        
        canvas = frame.copy()
        cv2.rectangle(canvas, (15, 15), (panel_w, panel_h), (12, 12, 12), -1)
        cv2.addWeighted(canvas, 0.75, frame, 0.25, 0, dst=frame)

        score = score_payload["confidence_score"]
        if score >= 75:
            accent_color = (75, 230, 100)      # Confident Green
        elif score >= 50:
            accent_color = (50, 190, 245)      # Neutral Yellow
        else:
            accent_color = (60, 60, 255)       # Nervous Red

        # Render Metrics HUD Content Text Fields
        cv2.putText(frame, "AUTOMATED INTERVIEW ENGINE", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.line(frame, (30, 55), (panel_w - 20, 55), (80, 80, 80), 1)

        cv2.putText(frame, f"TOTAL CONFIDENCE:  {score}%", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent_color, 2)
        cv2.putText(frame, f"EYE CONTACT SCORE: {score_payload.get('eye_contact_score', 100)}%", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1)
        cv2.putText(frame, f"BODY STABILITY:   {score_payload.get('posture_stability_score', 100)}%", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1)
        cv2.putText(frame, f"VOCAL FLUENCY:    {score_payload.get('speech_fluency_score', 100)}%", (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1)
        
        # ASYMMETRY DIAGNOSTIC INDICATOR
        dev_val = face_data.get("debug_deviation", 0.0)
        cv2.putText(frame, f"YAW DEV: {dev_val:.4f} (MAX LIMIT: 0.070)", (30, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 235, 235), 1)

        # Scale progress indicator bar inside panel container limitations
        bar_right = panel_w - 30
        cv2.rectangle(frame, (30, panel_h - 35), (bar_right, panel_h - 20), (45, 45, 45), -1)
        fill_w = int((score / 100.0) * (bar_right - 30))
        cv2.rectangle(frame, (30, panel_h - 35), (30 + fill_w, panel_h - 20), accent_color, -1)

        # 3. Bottom Transcripts Panel Configuration (Responsive Width alignment)
        bottom_canvas = frame.copy()
        cv2.rectangle(bottom_canvas, (15, h - 110), (w - 15, h - 15), (18, 18, 18), -1)
        cv2.addWeighted(bottom_canvas, 0.78, frame, 0.22, 0, dst=frame)

        raw_txt = audio_data.get("transcript", "Listening...")
        text_char_limit = max(45, int(w * 0.09))
        wrapped_txt = raw_txt[:text_char_limit] + "..." if len(raw_txt) > text_char_limit else raw_txt

        cv2.putText(frame, "LIVE LINGUISTIC ANALYTICS STREAM", (30, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 235, 235), 1)
        cv2.putText(frame, f"TRANSCRIPT: \"{wrapped_txt}\"", (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"FILLER WORDS: {audio_data.get('filler_count', 0)}   |   VOCAL SENTIMENT: {audio_data.get('sentiment_label', 'NEUTRAL')}", (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (175, 175, 175), 1)

        return frame