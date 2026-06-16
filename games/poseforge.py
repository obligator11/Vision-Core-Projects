import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
import pygame

# ==========================================
# CONFIGURATION AND PARAMETERS
# ==========================================
WINDOW_NAME = "PoseForge: Upper Body Tetris"
TARGET_FPS = 60

# Color definitions (BGR Format for OpenCV)
COLOR_BG = (15, 12, 12)
COLOR_WHITE = (245, 245, 245)
COLOR_NEON_CYAN = (255, 255, 0)
COLOR_NEON_MAGENTA = (200, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_GOLD = (0, 215, 255)

# MediaPipe upper-body pose connection tracking list
POSE_CONNECTIONS = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
]

# Upper body joint sets mapping for tracking comparison
JOINT_MAP = {
    "left_elbow": [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST],
    "right_elbow": [mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
    "left_shoulder": [mp.solutions.pose.PoseLandmark.LEFT_HIP, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
    "right_shoulder": [mp.solutions.pose.PoseLandmark.RIGHT_HIP, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
}

# ==========================================
# PROCEDURAL SYSTEM AUDIO SYNTHESIZER
# ==========================================
class ProceduralAudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self.sample_rate = 44100
        
        self.sound_success = self._synthesize_chime()
        self.sound_impact = self._synthesize_impact()
        self.sound_combo = self._synthesize_combo()
        self.last_tone_time = 0

    def _convert_to_sound(self, arr):
        arr_stereo = np.column_stack((arr, arr))
        return pygame.sndarray.make_sound(arr_stereo.astype(np.int16))

    def _synthesize_chime(self):
        t = np.linspace(0, 0.3, int(self.sample_rate * 0.3), False)
        sweep = np.sin(2 * np.pi * (440 + t * 1200) * t)
        decay = np.exp(-5 * t)
        wave = sweep * decay * 16383
        return self._convert_to_sound(wave)

    def _synthesize_impact(self):
        t = np.linspace(0, 0.4, int(self.sample_rate * 0.4), False)
        noise = np.random.uniform(-1, 1, len(t))
        bass_sine = np.sin(2 * np.pi * 80 * t)
        decay = np.exp(-8 * t)
        wave = (noise * 0.4 + bass_sine * 0.6) * decay * 20000
        return self._convert_to_sound(wave)

    def _synthesize_combo(self):
        t = np.linspace(0, 0.5, int(self.sample_rate * 0.5), False)
        sine1 = np.sin(2 * np.pi * 523.25 * t)
        sine2 = np.sin(2 * np.pi * 659.25 * t)
        sine3 = np.sin(2 * np.pi * 783.99 * t)
        decay = np.exp(-4 * t)
        wave = (sine1 + sine2 + sine3) / 3.0 * decay * 16383
        return self._convert_to_sound(wave)

    def play_success(self):
        self.sound_success.play()

    def play_impact(self):
        self.sound_impact.play()

    def play_combo(self):
        self.sound_combo.play()

    def play_accuracy_tone(self, accuracy):
        now = time.time()
        if now - self.last_tone_time < 0.12:
            return
        freq = 250 + int(accuracy * 600)
        t = np.linspace(0, 0.08, int(self.sample_rate * 0.08), False)
        wave = np.sin(2 * np.pi * freq * t) * 5000 * accuracy
        sound = self._convert_to_sound(wave)
        sound.play()
        self.last_tone_time = now

# ==========================================
# MATHEMATICAL SIGNAL SMOOTHING LAYER
# ==========================================
class KinematicSmoothingFilter:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.state = None

    def filter_payload(self, current_positions):
        if self.state is None or self.state.shape != current_positions.shape:
            self.state = np.copy(current_positions)
            return self.state
        self.state = self.alpha * current_positions + (1.0 - self.alpha) * self.state
        return self.state

# ==========================================
# GEOMETRIC EMBEDDING MODULE
# ==========================================
class PoseEmbeddingEngine:
    @staticmethod
    def compute_interior_angle(p1, p2, p3):
        # Explicit 3D vector construction to damp down Z-depth anomalies
        v1 = p1 - p2
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cosine_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        return math.degrees(np.arccos(cosine_angle))

    @staticmethod
    def generate_pose_embedding(landmarks):
        embedding = {}
        for joint_name, indices in JOINT_MAP.items():
            p1 = landmarks[indices[0]]
            p2 = landmarks[indices[1]]
            p3 = landmarks[indices[2]]
            embedding[joint_name] = PoseEmbeddingEngine.compute_interior_angle(p1, p2, p3)
        return embedding

# ==========================================
# TARGET SHAPE TEMPLATES (UPPER BODY ONLY)
# ==========================================
SHAPE_TEMPLATES = [
    {
        "name": "The Cyber-T",
        "angles": {"left_elbow": 175, "right_elbow": 175, "left_shoulder": 90, "right_shoulder": 90},
        "description": "Keep both arms perfectly straight out horizontally."
    },
    {
        "name": "Neon Atlas",
        "angles": {"left_elbow": 90, "right_elbow": 90, "left_shoulder": 90, "right_shoulder": 90},
        "description": "Raise elbows up out to the side, hands pointing straight up."
    },
    {
        "name": "Iron Aegis",
        "angles": {"left_elbow": 45, "right_elbow": 45, "left_shoulder": 20, "right_shoulder": 20},
        "description": "Bring elbows tight to chest, cross forearms in defense guard."
    },
    {
        "name": "Zen Bow",
        "angles": {"left_elbow": 175, "right_elbow": 45, "left_shoulder": 90, "right_shoulder": 90},
        "description": "Extend left arm completely sideways, flex right arm inward."
    }
]

# ==========================================
# SYSTEM CORE PIPELINE
# ==========================================
class PoseForgeGameEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        
        self.mp_pose = mp.solutions.pose
        self.pose_processor = self.mp_pose.Pose(
            min_detection_confidence=0.5,  # Lowered slightly to ensure stable retention under movement
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        self.audio = ProceduralAudioEngine()
        self.smoothing = KinematicSmoothingFilter(alpha=0.32)
        
        self.score = 0
        self.combo = 0
        self.current_level = 1
        self.base_speed = 0.010  # Slightly slowed down approach to give comfortable adjust window
        self.difficulty_escalation = 1.0
        
        self.target_shape = None
        self.shape_progress_scale = 0.05
        self.select_next_shape()
        
        self.viewport_center_x = 0.5
        self.viewport_center_y = 0.4
        self.viewport_scale = 1.0
        
        self.last_frame_time = time.time()
        self.fps = TARGET_FPS

    def select_next_shape(self):
        self.target_shape = random.choice(SHAPE_TEMPLATES)
        self.shape_progress_scale = 0.02
        
    def run_similarity_score(self, player_embedding):
        target_angles = self.target_shape["angles"]
        total_error = 0.0
        weight_sum = 0.0
        error_breakdown = {}
        
        for joint_name, target_angle in target_angles.items():
            current_angle = player_embedding[joint_name]
            angle_diff = abs(current_angle - target_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
                
            total_error += angle_diff
            weight_sum += 1.0
            
            # Widen error forgiveness matrix bounds to compensate camera depth perspective collapse
            tolerance = 35.0 if "shoulder" in joint_name else 28.0
            error_breakdown[joint_name] = angle_diff < tolerance
            
        mean_angle_error = total_error / weight_sum
        # Dynamic normalization mapping curve for realistic scoring feedback
        accuracy = max(0.0, min(1.0, 1.0 - (mean_angle_error / 55.0)))
        return accuracy, error_breakdown

    def apply_auto_framing(self, landmarks_2d, frame_w, frame_h):
        if len(landmarks_2d) == 0:
            return
        upper_joints = landmarks_2d[0:24]
        min_x = np.min(upper_joints[:, 0])
        max_x = np.max(upper_joints[:, 0])
        min_y = np.min(upper_joints[:, 1])
        max_y = np.max(upper_joints[:, 1])
        
        body_w = max_x - min_x
        body_h = max_y - min_y
        
        target_center_x = (min_x + max_x) / 2.0
        target_center_y = (min_y + max_y) / 2.0
        
        desired_span = max(body_w / 0.6, body_h / 0.6)
        target_scale = np.clip(desired_span, 0.4, 1.2)
        
        self.viewport_center_x += 0.08 * (target_center_x - self.viewport_center_x)
        self.viewport_center_y += 0.08 * (target_center_y - self.viewport_center_y)
        self.viewport_scale += 0.08 * (target_scale - self.viewport_scale)

    def draw_glowing_hud(self, render_frame, accuracy, joint_states):
        h, w = render_frame.shape[:2]
        
        vignette_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(vignette_mask, (w // 2, h // 2), int(w * 0.7), 255, -1)
        vignette_mask = cv2.GaussianBlur(vignette_mask, (51, 51), 0)
        render_frame[vignette_mask == 0] = (render_frame[vignette_mask == 0] * 0.4).astype(np.uint8)
        
        cv2.putText(render_frame, f"SHAPE: {self.target_shape['name'].upper()}", (25, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_NEON_CYAN, 2, cv2.LINE_AA)
        cv2.putText(render_frame, self.target_shape["description"], (25, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
        
        cv2.putText(render_frame, f"SCORE: {self.score}", (w - 240, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_NEON_MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(render_frame, f"COMBO: x{self.combo}", (w - 240, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GOLD, 2, cv2.LINE_AA)
        cv2.putText(render_frame, f"LVL: {self.current_level}", (w - 240, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
        
        bar_x1, bar_y1 = 30, h - 50
        bar_x2, bar_y2 = 380, h - 30
        cv2.rectangle(render_frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 40, 40), -1, cv2.LINE_AA)
        
        fill_width = int((bar_x2 - bar_x1) * accuracy)
        bar_color = COLOR_GREEN if accuracy > 0.70 else (COLOR_GOLD if accuracy > 0.50 else COLOR_RED)
        cv2.rectangle(render_frame, (bar_x1, bar_y1), (bar_x1 + fill_width, bar_y2), bar_color, -1, cv2.LINE_AA)
        
        cv2.putText(render_frame, f"MATCH ACCURACY: {int(accuracy * 100)}%", (bar_x1, bar_y1 - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 1, cv2.LINE_AA)

        self.render_target_shape_overlays(render_frame)

    def generate_template_coords(self):
        angles = self.target_shape["angles"]
        coords = {}
        
        coords[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value] = np.array([0.40, 0.40])
        coords[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value] = np.array([0.60, 0.40])
        coords[mp.solutions.pose.PoseLandmark.LEFT_HIP.value] = np.array([0.43, 0.75])
        coords[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value] = np.array([0.57, 0.75])
        
        l_sh_angle = math.radians(180 - angles["left_shoulder"])
        l_elbow_vec = np.array([-math.sin(l_sh_angle), math.cos(l_sh_angle)]) * 0.14
        coords[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value] = coords[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value] + l_elbow_vec
        
        l_el_angle = l_sh_angle + math.radians(180 - angles["left_elbow"])
        l_wrist_vec = np.array([-math.sin(l_el_angle), math.cos(l_el_angle)]) * 0.14
        coords[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value] = coords[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value] + l_wrist_vec
        
        r_sh_angle = math.radians(180 - angles["right_shoulder"])
        r_elbow_vec = np.array([math.sin(r_sh_angle), math.cos(r_sh_angle)]) * 0.14
        coords[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value] = coords[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value] + r_elbow_vec
        
        r_el_angle = r_sh_angle + math.radians(180 - angles["right_elbow"])
        r_wrist_vec = np.array([math.sin(r_el_angle), math.cos(r_el_angle)]) * 0.14
        coords[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value] = coords[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value] + r_wrist_vec

        return coords

    def render_target_shape_overlays(self, frame):
        h, w = frame.shape[:2]
        template_normalized_coords = self.generate_template_coords()
        
        # 1. Approaching Full-Screen Wall Silhouette
        wall_overlay = frame.copy()
        center_x, center_y = w // 2, h // 2
        current_wall_scale = self.shape_progress_scale
        alpha = min(1.0, current_wall_scale * 1.6)
        
        wall_pixels = {}
        for idx, norm_pt in template_normalized_coords.items():
            offset_x = (norm_pt[0] - 0.5) * current_wall_scale * w * 1.5
            offset_y = (norm_pt[1] - 0.5) * current_wall_scale * h * 1.5
            wall_pixels[idx] = (int(center_x + offset_x), int(center_y + offset_y))
            
        wall_color = COLOR_GREEN if current_wall_scale > 0.85 else COLOR_NEON_CYAN
        wall_thickness = int(2 + current_wall_scale * 6)
        
        for conn in POSE_CONNECTIONS:
            p1, p2 = conn[0].value, conn[1].value
            if p1 in wall_pixels and p2 in wall_pixels:
                cv2.line(wall_overlay, wall_pixels[p1], wall_pixels[p2], wall_color, wall_thickness, cv2.LINE_AA)
                
        cv2.addWeighted(wall_overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
        
        # 2. Picture-in-Picture Frame
        pip_w, pip_h = 160, 160
        pip_x, pip_y = 25, 95
        
        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), (30, 26, 26), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (pip_x, pip_y), (pip_x + pip_w, pip_y + pip_h), COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)
        cv2.putText(frame, "TARGET HOLE", (pip_x + 10, pip_y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_NEON_MAGENTA, 1, cv2.LINE_AA)
        
        pip_center_x = pip_x + (pip_w // 2)
        pip_center_y = pip_y + (pip_h // 2) + 12
        pip_scale = 120
        
        pip_pixels = {}
        for idx, norm_pt in template_normalized_coords.items():
            px = int(pip_center_x + (norm_pt[0] - 0.5) * pip_scale)
            py = int(pip_center_y + (norm_pt[1] - 0.5) * pip_scale)
            pip_pixels[idx] = (px, py)
            
        for conn in POSE_CONNECTIONS:
            p1, p2 = conn[0].value, conn[1].value
            if p1 in pip_pixels and p2 in pip_pixels:
                cv2.line(frame, pip_pixels[p1], pip_pixels[p2], COLOR_WHITE, 2, cv2.LINE_AA)
                
        for idx, pt in pip_pixels.items():
            cv2.circle(frame, pt, 3, COLOR_GOLD, -1, cv2.LINE_AA)

    def execute_main_loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        
        while self.cap.isOpened():
            success, raw_frame = self.cap.read()
            if not success:
                break
                
            raw_frame = cv2.flip(raw_frame, 1)
            h, w, _ = raw_frame.shape
            
            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            results = self.pose_processor.process(rgb_frame)
            
            joint_states = {k: False for k in JOINT_MAP.keys()}
            accuracy = 0.0
            
            if results.pose_landmarks:
                # Build true scale-invariant space matrices using raw 3D pose arrays
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
                coords_2d = landmarks_array[:, :2]
                smoothed_coords_2d = self.smoothing.filter_payload(coords_2d)
                landmarks_array[:, :2] = smoothed_coords_2d
                
                self.apply_auto_framing(smoothed_coords_2d, w, h)
                player_embedding = PoseEmbeddingEngine.generate_pose_embedding(landmarks_array)
                accuracy, joint_states = self.run_similarity_score(player_embedding)
                
                self.audio.play_accuracy_tone(accuracy)
                self.render_player_skeleton(raw_frame, landmarks_array, joint_states)
                
            self.shape_progress_scale += self.base_speed * self.difficulty_escalation
            
            if self.shape_progress_scale >= 1.0:
                # Lowered the baseline acceptance requirement from 0.76 to 0.70 for robust tracking matches
                if accuracy >= 0.70:  
                    self.score += int(100 * accuracy) + (self.combo * 25)
                    self.combo += 1
                    self.audio.play_success()
                    if self.combo % 3 == 0:
                        self.audio.play_combo()
                        self.current_level += 1
                        self.difficulty_escalation += 0.10
                else:
                    self.combo = 0
                    self.audio.play_impact()
                    self.difficulty_escalation = max(1.0, self.difficulty_escalation - 0.08)
                    
                self.select_next_shape()
                
            self.draw_glowing_hud(raw_frame, accuracy, joint_states)
            
            now = time.time()
            self.fps = 1.0 / (now - self.last_frame_time + 1e-6)
            self.last_frame_time = now
            cv2.putText(raw_frame, f"{int(self.fps)} FPS", (w - 95, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
            
            cv2.imshow(WINDOW_NAME, raw_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

    def render_player_skeleton(self, frame, landmarks, joint_states):
        h, w = frame.shape[:2]
        pixel_points = {}
        for idx, lm in enumerate(landmarks):
            pixel_points[idx] = (int(lm[0] * w), int(lm[1] * h))
            
        for conn in POSE_CONNECTIONS:
            p1_idx, p2_idx = conn[0].value, conn[1].value
            if p1_idx in pixel_points and p2_idx in pixel_points:
                line_color = COLOR_WHITE
                thickness = 2
                
                for joint_name, indices in JOINT_MAP.items():
                    if p1_idx in [i.value for i in indices] and p2_idx in [i.value for i in indices]:
                        if joint_states[joint_name]:
                            line_color = COLOR_GREEN
                            thickness = 4
                        else:
                            line_color = COLOR_RED
                            thickness = 2
                            
                cv2.line(frame, pixel_points[p1_idx], pixel_points[p2_idx], line_color, thickness, cv2.LINE_AA)
                
        for idx, pt in pixel_points.items():
            if idx in [j.value for indices in JOINT_MAP.values() for j in indices]:
                cv2.circle(frame, pt, 5, COLOR_NEON_MAGENTA, -1, cv2.LINE_AA)

if __name__ == "__main__":
    print("[POSEFORGE ECO SYSTEM] Booting Upper-Body comfort template with optimized angle tolerances...")
    engine = PoseForgeGameEngine()
    engine.execute_main_loop()