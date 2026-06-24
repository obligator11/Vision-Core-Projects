import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
import math
import random
import sys
from collections import deque

# -------------------------------------------------------------------------
# CONSTANTS & SETUP CONFIGURATION
# -------------------------------------------------------------------------
BASE_W, BASE_H = 1280, 720
FPS = 60

# MediaPipe Config Indexes
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1
SHOULDER_LINKS = [11, 12] # Left and Right shoulders from Pose

# -------------------------------------------------------------------------
# SYSTEM INITIALIZATION
# -------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
pygame.display.set_caption("🧠 Micro-Movement Detector (Extreme Stillness Target)")
clock = pygame.time.Clock()

# -------------------------------------------------------------------------
# THREAD-ISOLATED VIDEO CAPTURE ENGINE
# -------------------------------------------------------------------------
class ThreadedVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.stream.read()
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            grabbed, frame = self.stream.read()
            if not grabbed:
                continue
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.grabbed, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.running = False
        if self.stream.isOpened():
            self.stream.release()

# -------------------------------------------------------------------------
# PROCEDURAL AUDIO SYNTHESIZER ENGINE (REAL-TIME MATH WAVES)
# -------------------------------------------------------------------------
class AudioSynthesizer:
    @staticmethod
    def generate_wave(freq=440, duration=0.1, wave_type='sine', volume=0.4):
        sample_rate = 44100
        total_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, total_samples, False)
        
        if wave_type == 'sine':
            data = np.sin(2 * np.pi * freq * t)
        elif wave_type == 'square':
            data = np.sign(np.sin(2 * np.pi * freq * t))
        elif wave_type == 'sawtooth':
            data = 2 * (t * freq - np.floor(t * freq + 0.5))
        else:
            data = np.sin(2 * np.pi * freq * t)

        # Apply exponential decay envelope to avoid sudden audio pops
        envelope = np.exp(-3 * np.linspace(0, 1, total_samples))
        processed_data = data * envelope * volume
        
        # Convert to raw stereo sound bytes
        audio_buffer = np.repeat(processed_data.reshape(-1, 1), 2, axis=1)
        audio_buffer = (audio_buffer * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(audio_buffer)

# Initialize procedural tones
alert_beep = AudioSynthesizer.generate_wave(freq=880, duration=0.25, wave_type='square', volume=0.5)
heartbeat_low = AudioSynthesizer.generate_wave(freq=65, duration=0.08, wave_type='sine', volume=0.7)
heartbeat_high = AudioSynthesizer.generate_wave(freq=55, duration=0.08, wave_type='sine', volume=0.7)

# -------------------------------------------------------------------------
# SIGNAL SMOOTHING AND MATH HELPER UTILITIES
# -------------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def __call__(self, t, x):
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        
        dx = (x - self.x_prev) / dt
        edx = self._alpha(dt, self.d_cutoff) * dx + (1 - self._alpha(dt, self.d_cutoff)) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(dt, cutoff)
        
        x_filtered = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_filtered
        self.dx_prev = edx
        self.t_prev = t
        return x_filtered

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

def calculate_ear(landmarks, indices):
    # Eye Aspect Ratio calculation
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    h = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

# -------------------------------------------------------------------------
# MAIN GAME MACHINE ARCHITECTURE
# -------------------------------------------------------------------------
def run_game():
    stream = ThreadedVideoStream(src=0)
    
    # Initialize MediaPipe Solutions
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    # State tracking variables
    game_state = "START" # START, CALIBRATING, ACTIVE, DETECTED
    survival_time = 0.0
    precision_score = 100.0
    start_time = 0.0
    flash_timer = 0.0
    
    # Core Sensitivity Configuration
    base_sensitivity = 1.0
    current_sensitivity = 1.0
    sensitivity_spike_timer = 0.0
    is_spike_active = False
    is_fake_safe = False
    fake_safe_timer = 0.0
    extreme_mode = False
    
    # Heartbeat rhythm managers
    last_heartbeat_time = time.time()
    heartbeat_interval = 1.0
    heartbeat_toggle = True
    
    # Historical tracking deques for frame differencing and coordinate shifts
    prev_gray = None
    prev_face_nose = None
    prev_pose_shoulders = None
    prev_ear_left = None
    prev_ear_right = None
    
    # Visual metrics meters buffers
    motion_metrics = {"face_delta": 0.0, "pose_delta": 0.0, "frame_diff": 0.0, "eye_blink": 0.0}
    max_history_points = 40
    motion_history = deque(maxlen=max_history_points)
    
    print("[SYSTEM BOOT] Micro-Movement Detector Engine online.")

    while True:
        current_time = time.time()
        w, h = screen.get_size()
        
        # 1. Event Polling Processing Block
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stream.stop()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    stream.stop()
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    if game_state in ["START", "DETECTED"]:
                        game_state = "CALIBRATING"
                        calibration_start = time.time()
                        precision_score = 100.0
                        survival_time = 0.0
                if event.key == pygame.K_e and game_state == "START":
                    extreme_mode = not extreme_mode

        # 2. Frame Processing & Multi-Engine Vision Pipeline
        grabbed, bgr_frame = stream.read()
        if not grabbed or bgr_frame is None:
            # Render fallback screen state if camera isn't transmitting frame buffers
            screen.fill((15, 23, 42))
            font = pygame.font.SysFont("Consolas", 24)
            lbl = font.render("CRITICAL ERROR: WAITING FOR CAMERA SOURCE BUFFER...", True, (239, 68, 68))
            screen.blit(lbl, (50, h // 2))
            pygame.display.flip()
            continue

        # Convert frame color domains and invert across Y-axis for mirroring
        bgr_frame = cv2.flip(bgr_frame, 1)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        # Process MediaPipe Framework Pipeline
        face_results = mp_face_mesh.process(rgb_frame)
        pose_results = mp_pose.process(rgb_frame)
        
        # Calculate Screen-space bounding boxes for tracking overlay pipeline
        cam_h, cam_w, _ = bgr_frame.shape
        overlay_w = int(w * 0.28)
        overlay_h = int(overlay_w * (cam_h / cam_w))
        overlay_x = w - overlay_w - 30
        overlay_y = 30
        
        # Pixel-Level Frame Differencing Processing Matrix
        current_frame_diff_val = 0.0
        if prev_gray is not None and prev_gray.shape == gray_blur.shape:
            frame_delta = cv2.absdiff(prev_gray, gray_blur)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            current_frame_diff_val = np.sum(thresh) / float(thresh.size * 255)
        prev_gray = gray_blur

        # Extract metrics components safely
        face_movement_delta = 0.0
        pose_movement_delta = 0.0
        blink_delta = 0.0
        
        # Extract Face Mesh Tracking Vectors
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0].landmark
            nose_pos = np.array([face_lms[NOSE_TIP].x, face_lms[NOSE_TIP].y, face_lms[NOSE_TIP].z])
            
            if prev_face_nose is not None:
                face_movement_delta = np.linalg.norm(nose_pos - prev_face_nose)
            prev_face_nose = nose_pos
            
            # Eye Blink Delta Analytics (Detect rapid variations in EAR geometry)
            ear_l = calculate_ear(face_lms, LEFT_EYE_LANDMARKS)
            ear_r = calculate_ear(face_lms, RIGHT_EYE_LANDMARKS)
            
            if prev_ear_left is not None and prev_ear_right is not None:
                blink_delta = abs(ear_l - prev_ear_left) + abs(ear_r - prev_ear_right)
            prev_ear_left = ear_l
            prev_ear_right = ear_r
            
        # Extract Pose Tracking Structural Vectors
        if pose_results.pose_landmarks:
            pose_lms = pose_results.pose_landmarks.landmark
            sh_l = np.array([pose_lms[SHOULDER_LINKS[0]].x, pose_lms[SHOULDER_LINKS[0]].y])
            sh_r = np.array([pose_lms[SHOULDER_LINKS[1]].x, pose_lms[SHOULDER_LINKS[1]].y])
            current_shoulders = (sh_l + sh_r) / 2.0
            
            if prev_pose_shoulders is not None:
                pose_movement_delta = np.linalg.norm(current_shoulders - prev_pose_shoulders)
            prev_pose_shoulders = current_shoulders

        # Sync visual analytics metrics object data maps
        motion_metrics["face_delta"] = face_movement_delta * 120.0
        motion_metrics["pose_delta"] = pose_movement_delta * 150.0
        motion_metrics["frame_diff"] = current_frame_diff_val * 80.0
        motion_metrics["eye_blink"] = blink_delta * 40.0
        
        # 3. CORE STATISTICAL SCORING ENGINE AND STATE MACHINE
        total_instant_motion = (motion_metrics["face_delta"] + 
                                motion_metrics["pose_delta"] + 
                                motion_metrics["frame_diff"] + 
                                motion_metrics["eye_blink"])

        if game_state == "CALIBRATING":
            if current_time - calibration_start >= 3.0:
                game_state = "ACTIVE"
                start_time = time.time()
                last_heartbeat_time = time.time()
                motion_history.clear()
                
        elif game_state == "ACTIVE":
            survival_time = current_time - start_time
            
            # Continuous difficulty linear scale
            scaling_factor = 2.5 if extreme_mode else 1.0
            current_sensitivity = base_sensitivity + (survival_time * 0.08 * scaling_factor)
            
            # Random Sensitivity Spikes Matrix
            sensitivity_spike_timer -= 0.016
            if sensitivity_spike_timer <= 0:
                if is_spike_active:
                    is_spike_active = False
                    sensitivity_spike_timer = random.uniform(8.0, 15.0)
                else:
                    is_spike_active = True
                    sensitivity_spike_timer = random.uniform(2.0, 4.5)
            
            # Fake Safe Dynamic Windows
            fake_safe_timer -= 0.016
            if fake_safe_timer <= 0:
                is_fake_safe = not is_fake_safe
                fake_safe_timer = random.uniform(5.0, 10.0)

            # Apply modifiers to thresholds evaluations
            multiplier = current_sensitivity
            if is_spike_active:
                multiplier *= 2.2
            if is_fake_safe:
                multiplier *= 0.4
                
            # Compute Precision Evaluation Metrics
            step_penalty = total_instant_motion * multiplier * 0.15
            precision_score = max(0.0, min(100.0, precision_score - step_penalty + 0.02))
            
            # Append historical data bounds for tracking plot renders
            motion_history.append(total_instant_motion)
            
            # Heartbeat tempo manager linked to precision decay levels
            heartbeat_interval = max(0.2, 1.0 - (survival_time * 0.015) - ((100.0 - precision_score) * 0.008))
            if current_time - last_heartbeat_time >= heartbeat_interval:
                if heartbeat_toggle:
                    heartbeat_low.play()
                else:
                    heartbeat_high.play()
                heartbeat_toggle = not heartbeat_toggle
                last_heartbeat_time = current_time
                
            # Breach Detection Condition Thresholding Check
            breach_threshold = 1.6 if not extreme_mode else 0.6
            if total_instant_motion * (2.0 if is_spike_active else 1.0) > breach_threshold and not is_fake_safe:
                if survival_time > 1.2: # Brief structural entry grace frame windows
                    game_state = "DETECTED"
                    alert_beep.play()
                    flash_timer = 0.4

        # 4. RENDERING & DATA VISUALIZATION GRAPHICS INTERFACE LAYER
        # Render clean dark background palette
        screen.fill((10, 15, 30))
        
        # Handle Red Flash Overlays on structural motion detections
        if flash_timer > 0:
            flash_surface = pygame.Surface((w, h))
            flash_surface.fill((185, 28, 28))
            flash_surface.set_alpha(int(flash_timer * 255))
            screen.blit(flash_surface, (0, 0))
            flash_timer -= 0.016

        # Draw Title Banner Panel Elements
        font_title = pygame.font.SysFont("Consolas", 36, bold=True)
        font_sub = pygame.font.SysFont("Consolas", 18)
        font_metrics = pygame.font.SysFont("Consolas", 16)
        
        lbl_title = font_title.render("M I C R O - M O V E M E N T   D E T E C T O R", True, (255, 255, 255))
        lbl_subtitle = font_sub.render("EXTREME CV STILLNESS ANALYTICS SYSTEM ENGINE", True, (148, 163, 184))
        screen.blit(lbl_title, (40, 35))
        screen.blit(lbl_subtitle, (40, 75))
        
        # State Monitor Display Module
        state_colors = {"START": (56, 189, 248), "CALIBRATING": (234, 179, 8), "ACTIVE": (34, 197, 94), "DETECTED": (239, 68, 68)}
        pygame.draw.rect(screen, (30, 41, 59), (40, 115, 320, 45), border_radius=6)
        pygame.draw.circle(screen, state_colors[game_state], (65, 137), 8)
        lbl_state = font_sub.render(f"ENGINE STATUS: {game_state}", True, (241, 245, 249))
        screen.blit(lbl_state, (85, 127))

        # Render Core Game Analytics Dashboard Fields
        if game_state in ["ACTIVE", "DETECTED"]:
            # Telemetry readout layout frames
            pygame.draw.rect(screen, (15, 23, 42), (40, 180, 450, 260), border_radius=8)
            pygame.draw.rect(screen, (51, 65, 85), (40, 180, 450, 260), width=2, border_radius=8)
            
            lbl_time = font_title.render(f"SURVIVED: {survival_time:.2f}s", True, (248, 250, 252))
            lbl_prec = font_title.render(f"STABILITY: {precision_score:.1f}%", True, (34, 197, 94) if precision_score > 70 else (234, 179, 8))
            screen.blit(lbl_time, (60, 205))
            screen.blit(lbl_prec, (60, 255))
            
            # Draw Dynamic Interactive Sensitivity Data Matrix Row Lines
            lbl_sens = font_metrics.render(f"Multiplier Step Scale:  x{current_sensitivity:.3f}", True, (148, 163, 184))
            screen.blit(lbl_sens, (60, 320))
            
            # Indicator Badges for Random Modifiers Context Pools
            if is_spike_active:
                pygame.draw.rect(screen, (220, 38, 38), (60, 360, 180, 28), border_radius=4)
                lbl_spike = font_metrics.render("⚠️ SENSITIVITY SPIKE", True, (255, 255, 255))
                screen.blit(lbl_spike, (68, 366))
            elif is_fake_safe:
                pygame.draw.rect(screen, (29, 78, 216), (60, 360, 180, 28), border_radius=4)
                lbl_fake = font_metrics.render("🛡️ FAKE SAFE MOMENT", True, (255, 255, 255))
                screen.blit(lbl_fake, (68, 366))
            else:
                pygame.draw.rect(screen, (71, 85, 105), (60, 360, 180, 28), border_radius=4)
                lbl_normal = font_metrics.render("⚡ STEADY PRESSURE", True, (203, 213, 225))
                screen.blit(lbl_normal, (68, 366))

            if extreme_mode:
                pygame.draw.rect(screen, (217, 70, 239), (255, 360, 180, 28), border_radius=4)
                lbl_ext = font_metrics.render("💀 EXTREME MODE", True, (255, 255, 255))
                screen.blit(lbl_ext, (270, 366))

        # Real-Time Mathematical Motion Vector Graphs Visualization Window
        graph_x, graph_y, graph_w, graph_h = 40, 460, 450, 180
        pygame.draw.rect(screen, (15, 23, 42), (graph_x, graph_y, graph_w, graph_h), border_radius=8)
        pygame.draw.rect(screen, (51, 65, 85), (graph_x, graph_y, graph_w, graph_h), width=1, border_radius=8)
        lbl_graph_title = font_metrics.render("Real-time Tracking Motion Delta Divergence", True, (100, 116, 139))
        screen.blit(lbl_graph_title, (graph_x + 15, graph_y + 12))
        
        if len(motion_history) > 1:
            points = []
            max_val = max(max(motion_history), 2.5) # Dynamic scale bound scaling factor
            for idx, val in enumerate(motion_history):
                pt_x = graph_x + int((idx / (max_history_points - 1)) * (graph_w - 30)) + 15
                pt_y = graph_y + graph_h - int((val / max_val) * (graph_h - 50)) - 15
                points.append((pt_x, pt_y))
            pygame.draw.lines(screen, (56, 189, 248), False, points, 2)

        # 5. LIVE MATRIX INTERACTIVE CAMERA OVERLAY DISPLAY PANEL MODULE
        cam_surface = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB), -1))
        cam_surface = pygame.transform.scale(cam_surface, (overlay_w, overlay_h))
        
        # Build Camera Overlay Container Box
        pygame.draw.rect(screen, (15, 23, 42), (overlay_x - 4, overlay_y - 4, overlay_w + 8, overlay_h + 8), border_radius=10)
        screen.blit(cam_surface, (overlay_x, overlay_y))
        pygame.draw.rect(screen, state_colors[game_state], (overlay_x, overlay_y, overlay_w, overlay_h), width=2)
        
        # Render Live Telemetry Strips directly adjacent to tracking view matrix feeds
        panel_y = overlay_y + overlay_h + 20
        pygame.draw.rect(screen, (15, 23, 42), (overlay_x, panel_y, overlay_w, 180), border_radius=8)
        pygame.draw.rect(screen, (51, 65, 85), (overlay_x, panel_y, overlay_w, 180), width=1, border_radius=8)
        
        lbl_breakdown = font_sub.render("TRACKED COMPONENTS DELTA", True, (241, 245, 249))
        screen.blit(lbl_breakdown, (overlay_x + 15, panel_y + 15))
        
        # Render 4 component data bar lines cleanly mapped to visual registers
        comp_labels = [("Face Position Tracker", "face_delta", (56, 189, 248)),
                       ("Chest/Shoulder Link", "pose_delta", (168, 85, 247)),
                       ("Pixel Frame Difference", "frame_diff", (234, 179, 8)),
                       ("Ocular Blink Tracking", "eye_blink", (244, 63, 94))]
                       
        for i, (label_text, key, color) in enumerate(comp_labels):
            row_y = panel_y + 48 + (i * 30)
            lbl_comp = font_metrics.render(label_text, True, (203, 213, 225))
            screen.blit(lbl_comp, (overlay_x + 15, row_y))
            
            # Map values dynamically to data bar dimensions lines safely
            val = motion_metrics[key]
            bar_w = int(min(120, val * 45))
            pygame.draw.rect(screen, (30, 41, 59), (overlay_x + 200, row_y + 2, 120, 10), border_radius=2)
            if bar_w > 0:
                pygame.draw.rect(screen, color, (overlay_x + 200, row_y + 2, bar_w, 10), border_radius=2)

        # 6. APP STATE MENUS RENDERS
        if game_state == "START":
            overlay_menu = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay_menu.fill((15, 23, 42, 220)) # Translucent drop masking filter
            screen.blit(overlay_menu, (0, 0))
            
            pygame.draw.rect(screen, (30, 41, 59), (w//2 - 350, h//2 - 180, 700, 360), border_radius=12)
            pygame.draw.rect(screen, (56, 189, 248), (w//2 - 350, h//2 - 180, 700, 360), width=2, border_radius=12)
            
            lbl_m1 = font_title.render("ENGINE WAITING FOR INITIATION", True, (255, 255, 255))
            lbl_m2 = font_sub.render("Press [SPACEBAR] to perform hardware alignment diagnostics", True, (203, 213, 225))
            lbl_m3 = font_metrics.render("Goal: Remain perfectly static. Even breathing, minor blinking,", True, (148, 163, 184))
            lbl_m4 = font_metrics.render("or subtle shoulder tremors will collapse your precision score matrix.", True, (148, 163, 184))
            
            ext_status = "ACTIVE" if extreme_mode else "INACTIVE"
            lbl_m5 = font_sub.render(f"EXTREME ULTRA-PRECISION SYSTEM: {ext_status} (Press 'E' to toggle)", True, (217, 70, 239))
            
            screen.blit(lbl_m1, (w//2 - lbl_m1.get_width()//2, h//2 - 110))
            screen.blit(lbl_m2, (w//2 - lbl_m2.get_width()//2, h//2 - 40))
            screen.blit(lbl_m3, (w//2 - lbl_m3.get_width()//2, h//2 + 20))
            screen.blit(lbl_m4, (w//2 - lbl_m4.get_width()//2, h//2 + 50))
            screen.blit(lbl_m5, (w//2 - lbl_m5.get_width()//2, h//2 + 110))
            
        elif game_state == "CALIBRATING":
            overlay_menu = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay_menu.fill((15, 23, 42, 140))
            screen.blit(overlay_menu, (0, 0))
            
            # Draw standard clean rotating mathematical loading text components
            cal_elapsed = time.time() - calibration_start
            lbl_c = font_title.render(f"LOCKING BASELINE PROFILE... {3.0 - cal_elapsed:.1f}s", True, (234, 179, 8))
            screen.blit(lbl_c, (w//2 - lbl_c.get_width()//2, h//2 - 20))
            
        elif game_state == "DETECTED":
            overlay_menu = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay_menu.fill((127, 29, 29, 180)) # Red tint masking structure on failure session state termination
            screen.blit(overlay_menu, (0, 0))
            
            pygame.draw.rect(screen, (15, 23, 42), (w//2 - 300, h//2 - 130, 600, 260), border_radius=12)
            pygame.draw.rect(screen, (239, 68, 68), (w//2 - 300, h//2 - 130, 600, 260), width=2, border_radius=12)
            
            lbl_fail1 = font_title.render("💥 STILLNESS BREACHED", True, (239, 68, 68))
            lbl_fail2 = font_sub.render(f"Total Survival Sequence: {survival_time:.2f} Seconds", True, (241, 245, 249))
            lbl_fail3 = font_metrics.render(f"Final Matrix Integrity Score: {precision_score:.1f}%", True, (148, 163, 184))
            lbl_fail4 = font_sub.render("Press [SPACEBAR] to run calibration diagnostics again.", True, (34, 197, 94))
            
            screen.blit(lbl_fail1, (w//2 - lbl_fail1.get_width()//2, h//2 - 80))
            screen.blit(lbl_fail2, (w//2 - lbl_fail2.get_width()//2, h//2 - 10))
            screen.blit(lbl_fail3, (w//2 - lbl_fail3.get_width()//2, h//2 + 25))
            screen.blit(lbl_fail4, (w//2 - lbl_fail4.get_width()//2, h//2 + 75))

        # Bottom UI Action Footers
        lbl_exit_hint = font_sub.render("Press 'Q' to cleanly release tracking device hardware streams.", True, (100, 116, 139))
        screen.blit(lbl_exit_hint, (40, h - 35))

        pygame.display.flip()
        clock.tick(FPS)

# -------------------------------------------------------------------------
# PROCESS ENTRYPOINT EXECUTOR DETECTOR HOOK
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run_game()