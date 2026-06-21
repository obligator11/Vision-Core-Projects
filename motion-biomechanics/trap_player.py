import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import random
import time
import sys
import threading
from collections import deque

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
BASE_W, BASE_H = 1280, 720
CAMERA_OVERLAY_W = 240
CAMERA_OVERLAY_H = 180
FPS = 60
SAMPLE_RATE = 44100

# Colors (Hex/RGB Mapping)
COLOR_BG = (10, 10, 18)
COLOR_PANEL = (20, 20, 35)
COLOR_TEXT = (240, 240, 255)
COLOR_TRAP_ACTIVE = (255, 40, 40)
COLOR_TRAP_FAKE = (180, 0, 255)
COLOR_TRAP_WARN = (255, 150, 0)
COLOR_PREDICT_ARROW = (0, 255, 150)
COLOR_SKELETON = (0, 180, 255)
COLOR_DANGER_GLOW = (255, 20, 60)

# -------------------------------------------------------------------------
# SYNTHETIC AUDIO GENERATION (No External Dependencies)
# -------------------------------------------------------------------------
def generate_synth_wave(freq, duration, wave_type='sine', volume=0.3):
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, False)
    
    if wave_type == 'sine':
        data = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        data = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == 'sawtooth':
        data = 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        data = np.random.normal(0, 1, num_samples)
        
    envelope = np.exp(-3.5 * t / duration)
    data = data * envelope * volume
    
    audio_data = (data * 32767).astype(np.int16)
    stereo_data = np.column_stack((audio_data, audio_data))
    return pygame.sndarray.make_sound(stereo_data)

# Pre-render Audio assets inside hardware RAM buffers
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

sound_warn = generate_synth_wave(650, 0.15, 'square', volume=0.2)
sound_trigger = generate_synth_wave(120, 0.4, 'sawtooth', volume=0.5)

class BackgroundMusicManager:
    """Dynamically generates and cycles atmospheric audio patterns locally."""
    def __init__(self):
        self.channel = pygame.mixer.Channel(7)
        self.tempo = 0.4  
        self.notes = [55, 58, 62, 65, 55, 58, 69, 67] 
        self.current_idx = 0
        self.last_beat_time = 0

    def update(self, current_time, panic_mode=False):
        beat_delay = self.tempo * 0.5 if panic_mode else self.tempo
        if current_time - self.last_beat_time >= beat_delay:
            freq = 440 * (2 ** ((self.notes[self.current_idx] - 69) / 12.0))
            wave = 'square' if panic_mode else 'sine'
            vol = 0.25 if panic_mode else 0.15
            beat_sound = generate_synth_wave(freq, beat_delay * 0.9, wave, volume=vol)
            self.channel.play(beat_sound)
            self.current_idx = (self.current_idx + 1) % len(self.notes)
            self.last_beat_time = current_time

bg_music = BackgroundMusicManager()

# -------------------------------------------------------------------------
# THREAD-ISOLATED VIDEO CAMERA STREAM
# -------------------------------------------------------------------------
class ThreadedVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.stream.read()
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update_loop, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update_loop(self):
        while self.running:
            grabbed, frame = self.stream.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.stream.release()

# -------------------------------------------------------------------------
# ADAPTIVE AI ENGINE & SYSTEM STATE
# -------------------------------------------------------------------------
class GameEngine:
    def __init__(self):
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.start_time = time.time()
        self.player_center_x = 0.5  
        self.player_center_y = 0.5
        self.player_velocity_x = 0.0
        
        # Complete full-body tracking collections
        self.skeletal_joints = {} 
        
        # Pattern tracking buffers
        self.history_buffer = deque(maxlen=90) 
        self.dodge_history = []  
        
        # AI Profile Tracking Variables
        self.favorite_dodge_dir = "None Detected"
        self.ai_intelligence = 1.0  
        self.panic_mode = False
        
        # Interactive Objects Collection
        self.traps = [] 
        self.last_attack_time = time.time()
        self.attack_cooldown = 2.5  
        self.screen_shake_intensity = 0
        self.predicted_next_x = 0.5

    def extract_and_log_patterns(self):
        """Analyzes recent kinematic trends to update the AI's probabilistic model."""
        if len(self.history_buffer) < 30:
            return
        
        first_frames = list(self.history_buffer)[:10]
        last_frames = list(self.history_buffer)[-10:]
        
        start_avg_x = np.mean([p[0] for p in first_frames])
        end_avg_x = np.mean([p[0] for p in last_frames])
        
        delta_x = end_avg_x - start_avg_x
        
        if abs(delta_x) > 0.05:
            direction = "Left" if delta_x < 0 else "Right"
            self.dodge_history.append(direction)
            if len(self.dodge_history) > 20:
                self.dodge_history.pop(0)
                
            left_count = self.dodge_history.count("Left")
            right_count = self.dodge_history.count("Right")
            if left_count > right_count:
                self.favorite_dodge_dir = "Left"
            elif right_count > left_count:
                self.favorite_dodge_dir = "Right"
            else:
                self.favorite_dodge_dir = "Balanced"

    def deploy_trap_infrastructure(self):
        """AI leverages user behavioral matrices to intelligently predict paths or drop chaos hazards."""
        if self.game_over:
            return
            
        now = time.time()
        cooldown = self.attack_cooldown / (1.0 + (self.ai_intelligence * 0.18))
        if self.panic_mode:
            cooldown *= 0.35  
            
        if now - self.last_attack_time < cooldown:
            return

        self.last_attack_time = now
        is_fake = random.random() < 0.20 and self.ai_intelligence > 1.5
        is_pure_random = random.random() < 0.40
        
        if is_pure_random:
            target_x = random.uniform(0.15, 0.85)
            target_y = random.uniform(0.25, 0.75)
        else:
            bias_offset = 0.0
            if self.favorite_dodge_dir == "Left":
                bias_offset = -0.15
            elif self.favorite_dodge_dir == "Right":
                bias_offset = 0.15
                
            predicted_target_x = self.player_center_x + (self.player_velocity_x * 4.5) + (bias_offset * random.uniform(0.5, 1.2))
            target_x = max(0.1, min(0.9, predicted_target_x))
            target_y = random.uniform(0.2, 0.8)
            
        self.predicted_next_x = target_x
        trap_type = "Fake" if is_fake else "Standard"
        duration = max(1.0, 2.5 - (self.ai_intelligence * 0.08))
        
        new_trap = {
            "x": target_x,
            "y": target_y,
            "radius": random.uniform(0.04, 0.06), 
            "spawn_time": now,
            "warn_duration": max(0.4, 1.2 - (self.ai_intelligence * 0.10)),
            "total_duration": duration,
            "type": trap_type,
            "triggered": False
        }
        self.traps.append(new_trap)
        sound_warn.play()

    def evaluate_collisions(self):
        """Cross-checks all active skeletal joint points. 1 Shot = 1 Kill."""
        if self.game_over:
            return
            
        now = time.time()
        for trap in self.traps:
            if now - trap["spawn_time"] >= trap["warn_duration"] and not trap["triggered"]:
                
                hit_detected = False
                for joint_name, (jx, jy) in self.skeletal_joints.items():
                    dist = math.hypot(jx - trap["x"], jy - trap["y"])
                    if dist < trap["radius"]:
                        hit_detected = True
                        break
                
                if hit_detected:
                    trap["triggered"] = True
                    if trap["type"] == "Standard":
                        sound_trigger.play()
                        self.lives -= 1 
                        self.screen_shake_intensity = 35
                        if self.lives <= 0:
                            self.game_over = True
                    elif trap["type"] == "Fake":
                        self.score += 50

        self.traps = [t for t in self.traps if now - t["spawn_time"] < t["total_duration"]]

# -------------------------------------------------------------------------
# GRAPHICAL PRESENTATION ENGINE
# -------------------------------------------------------------------------
def render_hud_dashboard(surf, game, w, h, font_main):
    panel_rect = pygame.Rect(30, 30, max(400, w - CAMERA_OVERLAY_W - 80), 110)
    pygame.draw.rect(surf, COLOR_PANEL, panel_rect, border_radius=8)
    pygame.draw.rect(surf, (60, 60, 90), panel_rect, 2, border_radius=8)
    
    txt_score = font_main.render(f"SURVIVAL SCORE: {int(game.score)}", True, COLOR_TEXT)
    txt_intel = font_main.render(f"AI INTEL LEVEL: {game.ai_intelligence:.2f}", True, COLOR_PREDICT_ARROW)
    txt_habit = font_main.render(f"DETECTED DODGE HABIT: {game.favorite_dodge_dir}", True, COLOR_TRAP_WARN)
    
    surf.blit(txt_score, (50, 45))
    surf.blit(txt_intel, (50, 75))
    surf.blit(txt_habit, (450, 45))
    
    txt_life_lbl = font_main.render("LIVES:", True, COLOR_TEXT)
    surf.blit(txt_life_lbl, (w - CAMERA_OVERLAY_W - 220, 72))
    for i in range(3):
        color = (255, 40, 40) if i < game.lives else (40, 40, 50)
        heart_x = w - CAMERA_OVERLAY_W - 150 + (i * 35)
        pygame.draw.circle(surf, color, (heart_x, 85), 10)
        pygame.draw.circle(surf, color, (heart_x + 10, 85), 10)
        pygame.draw.polygon(surf, color, [(heart_x - 5, 89), (heart_x + 15, 89), (heart_x + 5, 104)])

# -------------------------------------------------------------------------
# APPLICATION ENTRYPOINT & CORE MAIN LOOP
# -------------------------------------------------------------------------
def main():
    screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
    pygame.display.set_caption("🎮 Trap the Player: Controlled Delay Mode")
    clock = pygame.time.Clock()
    
    font_main = pygame.font.SysFont("Consolas", 24)
    font_big = pygame.font.SysFont("Consolas", 48, bold=True)
    
    game = GameEngine()
    video_feed = ThreadedVideoStream(src=0)
    
    mp_pose = mp.solutions.pose
    pose_tracker = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    running = True
    while running:
        current_w, current_h = screen.get_size()
        now_time = time.time()
        
        if not game.game_over:
            session_elapsed = now_time - game.start_time
            game.ai_intelligence = 1.0 + (session_elapsed * 0.05)
            
            # Panic Mode activates exactly 6 seconds after the script executes
            game.panic_mode = session_elapsed >= 10.0
            
            game.score += (clock.get_time() * 0.05) * (game.ai_intelligence * 0.5)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

        bg_music.update(now_time, panic_mode=game.panic_mode and not game.game_over)

        grabbed, frame = video_feed.read()
        cv_overlay_surface = None
        
        if grabbed and frame is not None:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb_frame)
            
            if results.pose_landmarks and not game.game_over:
                landmarks = results.pose_landmarks.landmark
                
                game.skeletal_joints = {
                    "nose": (landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y),
                    "left_shoulder": (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
                    "right_shoulder": (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
                    "left_elbow": (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y),
                    "right_elbow": (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y),
                    "left_wrist": (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y),
                    "right_wrist": (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y),
                    "left_hip": (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y),
                    "right_hip": (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y),
                    "left_knee": (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y),
                    "right_knee": (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y),
                    "left_ankle": (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y),
                    "right_ankle": (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y)
                }
                
                mid_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2.0
                mid_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2.0
                
                if len(game.history_buffer) > 0:
                    game.player_velocity_x = mid_x - game.history_buffer[-1][0]
                
                game.player_center_x = mid_x
                game.player_center_y = mid_y
                game.history_buffer.append((mid_x, mid_y))
                game.extract_and_log_patterns()
                
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 200, 255), thickness=3, circle_radius=3),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 100), thickness=2)
                )

            frame_resized = cv2.resize(frame, (CAMERA_OVERLAY_W, CAMERA_OVERLAY_H))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_flipped = np.rot90(frame_rgb)
            cv_overlay_surface = pygame.surfarray.make_surface(frame_flipped)
            cv_overlay_surface = pygame.transform.flip(cv_overlay_surface, True, False)

        game.deploy_trap_infrastructure()
        game.evaluate_collisions()

        # -----------------------------------------------------------------
        # GRAPHICAL RENDERING PIPELINE STAGE
        # -----------------------------------------------------------------
        screen.fill(COLOR_BG)
        
        # Grid visual decoration layer
        grid_space = 60
        for x in range(0, current_w, grid_space):
            pygame.draw.line(screen, (22, 22, 40), (x, 0), (x, current_h), 1)
        for y in range(0, current_h, grid_space):
            pygame.draw.line(screen, (22, 22, 40), (0, y), (current_w, y), 1)

        # Draw traps
        for trap in game.traps:
            tx, ty = int(trap["x"] * current_w), int(trap["y"] * current_h)
            rad = int(trap["radius"] * current_w)
            elapsed = now_time - trap["spawn_time"]
            
            if elapsed < trap["warn_duration"]:
                color = COLOR_TRAP_WARN
                thickness = 1 if int(elapsed * 15) % 2 == 0 else 2
                pygame.draw.circle(screen, color, (tx, ty), rad, thickness)
                pygame.draw.circle(screen, color, (tx, ty), int(rad * (elapsed / trap["warn_duration"])), 1)
            else:
                if trap["type"] == "Fake":
                    color = COLOR_TRAP_FAKE
                else:
                    color = COLOR_TRAP_ACTIVE if not trap["triggered"] else (40, 40, 40)
                pygame.draw.circle(screen, color, (tx, ty), rad, 0 if trap["triggered"] else 2)

        # Draw targeted prediction helpers
        if not game.game_over:
            target_x = int(game.predicted_next_x * current_w)
            target_y = current_h - 120
            pygame.draw.circle(screen, COLOR_PREDICT_ARROW, (target_x, target_y), 12, 1)
            pygame.draw.line(screen, COLOR_PREDICT_ARROW, (target_x - 15, target_y), (target_x + 15, target_y), 1)
            pygame.draw.line(screen, COLOR_PREDICT_ARROW, (target_x, target_y - 15), (target_x, target_y + 15), 1)
        
        def get_pt(name):
            return (int(game.skeletal_joints[name][0] * current_w), int(game.skeletal_joints[name][1] * current_h))

        # Render full mirror skeleton bones links matrix
        if len(game.skeletal_joints) > 0:
            try:
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("left_shoulder"), get_pt("right_shoulder"), 3)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("left_shoulder"), get_pt("left_hip"), 3)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("right_shoulder"), get_pt("right_hip"), 3)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("left_hip"), get_pt("right_hip"), 3)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("left_shoulder"), get_pt("left_elbow"), 2)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("left_elbow"), get_pt("left_wrist"), 2)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("right_shoulder"), get_pt("right_elbow"), 2)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("right_elbow"), get_pt("right_wrist"), 2)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("left_hip"), get_pt("left_knee"), 2)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("left_knee"), get_pt("left_ankle"), 2)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("right_hip"), get_pt("right_knee"), 2)
                pygame.draw.line(screen, COLOR_SKELETON, get_pt("right_knee"), get_pt("right_ankle"), 2)
            except KeyError:
                pass

        # Draw joint nodes
        for joint_name, (jx, jy) in game.skeletal_joints.items():
            px, py = int(jx * current_w), int(jy * current_h)
            pygame.draw.circle(screen, COLOR_SKELETON, (px, py), 6, 0)
            pygame.draw.circle(screen, COLOR_TEXT, (px, py), 8, 1)

        # Build HUD Layer
        render_hud_dashboard(screen, game, current_w, current_h, font_main)
        if game.panic_mode and not game.game_over:
            txt_panic = font_big.render("🚨 PANIC MODE ACTIVATED 🚨", True, COLOR_DANGER_GLOW)
            screen.blit(txt_panic, (current_w // 2 - txt_panic.get_width() // 2, current_h - 80))

        # Render Camera Stream
        if cv_overlay_surface:
            cam_x = current_w - CAMERA_OVERLAY_W - 30
            cam_y = 30
            screen.blit(cv_overlay_surface, (cam_x, cam_y))
            pygame.draw.rect(screen, (0, 180, 255), (cam_x, cam_y, CAMERA_OVERLAY_W, CAMERA_OVERLAY_H), 2, border_radius=4)

        # Game Over Monitor Sequence
        if game.game_over:
            black_overlay = pygame.Surface((current_w, current_h))
            black_overlay.fill((0, 0, 0))
            black_overlay.set_alpha(200)
            screen.blit(black_overlay, (0, 0))
            
            txt_dead = font_big.render("❌ SYSTEM TERMINATED: WASTED ❌", True, COLOR_TRAP_ACTIVE)
            txt_hint = font_main.render("Press 'Q' to Exit System Safely", True, COLOR_TEXT)
            screen.blit(txt_dead, (current_w // 2 - txt_dead.get_width() // 2, current_h // 2 - 40))
            screen.blit(txt_hint, (current_w // 2 - txt_hint.get_width() // 2, current_h // 2 + 30))

        # Apply Screen Shake 
        if game.screen_shake_intensity > 0:
            shake_x = random.randint(-game.screen_shake_intensity, game.screen_shake_intensity)
            shake_y = random.randint(-game.screen_shake_intensity, game.screen_shake_intensity)
            shake_surface = pygame.Surface(screen.get_size())
            shake_surface.blit(screen, (shake_x, shake_y))
            screen.blit(shake_surface, (0, 0))
            
            flash = pygame.Surface((current_w, current_h))
            flash.fill(COLOR_DANGER_GLOW)
            flash.set_alpha(int(game.screen_shake_intensity * 4))
            screen.blit(flash, (0, 0))
            game.screen_shake_intensity = max(0, game.screen_shake_intensity - 2)

        pygame.display.flip()
        clock.tick(FPS)

    video_feed.stop()
    pose_tracker.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()