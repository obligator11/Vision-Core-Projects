import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
import math
import random

# --- AUDIO PRE-INIT ---
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.init()

def create_synthesized_sound(freq, duration, wave_type='sine'):
    """Generates mathematical sounds in memory if .wav files are missing."""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if wave_type == 'sine': wave = np.sin(freq * t * 2 * np.pi)
    elif wave_type == 'saw': wave = 2 * (freq * t - np.floor(0.5 + freq * t))
    else: wave = np.random.uniform(-1, 1, len(t))
    envelope = np.exp(-3 * t)
    wave = wave * envelope
    sound_array = np.int16(wave * 32767)
    stereo_array = np.column_stack((sound_array, sound_array))
    return pygame.sndarray.make_sound(stereo_array)

try:
    sound_tick = pygame.mixer.Sound("wall_countdown.wav")
    sound_fail = pygame.mixer.Sound("shatter_fail.wav")
    sound_pass = pygame.mixer.Sound("ding_pass.wav")
except:
    sound_tick = create_synthesized_sound(800, 0.1, 'sine')
    sound_fail = create_synthesized_sound(150, 1.5, 'noise')
    sound_pass = create_synthesized_sound(1200, 1.0, 'sine')
    
# Extra harsh sound for the impossible level
sound_glitch = create_synthesized_sound(200, 2.0, 'saw')

# --- POSE EVALUATOR ---
class PoseEvaluator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.target_poses = {
            "HANDS_UP": [
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, 180),
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, 180),
            ],
            "T_POSE": [
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, 90),
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, 90),
            ],
            "STAR_JUMP": [
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, 140),
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, 140),
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, 160),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, 160)
            ],
            "Y_POSE": [
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, 135),
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, 135)
            ],
            "CACTUS": [
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, 90),
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, 90),
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST, 90),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST, 90)
            ],
            "DIAGONAL": [
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, 135),
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, 45)
            ],
            "CRANE": [ 
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW, 90),
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW, 90),
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, 90)
            ],
            "CROUCH": [ 
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE, 90),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE, 90)
            ],
            # THE IMPOSSIBLE TROLL SHAPE
            "IMPOSSIBLE_GLITCH": [
                # Requires elbows to bend backwards 360 degrees and your head to be attached to your knee
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_SHOULDER, 360),
                (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_ANKLE, 0)
            ]
        }

    def get_angle(self, a, b, c):
        a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
        v1, v2 = a - b, c - b
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: return 0.0
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def evaluate_pose(self, landmarks, pose_name):
        if not landmarks or pose_name not in self.target_poses: return False
        targets = self.target_poses[pose_name]
        
        for p1_idx, p2_idx, p3_idx, target_angle in targets:
            p1, p2, p3 = landmarks.landmark[p1_idx], landmarks.landmark[p2_idx], landmarks.landmark[p3_idx]
            if p1.visibility < 0.4 or p2.visibility < 0.4 or p3.visibility < 0.4:
                return False

            current_angle = self.get_angle(p1, p2, p3)
            margin = max(25.0, target_angle * 0.20) 
            if abs(current_angle - target_angle) > margin:
                return False
        return True

# --- WALL MANAGER ---
class WallManager:
    def __init__(self, screen_w, screen_h):
        self.base_w, self.base_h = 800, 600
        self.duration = 5.0 
        self.elapsed = 0.0
        self.active_pose = "HANDS_UP"
        
        self.wall_surface = pygame.Surface((self.base_w, self.base_h))
        self.magic_pink = (255, 0, 255) 
        self.wall_surface.set_colorkey(self.magic_pink)
        self.wall_surface.set_alpha(190) 
        
    def reset(self, pose_name, new_duration):
        self.elapsed = 0.0
        self.duration = new_duration
        self.active_pose = pose_name
        self._generate_clean_cutout()

    def _generate_clean_cutout(self):
        # Change wall color to Red if it's the impossible prank level
        wall_color = (200, 0, 0) if self.active_pose == "IMPOSSIBLE_GLITCH" else (0, 255, 255)
        self.wall_surface.fill(wall_color)
        
        cx, cy = self.base_w // 2, self.base_h // 2
        cy_offset = 120 if self.active_pose == "CROUCH" else 0
        
        # If it's the troll level, draw a terrifying abstract mess
        if self.active_pose == "IMPOSSIBLE_GLITCH":
            # Head on the floor
            pygame.draw.circle(self.wall_surface, self.magic_pink, (cx, cy + 200), 50)
            # Jagged crazy arms
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 200), (cx - 200, cy - 200), 40)
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 200), (cx + 200, cy - 200), 40)
            # Legs floating in mid air
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx - 150, cy - 100), (cx + 150, cy - 100), 40)
            return

        # Normal Rendering for fair levels
        pygame.draw.circle(self.wall_surface, self.magic_pink, (cx, cy - 180 + cy_offset), 50)
        pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 140 + cy_offset), (cx, cy + 80 + cy_offset), 70)
        
        if self.active_pose == "HANDS_UP":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx - 70, cy - 260), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx + 70, cy - 260), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx - 50, cy + 250), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx + 50, cy + 250), 60) 

        elif self.active_pose == "T_POSE":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx - 200, cy - 100), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx + 200, cy - 100), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx - 50, cy + 250), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx + 50, cy + 250), 60) 

        elif self.active_pose == "STAR_JUMP":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx - 160, cy - 200), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx + 160, cy - 200), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx - 150, cy + 250), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx + 150, cy + 250), 60) 

        elif self.active_pose == "Y_POSE":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx - 120, cy - 220), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx + 120, cy - 220), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx - 50, cy + 250), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx + 50, cy + 250), 60) 

        elif self.active_pose == "CACTUS":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx - 120, cy - 100), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx - 120, cy - 100), (cx - 120, cy - 200), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx + 120, cy - 100), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx + 120, cy - 100), (cx + 120, cy - 200), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx - 50, cy + 250), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx + 50, cy + 250), 60) 

        elif self.active_pose == "DIAGONAL":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx - 150, cy - 200), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx + 150, cy + 50), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx - 50, cy + 250), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx + 50, cy + 250), 60) 

        elif self.active_pose == "CRANE":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx - 200, cy - 100), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100), (cx + 200, cy - 100), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx + 50, cy + 250), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50), (cx - 120, cy + 50), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx - 120, cy + 50), (cx - 120, cy + 180), 60) 

        elif self.active_pose == "CROUCH":
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100 + cy_offset), (cx - 80, cy - 20 + cy_offset), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy - 100 + cy_offset), (cx + 80, cy - 20 + cy_offset), 50) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50 + cy_offset), (cx - 100, cy + 50 + cy_offset), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx - 100, cy + 50 + cy_offset), (cx - 100, cy + 180 + cy_offset), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx, cy + 50 + cy_offset), (cx + 100, cy + 50 + cy_offset), 60) 
            pygame.draw.line(self.wall_surface, self.magic_pink, (cx + 100, cy + 50 + cy_offset), (cx + 100, cy + 180 + cy_offset), 60) 

    def update(self, dt):
        self.elapsed += dt
        return self.elapsed >= self.duration 

    def render(self, screen):
        scale_factor = 0.1 + (self.elapsed / self.duration) ** 2 * 1.9
        scaled_w, scaled_h = int(self.base_w * scale_factor), int(self.base_h * scale_factor)
        scaled_wall = pygame.transform.smoothscale(self.wall_surface, (scaled_w, scaled_h))
        sw, sh = screen.get_size()
        screen.blit(scaled_wall, ((sw - scaled_w) // 2, (sh - scaled_h) // 2))

# --- MAIN GAME ENGINE ---
class GameLoop:
    def __init__(self):
        self.screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
        pygame.display.set_caption("Survival AR: TROLL EDITION")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("impact", 64)

        self.cap = cv2.VideoCapture(0)
        self.pose_evaluator = PoseEvaluator()
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.wall_manager = WallManager(1280, 720)

        self.state = "START_MENU"
        self.level = 1
        self.score = 0
        self.last_tick = 0
        self.result_timer = 0
        self.previous_pose = ""

    def get_difficulty_pose(self):
        """Prevents repetition and forces the impossible prank on Level 4."""
        # ⚠️ THE TROLL LEVEL ⚠️
        if self.level == 4:
            sound_glitch.play()
            return "IMPOSSIBLE_GLITCH"

        # Standard Level Pools
        pool = []
        if self.level <= 2: pool = ["HANDS_UP", "T_POSE"]
        elif self.level == 3: pool = ["STAR_JUMP", "Y_POSE"]
        elif self.level <= 6: pool = ["CACTUS", "DIAGONAL"]
        else: pool = ["CRANE", "CROUCH"]

        # Anti-Repetition: Remove the last pose from the pool so it NEVER repeats twice in a row
        if len(pool) > 1 and self.previous_pose in pool:
            pool.remove(self.previous_pose)
            
        chosen_pose = random.choice(pool)
        self.previous_pose = chosen_pose
        return chosen_pose

    def draw_text(self, text, y_ratio, color=(255,255,255), shadow=True):
        img = self.font.render(text, True, color)
        rect = img.get_rect(center=(self.screen.get_width()//2, int(self.screen.get_height() * y_ratio)))
        if shadow:
            shadow_img = self.font.render(text, True, (0,0,0))
            self.screen.blit(shadow_img, (rect.x + 4, rect.y + 4))
        self.screen.blit(img, rect)

    def is_palm_open(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        fingers_extended = 0
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]: 
            dist_tip = math.hypot(wrist.x - hand_landmarks.landmark[tip].x, wrist.y - hand_landmarks.landmark[tip].y)
            dist_pip = math.hypot(wrist.x - hand_landmarks.landmark[pip].x, wrist.y - hand_landmarks.landmark[pip].y)
            if dist_tip > dist_pip * 1.15: fingers_extended += 1
        return fingers_extended >= 3

    def run(self):
        running = True
        
        while running:
            dt = self.clock.tick(60) / 1000.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q): running = False
                elif event.type == pygame.VIDEORESIZE: self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            ret, frame = self.cap.read()
            if not ret: continue
            
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            pose_results = self.pose_evaluator.pose.process(frame_rgb)
            hand_results = self.mp_hands.process(frame_rgb)

            if pose_results.pose_landmarks:
                self.pose_evaluator.mp_draw.draw_landmarks(frame_rgb, pose_results.pose_landmarks, self.pose_evaluator.mp_pose.POSE_CONNECTIONS)

            surf_array = np.swapaxes(frame_rgb, 0, 1)
            bg_surface = pygame.surfarray.make_surface(surf_array)
            win_w, win_h = self.screen.get_size()
            scale = max(win_w / bg_surface.get_width(), win_h / bg_surface.get_height())
            bg_surface = pygame.transform.smoothscale(bg_surface, (int(bg_surface.get_width() * scale), int(bg_surface.get_height() * scale)))
            self.screen.blit(bg_surface, (0,0))

            palm_detected = any(self.is_palm_open(hl) for hl in hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else False

            # --- GAME STATE MACHINE ---
            if self.state == "START_MENU":
                self.draw_text("SURVIVAL AR", 0.3, (0, 255, 255))
                self.draw_text("Show an OPEN PALM to Start", 0.7, (255, 255, 0))
                
                if palm_detected: 
                    self.level, self.score = 1, 0
                    self.previous_pose = ""
                    self.wall_manager.reset(self.get_difficulty_pose(), 5.0)
                    self.state = "WALL_APPROACHING"

            elif self.state == "WALL_APPROACHING":
                impact = self.wall_manager.update(dt)
                self.wall_manager.render(self.screen)
                
                # Dynamic HUD
                img_score = pygame.font.SysFont("impact", 40).render(f"SCORE: {self.score} | LVL: {self.level}", True, (255,255,0))
                self.screen.blit(img_score, (20, 20))
                
                # Mocking Text for Level 4
                if self.level == 4:
                    self.draw_text("⚠️ AI OVERRIDE: GENERATING IMPOSSIBLE GEOMETRY ⚠️", 0.8, (255, 0, 0))

                if int(self.wall_manager.elapsed) > self.last_tick:
                    if self.level != 4: sound_tick.play()
                    self.last_tick = int(self.wall_manager.elapsed)
                    
                if impact: self.state = "IMPACT_CHECK"

            elif self.state == "IMPACT_CHECK":
                if self.pose_evaluator.evaluate_pose(pose_results.pose_landmarks, self.wall_manager.active_pose):
                    sound_pass.play()
                    self.score += 100
                    self.level += 1
                    self.result_timer = time.time()
                    self.state = "SUCCESS_FLASH"
                else:
                    sound_fail.play()
                    self.result_timer = time.time()
                    self.state = "GAME_OVER"

            elif self.state == "SUCCESS_FLASH":
                self.wall_manager.render(self.screen)
                self.draw_text("CLEAR! +100", 0.4, (0, 255, 0))
                
                if time.time() - self.result_timer > 1.5:
                    new_speed = max(1.5, 5.0 - (self.level * 0.4))
                    self.wall_manager.reset(self.get_difficulty_pose(), new_speed)
                    self.state = "WALL_APPROACHING"

            elif self.state == "GAME_OVER":
                self.wall_manager.render(self.screen)
                
                if self.level == 4:
                    self.draw_text("AI SAYS: NICE TRY, HUMAN. 🤖", 0.4, (255, 0, 0))
                else:
                    self.draw_text(f"CRASH! FINAL SCORE: {self.score}", 0.4, (255, 0, 0))
                
                if time.time() - self.result_timer > 3.0:
                    self.draw_text("Show OPEN PALM to Restart", 0.6, (255, 255, 0))
                    if palm_detected:
                        self.level, self.score = 1, 0
                        self.previous_pose = ""
                        self.wall_manager.reset(self.get_difficulty_pose(), 5.0)
                        self.state = "WALL_APPROACHING"

            pygame.display.flip()

        self.cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = GameLoop()
    app.run()