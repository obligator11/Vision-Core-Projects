import cv2
import numpy as np
import pygame
import mediapipe as mp
import time
import random
from collections import deque

# ==========================================
# INITIALIZATION & AUDIO SUBSYSTEM
# ==========================================
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Finger Runner X - Cyber Engine V3")
clock = pygame.time.Clock()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def generate_synth_sound(sound_type):
    sample_rate = 44100
    duration = 0.15 if sound_type in ["jump", "slide"] else (0.4 if sound_type == "hit" else 0.05)
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if sound_type == "jump":
        freq = np.linspace(300, 900, len(t))
        wave = np.sin(2 * np.pi * freq * t) * 0.4
    elif sound_type == "slide":
        freq = np.linspace(700, 150, len(t))
        wave = np.sin(2 * np.pi * freq * t) * np.exp(-5 * t) * 0.5
    elif sound_type == "hit":
        noise = np.random.uniform(-1, 1, len(t))
        envelope = np.exp(-6 * t)
        wave = noise * envelope * 0.7
    else:
        wave = np.sin(2 * np.pi * 880 * t) * np.exp(-20 * t) * 0.1
        
    audio_buffer = np.int16(wave * 32767)
    stereo_buffer = np.column_stack((audio_buffer, audio_buffer))
    return pygame.mixer.Sound(buffer=stereo_buffer)

sound_jump = generate_synth_sound("jump")
sound_slide = generate_synth_sound("slide")
sound_hit = generate_synth_sound("hit")

bg_duration = 2.0
bg_t = np.linspace(0, bg_duration, int(44100 * bg_duration), False)
bg_wave = 0.15 * np.sin(2 * np.pi * 65.41 * bg_t) + 0.05 * np.sin(2 * np.pi * 130.81 * bg_t)
bg_pcm = np.int16(bg_wave * 32767)
bg_stereo = np.column_stack((bg_pcm, bg_pcm))
sound_bg_loop = pygame.mixer.Sound(buffer=bg_stereo)
sound_bg_loop.play(loops=-1)

# NEON PALETTE MATRIX
NEON_CYAN = (0, 255, 255)
NEON_MAGENTA = (255, 0, 255)
NEON_YELLOW = (255, 255, 0)
NEON_ORANGE = (255, 128, 0)
PURE_WHITE = (255, 255, 255)
DARK_BG = (10, 10, 18)
GRID_COLOR = (25, 25, 45)

# ==========================================
# GAME ENGINE DATA STRUCTURES
# ==========================================
class Obstacle:
    def __init__(self, speed_modifier):
        self.lane = random.randint(0, 2)
        # Unique visual properties per hazard classification type
        self.type = random.choice(["JUMP", "SLIDE", "SIDEWAY"])
        self.z = 100.0  # Deep horizon vanishing point origin scale
        self.speed = 1.3 * speed_modifier
        self.active = True

    def update(self, dt):
        self.z -= self.speed * dt * 60
        if self.z <= 0:
            self.active = False

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-14, -3) if color == NEON_MAGENTA else random.uniform(-6, 6)
        self.color = color
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.05)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay

class FingerRunnerXEngine:
    def __init__(self):
        self.w = WIDTH
        self.h = HEIGHT
        self.current_lane = 1
        self.update_lane_geometry()
        
        self.player_state = "RUNNING" 
        self.state_timer = 0.0
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.game_speed = 1.0
        self.run_time = 0.0
        self.game_over = False
        
        # Absolute kinematic history queue trackers
        self.pos_history = deque(maxlen=6)
        self.gesture_cooldown = 0.0
        self.smoothed_finger_x = 0.5
        
        self.screen_shake = 0.0
        self.slow_mo_factor = 1.0
        self.combo_glow = 0.0
        self.obstacles = []
        self.particles = []
        self.motion_trail = deque(maxlen=5)
        self.spawn_timer = 0.0

        self.font_main = pygame.font.SysFont("Courier", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Courier", 54, bold=True)
        self.font_combo = pygame.font.SysFont("Courier", 38, bold=True)

    def update_lane_geometry(self):
        self.lane_xs = [self.w // 6, self.w // 2, 5 * self.w // 6]
        self.player_x = self.lane_xs[self.current_lane]

    def handle_resize(self, new_w, new_h):
        self.w = max(640, new_w)
        self.h = max(480, new_h)
        self.update_lane_geometry()

    def process_vision(self):
        success, frame = cap.read()
        if not success:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb_frame)

        detected_gesture = None
        raw_finger_x = None

        if result.multi_hand_landmarks:
            hand_lms = result.multi_hand_landmarks[0]
            tip = hand_lms.landmark[8]  # Index Fingertip
            raw_finger_x = tip.x
            
            # EMA filter stabilizes pixel jitter noise perfectly
            self.smoothed_finger_x = (self.smoothed_finger_x * 0.4) + (raw_finger_x * 0.6)
            
            t_now = time.time()
            self.pos_history.append((tip.x, tip.y, t_now))

            # Balanced displacement kinematics computation over time slice framework
            if len(self.pos_history) >= 3 and self.gesture_cooldown <= 0:
                oldest = self.pos_history[0]
                latest = self.pos_history[-1]
                dt = latest[2] - oldest[2]

                if dt > 0:
                    # Spatial velocity projections normalized across viewport matrices
                    vel_x = (latest[0] - oldest[0]) / dt
                    vel_y = (latest[1] - oldest[1]) / dt

                    # Re-calibrated velocity window parameters for seamless responsive caps
                    if vel_y < -1.4: 
                        detected_gesture = "SWIPE_UP"
                    elif vel_y > 1.4:
                        detected_gesture = "SWIPE_DOWN"

            mp_draw = mp.solutions.drawing_utils
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2))
        else:
            self.pos_history.clear()

        # Command mapping deployment logic structures
        if detected_gesture and self.player_state == "RUNNING" and not self.game_over:
            self.gesture_cooldown = 0.25
            if detected_gesture == "SWIPE_UP":
                self.player_state = "JUMPING"
                self.state_timer = 0.55
                sound_jump.play()
            elif detected_gesture == "SWIPE_DOWN":
                self.player_state = "SLIDING"
                self.state_timer = 0.55
                sound_slide.play()

        # Mirror alignment tracking loops
        if raw_finger_x is not None and not self.game_over:
            if self.smoothed_finger_x < 0.35:
                self.current_lane = 0
            elif self.smoothed_finger_x > 0.65:
                self.current_lane = 2
            else:
                self.current_lane = 1

        resized_cam = cv2.resize(frame, (320, 240))
        return cv2.cvtColor(resized_cam, cv2.COLOR_BGR2RGB)

    def update_physics(self, dt):
        if self.game_over:
            if self.screen_shake > 0:
                self.screen_shake -= dt * 2
            return

        self.run_time += dt
        self.gesture_cooldown = max(0.0, self.gesture_cooldown - dt)
        self.game_speed = 1.0 + (self.run_time * 0.015)
        
        target_x = self.lane_xs[self.current_lane]
        self.player_x += (target_x - self.player_x) * 16 * dt

        self.motion_trail.append((self.player_x, self.get_player_y()))

        if self.player_state != "RUNNING":
            self.state_timer -= dt * self.slow_mo_factor
            if self.state_timer <= 0:
                self.player_state = "RUNNING"

        self.spawn_timer += dt * self.game_speed
        if self.spawn_timer > random.uniform(1.6, 2.5):
            self.obstacles.append(Obstacle(self.game_speed))
            self.spawn_timer = 0.0

        self.slow_mo_factor = 1.0
        near_miss_detected = False

        for obs in self.obstacles:
            obs.update(dt * self.slow_mo_factor)
            
            if 10.0 < obs.z < 26.0 and obs.lane == self.current_lane:
                if (obs.type == "JUMP" and self.player_state != "SLIDING") or \
                   (obs.type == "SLIDE" and self.player_state != "JUMPING") or \
                   (obs.type == "SIDEWAY"):
                    near_miss_detected = True

            if obs.z <= 4.0 and obs.active:
                if obs.lane == self.current_lane:
                    collided = False
                    if obs.type == "JUMP" and self.player_state != "SLIDING":
                        collided = True
                    elif obs.type == "SLIDE" and self.player_state != "JUMPING":
                        collided = True
                    elif obs.type == "SIDEWAY":
                        # SIDEWAY spikes can only be avoided by switching out of the lane entirely
                        collided = True

                    if collided:
                        sound_hit.play()
                        self.game_over = True
                        self.screen_shake = 1.5
                        for _ in range(40):
                            self.particles.append(Particle(self.player_x, self.get_player_y(), NEON_MAGENTA))
                        obs.active = False
                    else:
                        self.score += 100 * (1 + self.combo // 5)
                        self.combo += 1
                        self.max_combo = max(self.max_combo, self.combo)
                        self.combo_glow = 1.0
                        for _ in range(12):
                            self.particles.append(Particle(self.player_x, self.get_player_y(), NEON_YELLOW))
                        obs.active = False
                else:
                    # Clear out obstacle scoring matrices safely if passed successfully in another lane
                    obs.active = False

        if near_miss_detected:
            self.slow_mo_factor = 0.35

        self.obstacles = [o for o in self.obstacles if o.active]

        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

        if self.combo_glow > 0:
            self.combo_glow -= dt * 2

    def get_player_y(self):
        base_y = self.h - 130
        if self.player_state == "JUMPING":
            normalized_t = (0.55 - self.state_timer) / 0.55
            height_offset = 170 * np.sin(normalized_t * np.pi)
            return base_y - height_offset
        elif self.player_state == "SLIDING":
            return base_y + 35
        return base_y

    def draw_scene(self, cam_surf):
        dx, dy = 0, 0
        if self.screen_shake > 0:
            dx = int(random.uniform(-16, 16) * self.screen_shake)
            dy = int(random.uniform(-16, 16) * self.screen_shake)

        screen.fill(DARK_BG)

        horizon_y = self.h // 2.5
        pygame.draw.line(screen, GRID_COLOR, (0, horizon_y), (self.w, horizon_y), 2)
        
        # Pure Linear Grid Projection Structure: Lanes run direct, parallel and straight forward
        lane_widths = self.w // 3
        for i in range(4):
            x_line = i * lane_widths
            pygame.draw.line(screen, GRID_COLOR, (x_line + dx, horizon_y), (x_line + dx, self.h + dy), 2)

        grid_step = 60
        offset_y = int((pygame.time.get_ticks() * 0.15 * self.game_speed) % grid_step)
        current_y = horizon_y + offset_y
        while current_y < self.h:
            pygame.draw.line(screen, GRID_COLOR, (0 + dx, current_y + dy), (self.w + dx, current_y + dy), 1)
            current_y += grid_step

        # Draw Hazards Layout Vectors using Straight Projection Scaling Maps
        for obs in self.obstacles:
            ratio = (100.0 - obs.z) / 100.0
            # Scale geometric size cleanly relative to perspective proximity depths
            scale = 0.1 + 0.9 * ratio
            
            obs_w = int(120 * scale) + 15
            obs_h = int(90 * scale) + 15
            
            # Straight forward horizontal layout alignment structure mapping paths directly
            lane_center_x = self.lane_xs[obs.lane]
            obs_x = lane_center_x - obs_w // 2
            obs_y = horizon_y + (self.h - horizon_y - 120) * ratio + 30

            # Dynamic structural variation rendering schemas across obstacle categories
            if obs.type == "JUMP":
                # High energy grid wall (Must slide down under it)
                color = NEON_CYAN
                rect_shape = pygame.Rect(obs_x + dx, obs_y - obs_h + dy, obs_w, obs_h)
                pygame.draw.rect(screen, color, rect_shape, 0, border_radius=4)
                pygame.draw.rect(screen, PURE_WHITE, rect_shape, 2, border_radius=4)
                # Drawing visual signifiers inside the block mesh layout parameters
                pygame.draw.line(screen, DARK_BG, (obs_x + dx, obs_y - obs_h // 2 + dy), (obs_x + obs_w + dx, obs_y - obs_h // 2 + dy), 3)
            elif obs.type == "SLIDE":
                # Low ground barrier (Must jump up over it)
                color = NEON_MAGENTA
                rect_shape = pygame.Rect(obs_x + dx, obs_y - (obs_h // 2) + dy, obs_w, obs_h // 2)
                pygame.draw.rect(screen, color, rect_shape, 4, border_radius=4)
                pygame.draw.rect(screen, PURE_WHITE, rect_shape, 1, border_radius=4)
                # Draw sharp upward chevron mesh vectors
                pygame.draw.lines(screen, NEON_MAGENTA, False, [
                    (obs_x + 5 + dx, obs_y + dy), 
                    (obs_x + obs_w // 2 + dx, obs_y - obs_h // 3 + dy), 
                    (obs_x + obs_w - 5 + dx, obs_y + dy)
                ], 3)
            else:
                # SIDEWAY laser spire core block layout profiles (Must change lane completely)
                color = NEON_ORANGE
                rect_shape = pygame.Rect(obs_x + obs_w // 4 + dx, obs_y - obs_h + dy, obs_w // 2, obs_h)
                pygame.draw.rect(screen, color, rect_shape, 0, border_radius=8)
                pygame.draw.rect(screen, PURE_WHITE, rect_shape, 2, border_radius=8)
                # Outer flashing side safety parameters indicators design bloom lines array
                pygame.draw.circle(screen, NEON_YELLOW, (obs_x + obs_w // 2 + dx, obs_y - obs_h // 2 + dy), int(8 * scale) + 2)

        if len(self.motion_trail) > 1:
            for idx, pos in enumerate(self.motion_trail):
                alpha = int((idx / len(self.motion_trail)) * 110)
                trail_surf = pygame.Surface((80, 80), pygame.SRCALPHA)
                pygame.draw.circle(trail_surf, (0, 255, 255, alpha), (40, 40), 26 - (len(self.motion_trail) - idx) * 2)
                screen.blit(trail_surf, (pos[0] - 40 + dx, pos[1] - 40 + dy))

        p_x = int(self.player_x)
        p_y = int(self.get_player_y())
        
        player_color = NEON_YELLOW if self.player_state != "RUNNING" else NEON_CYAN
        if self.player_state == "SLIDING":
            pygame.draw.ellipse(screen, player_color, (p_x - 42 + dx, p_y - 15 + dy, 84, 40))
            pygame.draw.ellipse(screen, PURE_WHITE, (p_x - 42 + dx, p_y - 15 + dy, 84, 40), 2)
        else:
            pygame.draw.circle(screen, player_color, (p_x + dx, p_y + dy), 28)
            pygame.draw.circle(screen, PURE_WHITE, (p_x + dx, p_y + dy), 28, 3)
            pygame.draw.circle(screen, PURE_WHITE, (p_x + dx, p_y - 8 + dy), 6)

        for p in self.particles:
            alpha_color = [int(c * p.life) for c in p.color]
            pygame.draw.circle(screen, alpha_color, (int(p.x) + dx, int(p.y) + dy), int(6 * p.life))

        score_txt = self.font_main.render(f"SYSTEM DATA MATRIX: {self.score:06d}", True, PURE_WHITE)
        speed_txt = self.font_main.render(f"VEL VELOCITY MULTIPLIER: {self.game_speed:.2f}x", True, NEON_CYAN)
        screen.blit(score_txt, (30, 30))
        screen.blit(speed_txt, (30, 65))

        if self.combo > 0:
            glow_intensity = int(abs(np.sin(pygame.time.get_ticks() * 0.01)) * 155) + 100 if self.combo_glow > 0 else 200
            combo_color = (glow_intensity, glow_intensity, 0) if self.combo >= 10 else NEON_MAGENTA
            combo_txt = self.font_combo.render(f"COMBO STACK: {self.combo}x", True, combo_color)
            screen.blit(combo_txt, (30, 110))

        if self.game_over:
            over_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            over_surf.fill((10, 5, 20, 210))
            screen.blit(over_surf, (0, 0))
            
            msg_txt = self.font_huge.render("CRITICAL HARDWARE FAULT", True, NEON_MAGENTA)
            sub_txt = self.font_main.render(f"FINAL MATRIX TRANSITION SCORE: {self.score}  |  MAX STACK: {self.max_combo}", True, PURE_WHITE)
            hint_txt = self.font_main.render("FLASH OPEN PALM IN CENTER OR TYPE 'R' TO REBOOT TERMINAL", True, NEON_CYAN)
            
            screen.blit(msg_txt, (self.w // 2 - msg_txt.get_width() // 2, self.h // 2 - 80))
            screen.blit(sub_txt, (self.w // 2 - sub_txt.get_width() // 2, self.h // 2 + 10))
            screen.blit(hint_txt, (self.w // 2 - hint_txt.get_width() // 2, self.h // 2 + 60))

        cam_pygame = pygame.image.fromstring(cam_surf.tobytes(), cam_surf.shape[1::-1], "RGB")
        pygame.draw.rect(screen, NEON_MAGENTA, (self.w - 320 - 28, 28, 320 + 4, 240 + 4), 3, border_radius=4)
        screen.blit(cam_pygame, (self.w - 320 - 30, 30))

        pygame.display.flip()

    def reset_engine(self):
        self.obstacles.clear()
        self.particles.clear()
        self.motion_trail.clear()
        self.current_lane = 1
        self.update_lane_geometry()
        self.score = 0
        self.combo = 0
        self.game_speed = 1.0
        self.run_time = 0.0
        self.game_over = False
        self.player_state = "RUNNING"
        self.screen_shake = 0.0

def main():
    engine = FingerRunnerXEngine()
    last_time = time.time()
    is_running = True

    while is_running:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            elif event.type == pygame.VIDEORESIZE:
                engine.handle_resize(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    is_running = False
                elif event.key == pygame.K_r and engine.game_over:
                    engine.reset_engine()

        cam_frame_data = engine.process_vision()

        if engine.game_over and len(engine.pos_history) == engine.pos_history.maxlen:
            if 0.4 < engine.smoothed_finger_x < 0.6:
                engine.reset_engine()

        engine.update_physics(dt)
        engine.draw_scene(cam_frame_data)

        clock.tick(60)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()