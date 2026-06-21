import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import random
import time
import sys
from collections import deque

# Initialize Pygame and the Audio Synthesizer Engine
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Global Configuration and Window Management
BASE_WIDTH, BASE_HEIGHT = 1280, 720
screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("AI Controls Your Body 🤖⚡")
clock = pygame.time.Clock()

# Procedural Math Audio Synthesis Stream System
SAMPLE_RATE = 44100

def generate_sound_buffer(frequency, duration, wave_type='sine', volume=0.3, glitch_mod=0):
    """Synthesizes dynamic scientific mathematical audio wave buffers in real time inside memory arrays."""
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, False)
    
    if wave_type == 'sine':
        data = np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'square':
        data = np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == 'sawtooth':
        data = 2 * (t * frequency - np.floor(t * frequency + 0.5))
    else: # white noise
        data = np.random.uniform(-1, 1, num_samples)
        
    if glitch_mod > 0:
        glitch_mask = (np.sin(2 * np.pi * glitch_mod * t) > 0.7).astype(np.float32)
        data = data * (1.0 - glitch_mask * 0.8)

    fade_len = min(int(num_samples * 0.1), 1000)
    if fade_len > 0:
        envelope = np.ones(num_samples, dtype=np.float32)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        data *= envelope

    scaled_data = (data * volume * 32767).astype(np.int16)
    stereo_data = np.vstack((scaled_data, scaled_data)).T
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_data))

# Pre-generate core game sound channels to secure high performance tracking speeds
sound_background_loop = generate_sound_buffer(110, 2.0, 'sine', volume=0.15)
sound_glitch = generate_sound_buffer(440, 0.25, 'sawtooth', volume=0.25, glitch_mod=35)
sound_warning = generate_sound_buffer(880, 0.15, 'square', volume=0.3)
sound_fail = generate_sound_buffer(80, 0.6, 'sawtooth', volume=0.4)

# Keep the ambient synthesized loop constantly executing on background channel 0
bg_channel = pygame.mixer.Channel(0)
bg_channel.play(sound_background_loop, loops=-1)

# MediaPipe Hands Tracking Engine Context
mp_hands = mp.solutions.hands
hands_estimator = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

class LowPassFilter:
    """Implements a discrete 1st-order exponential low-pass signal filter to crush optical tracking noise."""
    def __init__(self, alpha=0.2): 
        self.alpha = alpha
        self.prev_val = None

    def apply(self, val):
        if self.prev_val is None:
            self.prev_val = val
            return val
        filtered = self.alpha * val + (1.0 - self.alpha) * self.prev_val
        self.prev_val = filtered
        return filtered

class AntiJitterTracker:
    """Clamps coordinates against secondary micro-movements using dynamic deadband thresholds."""
    def __init__(self, threshold=0.005): 
        self.threshold = threshold
        self.anchor = None

    def stabilize(self, current_val):
        if self.anchor is None:
            self.anchor = current_val
            return current_val
        if abs(current_val - self.anchor) > self.threshold:
            self.anchor = current_val
        return self.anchor

# Instantiate structural filters for absolute coordinate smoothing
pose_filter_x = LowPassFilter(alpha=0.2)
jitter_x = AntiJitterTracker(threshold=0.005)

class Obstacle:
    """Defines mathematical bounded targets/walls moving linearly across the viewport space."""
    def __init__(self, x, y, w, h, obs_type="wall", speed=6.0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.obs_type = obs_type # "wall", "ceiling"
        self.speed = speed

    def update(self, delta_time, time_modifier=1.0):
        self.x -= self.speed * 60.0 * delta_time * time_modifier

    def get_rect(self):
        return pygame.Rect(int(self.x), int(self.y), int(self.w), int(self.h))

    def draw(self, surface):
        rect = self.get_rect()
        if self.obs_type == "wall":
            pygame.draw.rect(surface, (255, 65, 54), rect)
            pygame.draw.rect(surface, (255, 120, 100), rect, 3)
        elif self.obs_type == "ceiling":
            pygame.draw.rect(surface, (240, 173, 78), rect)
            pygame.draw.rect(surface, (255, 220, 150), rect, 3)

class GameEngine:
    """Orchestrates kinematics processing, state remap matrices, performance loops, and game mechanics."""
    def __init__(self):
        # Character Kinematics Attributes
        self.char_x = 250.0
        self.char_y = 500.0
        self.char_w = 40
        self.char_h = 70
        self.vel_y = 0.0
        self.is_grounded = True
        self.is_crouching = False
        
        # Ground level layout matching coordinate matrices
        self.ground_y = 550.0

        # Core Remap State Array Architecture
        self.active_mapping_scheme = 0
        self.all_mappings_desc = {
            0: "NOMINAL: LEAN HAND RIGHT -> MOVE RIGHT, LEAN LEFT -> MOVE LEFT, OPEN HAND -> JUMP",
            1: "CHAOTIC INVERSE: LEAN HAND RIGHT -> MOVE LEFT, LEAN LEFT -> MOVE RIGHT",
            2: "HIJACK: OPEN HAND -> CROUCH, CLOSED FIST -> JUMP",
            3: "TEMPORAL FLIP: LEAN HAND RIGHT -> JUMP, OPEN HAND -> MOVE LEFT"
        }
        
        # Chronos Mapping Timers
        self.last_scheme_rotation_timestamp = time.time()
        self.rotation_interval_seconds = 7.0
        self.warning_duration_seconds = 2.0
        self.has_triggered_warning_sound = False
        
        # Difficulty scaling thresholds
        self.game_start_timestamp = time.time()
        self.chaos_index = 1.0 
        self.is_chaos_mode_active = False
        
        # Temporal Slow Motion System Variables
        self.slow_motion_timer = 0.0
        self.time_modifier = 1.0

        # Structural Scoreboard metrics
        self.current_score = 0.0
        self.survival_combo_multiplier = 1.0
        self.high_score = 0
        
        # Visual Rendering Effects States
        self.screen_shake_magnitude = 0.0
        self.frame_flash_intensity = 0
        
        # Smooth horizontal navigation node
        self.smooth_hand_x = 0.5
        
        self.obstacles = []
        self.last_obstacle_spawn_timestamp = time.time()
        self.spawn_delay_seconds = 2.5

    def rotation_scheme_logic(self):
        """Monitors clock timelines to shift human-to-machine input control configurations."""
        now = time.time()
        elapsed = now - self.last_scheme_rotation_timestamp
        time_until_rotation = self.rotation_interval_seconds - elapsed

        if time_until_rotation <= self.warning_duration_seconds:
            self.frame_flash_intensity = int((1.0 - (time_until_rotation / self.warning_duration_seconds)) * 120)
            if not self.has_triggered_warning_sound:
                pygame.mixer.Channel(1).play(sound_warning)
                self.has_triggered_warning_sound = True
        else:
            self.frame_flash_intensity = 0

        if elapsed >= self.rotation_interval_seconds:
            available_pools = [0, 1, 2, 3]
            available_pools.remove(self.active_mapping_scheme)
            self.active_mapping_scheme = random.choice(available_pools)
            
            runtime_duration = now - self.game_start_timestamp
            self.chaos_index = 1.0 + (runtime_duration / 25.0)
            self.rotation_interval_seconds = max(3.5, 8.0 - (runtime_duration / 20.0))
            
            self.is_chaos_mode_active = random.random() < 0.35 if runtime_duration > 15.0 else False
            
            if random.random() < 0.25 and not self.is_chaos_mode_active:
                self.slow_motion_timer = 3.0
                pygame.mixer.Channel(2).play(sound_glitch)
                
            self.last_scheme_rotation_timestamp = now
            self.has_triggered_warning_sound = False
            pygame.mixer.Channel(1).play(sound_glitch)

    def process_body_kinematics(self, landmarks, frame_w, frame_h):
        """Processes hand landmarks using intuitive positional alignment metrics instead of tricky velocities."""
        if not landmarks:
            return

        # Core anchor node extraction
        wrist = landmarks[mp_hands.HandLandmark.WRIST.value]
        
        # Filter and eliminate jitter from horizontal position smoothly
        filtered_x = pose_filter_x.apply(wrist.x)
        self.smooth_hand_x = jitter_x.stabilize(filtered_x)

        # Clear, easy horizontal gesture map (Center threshold boundaries)
        # Raw coordinates are mirrored naturally from camera feeds
        intended_right = self.smooth_hand_x < 0.43
        intended_left = self.smooth_hand_x > 0.57
        
        # Robust distance checking between finger tip and knuckle to deduce Open vs Closed hand state cleanly
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
        index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
        
        # True Euclidean distance calculation
        finger_extension = math.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
        knuckle_base = math.sqrt((index_mcp.x - wrist.x)**2 + (index_mcp.y - wrist.y)**2)
        
        is_hand_open = finger_extension > (knuckle_base * 1.3)

        # Base nominal triggers setup
        move_dir = 0
        wants_jump = False
        wants_crouch = not is_hand_open # Closed fist defaults to Crouch command

        if intended_right: move_dir = 1
        if intended_left: move_dir = -1
        if is_hand_open: wants_jump = True

        # =========================================================================
        # MALICIOUS HAND REMAP ENGINE MATRIX
        # =========================================================================
        if self.active_mapping_scheme == 1:
            # CHAOTIC INVERSE
            move_dir = -move_dir

        elif self.active_mapping_scheme == 2:
            # HIJACK
            if is_hand_open:
                wants_crouch = True
                wants_jump = False
            else:
                wants_jump = True
                wants_crouch = False

        elif self.active_mapping_scheme == 3:
            # TEMPORAL FLIP
            if intended_right:
                wants_jump = True
                move_dir = 0
            if is_hand_open:
                move_dir = -1
                wants_jump = False

        # Translate coordinates seamlessly onto character physics trackers
        speed_factor = 10.0 * self.chaos_index
        if self.is_chaos_mode_active:
            speed_factor *= 1.4

        self.char_x += move_dir * speed_factor
        
        if wants_jump and self.is_grounded:
            self.vel_y = -16.0
            self.is_grounded = False
            
        self.is_crouching = wants_crouch

    def spawn_obstacles(self, screen_w, screen_h):
        now = time.time()
        adjusted_delay = max(1.1, self.spawn_delay_seconds / self.chaos_index)
        
        if now - self.last_obstacle_spawn_timestamp >= adjusted_delay:
            obs_type = random.choice(["wall", "ceiling"])
            speed = (5.5 + random.uniform(0.0, 3.5)) * self.chaos_index
            
            if obs_type == "wall":
                h = random.randint(80, 140)
                y = self.ground_y - h
                w = random.randint(35, 60)
                self.obstacles.append(Obstacle(screen_w, y, w, h, "wall", speed))
            else:
                h = random.randint(100, 160)
                y = self.ground_y - 220
                w = random.randint(40, 70)
                self.obstacles.append(Obstacle(screen_w, y, w, h, "ceiling", speed))
                
            self.last_obstacle_spawn_timestamp = now

    def update_physics(self, delta_time):
        if self.slow_motion_timer > 0.0:
            self.slow_motion_timer -= delta_time
            self.time_modifier = 0.45
        else:
            self.time_modifier = 1.0

        if self.is_crouching:
            self.char_h = 35
        else:
            self.char_h = 70

        if not self.is_grounded:
            self.vel_y += 0.85 * 60.0 * delta_time * self.time_modifier
            self.char_y += self.vel_y * 60.0 * delta_time * self.time_modifier
        
        expected_surface_y = self.ground_y - self.char_h
        if self.char_y >= expected_surface_y:
            self.char_y = expected_surface_y
            self.vel_y = 0.0
            self.is_grounded = True

        self.char_x = max(20.0, min(self.char_x, pygame.display.get_surface().get_width() - 60.0))
        char_rect = pygame.Rect(int(self.char_x), int(self.char_y), self.char_w, self.char_h)

        for obs in self.obstacles[:]:
            obs.update(delta_time, self.time_modifier)
            
            if char_rect.colliderect(obs.get_rect()):
                self.screen_shake_magnitude = 22.0
                pygame.mixer.Channel(3).play(sound_fail)
                self.current_score = max(0.0, self.current_score - 150.0)
                self.survival_combo_multiplier = 1.0
                self.obstacles.remove(obs)
                continue

            if obs.x + obs.w < 0:
                self.obstacles.remove(obs)
                bonus = 100.0 * self.survival_combo_multiplier
                self.current_score += bonus
                self.survival_combo_multiplier += 0.1

        self.current_score += delta_time * 15.0 * self.survival_combo_multiplier
        if self.current_score > self.high_score:
            self.high_score = int(self.current_score)

        if self.screen_shake_magnitude > 0:
            self.screen_shake_magnitude -= delta_time * 45.0
            if self.screen_shake_magnitude < 0:
                self.screen_shake_magnitude = 0.0

    def render_graphics(self, surface, cv_frame):
        offset_x = 0
        offset_y = 0
        if self.screen_shake_magnitude > 0:
            offset_x = random.randint(-int(self.screen_shake_magnitude), int(self.screen_shake_magnitude))
            offset_y = random.randint(-int(self.screen_shake_magnitude), int(self.screen_shake_magnitude))

        w, h = surface.get_size()
        render_canvas = pygame.Surface((w, h))
        render_canvas.fill((20, 20, 28))

        pygame.draw.line(render_canvas, (70, 80, 95), (0, int(self.ground_y)), (w, int(self.ground_y)), 4)

        player_color = (0, 255, 135) if not self.is_crouching else (0, 195, 255)
        player_rect = pygame.Rect(int(self.char_x), int(self.char_y), self.char_w, self.char_h)
        pygame.draw.rect(render_canvas, player_color, player_rect)
        pygame.draw.rect(render_canvas, (255, 255, 255), player_rect, 2)
        pygame.draw.circle(render_canvas, (255, 255, 255), player_rect.center, 5)

        for obs in self.obstacles:
            obs.draw(render_canvas)

        if cv_frame is not None:
            cv_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            cv_rgb = cv2.flip(cv_rgb, 1)
            overlay_h = 160
            overlay_w = int(cv_rgb.shape[1] * (overlay_h / cv_rgb.shape[0]))
            
            pg_img = pygame.image.frombuffer(cv_rgb.tobytes(), cv_rgb.shape[1::-1], "RGB")
            pg_img = pygame.transform.scale(pg_img, (overlay_w, overlay_h))
            
            overlay_rect = pygame.Rect(w - overlay_w - 20, 20, overlay_w, overlay_h)
            render_canvas.blit(pg_img, overlay_rect)
            pygame.draw.rect(render_canvas, (0, 255, 150), overlay_rect, 2)

        hud_font = pygame.font.SysFont("Courier", 20, bold=True)
        scheme_msg = self.all_mappings_desc[self.active_mapping_scheme]
        
        hud_color = (255, 65, 54) if self.active_mapping_scheme != 0 else (0, 255, 135)
        if self.is_chaos_mode_active:
            scheme_msg += " !! CHAOS MODE MAXIMUM ACCELERATION !!"
            hud_color = (255, 0, 128)

        txt_scheme = hud_font.render(scheme_msg, True, hud_color)
        txt_score = hud_font.render(f"SCORE: {int(self.current_score)}", True, (255, 255, 255))
        txt_combo = hud_font.render(f"COMBO: {self.survival_combo_multiplier:.1f}x", True, (255, 220, 100))
        txt_high = hud_font.render(f"HI-SCORE: {self.high_score}", True, (200, 200, 200))
        
        render_canvas.blit(txt_scheme, (20, 20))
        render_canvas.blit(txt_score, (20, 55))
        render_canvas.blit(txt_combo, (20, 85))
        render_canvas.blit(txt_high, (20, 115))

        if self.slow_motion_timer > 0:
            txt_slow = hud_font.render(f"TIME DILATION ACTIVE: {self.slow_motion_timer:.1f}s", True, (0, 195, 255))
            render_canvas.blit(txt_slow, (20, 145))

        surface.blit(render_canvas, (offset_x, offset_y))

        if self.frame_flash_intensity > 0:
            flash_overlay = pygame.Surface((w, h))
            flash_overlay.fill((255, 65, 54))
            flash_overlay.set_alpha(self.frame_flash_intensity)
            surface.blit(flash_overlay, (0, 0))

def execute_main_loop():
    """Bootstraps application runtime variables and kicks off real-time process iteration loops."""
    global screen
    engine = GameEngine()
    
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[CRITICAL ENGINE EXCEPTION] Hardware webcam asset missing or locked.")
        sys.exit(1)

    last_frame_time = time.time()
    is_app_running = True

    while is_app_running:
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        if delta_time > 0.1:
            delta_time = 0.1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_app_running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    is_app_running = False

        success, frame = video_capture.read()
        if success and frame is not None:
            rgb_inference_target = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inference_results = hands_estimator.process(rgb_inference_target)

            if inference_results.multi_hand_landmarks:
                primary_hand_landmarks = inference_results.multi_hand_landmarks[0]
                
                engine.process_body_kinematics(
                    primary_hand_landmarks.landmark, 
                    frame.shape[1], 
                    frame.shape[0]
                )
                
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, 
                    primary_hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

        engine.rotation_scheme_logic()
        engine.spawn_obstacles(screen.get_width(), screen.get_height())
        engine.update_physics(delta_time)
        engine.render_graphics(screen, frame)

        pygame.display.flip()
        clock.tick(60)

    video_capture.release()
    hands_estimator.close()
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    execute_main_loop()