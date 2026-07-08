import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time
import sys
import math

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
BASE_WIDTH, BASE_HEIGHT = 1280, 720
FPS_TARGET = 60
SHAKE_DURATION = 0.3
FLASH_DURATION = 0.2

# ---------------------------------------------------------
# AUDIO MANAGER (Graceful Fallback)
# ---------------------------------------------------------
class AudioManager:
    def __init__(self):
        # Initialize mixer with low-latency buffer settings
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        
        self.sounds = {}
        self.load_sound("bg_loop", "Background_loop.wav", is_bg=True)
        self.load_sound("spawn", "Swoosh.wav")
        self.load_sound("hit", "Buzzer_hit.wav")
        self.load_sound("level_up", "Level_up.wav")

    def load_sound(self, name, filename, is_bg=False):
        """Attempts to load a sound, creates a silent placeholder if missing."""
        try:
            sound = pygame.mixer.Sound(filename)
            if is_bg:
                sound.play(loops=-1)
                sound.set_volume(0.3)
            self.sounds[name] = sound
        except Exception:
            print(f"[AUDIO WARN] Missing {filename}. Using silent placeholder.")
            self.sounds[name] = None

    def play(self, name):
        if self.sounds.get(name):
            self.sounds[name].play()

# ---------------------------------------------------------
# PLAYER SKELETON & TRACKING
# ---------------------------------------------------------
class PlayerSkeleton:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.segments = []
        self.core_rect = None  # NEW: hitbox for collisions

    def process_frame(self, frame, render_rect):
        results = self.pose.process(frame)
        self.segments.clear()
        self.core_rect = None

        ox, oy, sw, sh = render_rect
        if not results.pose_landmarks:
            return []

        landmarks = results.pose_landmarks.landmark
        drawn_points = {}

        for idx, lm in enumerate(landmarks):
            if lm.visibility > 0.5:
                screen_x = int(lm.x * sw) + ox
                screen_y = int(lm.y * sh) + oy
                drawn_points[idx] = np.array([screen_x, screen_y])

        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in drawn_points and end_idx in drawn_points:
                self.segments.append((drawn_points[start_idx], drawn_points[end_idx]))

        # NEW: build a small core hitbox from shoulders + hips (11,12,23,24)
        core_ids = [11, 12, 23, 24]
        core_pts = [drawn_points[i] for i in core_ids if i in drawn_points]
        if len(core_pts) >= 2:
            xs = [p[0] for p in core_pts]
            ys = [p[1] for p in core_pts]
            pad = 20
            self.core_rect = pygame.Rect(
                min(xs) - pad, min(ys) - pad,
                (max(xs) - min(xs)) + pad * 2,
                (max(ys) - min(ys)) + pad * 2
            )

        return self.segments

    def draw(self, surface):
        """Draws the skeleton hitboxes as neon lines."""
        for p1, p2 in self.segments:
            # Draw the line connection between joints
            pygame.draw.line(surface, (0, 255, 255), p1.tolist(), p2.tolist(), 4)
            # Draw the joint nodes
            pygame.draw.circle(surface, (0, 255, 100), p1.tolist(), 5)
            pygame.draw.circle(surface, (0, 255, 100), p2.tolist(), 5)

# ---------------------------------------------------------
# OBSTACLE ENTITIES
# ---------------------------------------------------------
class Obstacle:
    def __init__(self, win_w, win_h):
        self.type = random.choice(["LASER", "BUZZSAW", "ANVIL"])
        self.active = True
        self.win_w = win_w
        self.win_h = win_h
        
        # Default initialization to prevent AttributeError
        self.rect = pygame.Rect(0, 0, 50, 50)
        self.vel = np.array([0.0, 0.0])
        self.color = (255, 255, 255)
        self.time_offset = random.uniform(0, 10)
        
        # Specific spawn logic
        if self.type == "LASER":
            self.rect = pygame.Rect(win_w, random.randint(100, win_h - 100), 150, 20)
            self.vel = np.array([-random.uniform(20, 30), 0])
            self.color = (255, 50, 50)
            
        elif self.type == "BUZZSAW":
            self.rect = pygame.Rect(-100, random.randint(200, win_h - 200), 80, 80)
            self.vel = np.array([random.uniform(12, 18), 0])
            self.color = (255, 150, 0)
            
        elif self.type == "ANVIL":
            self.rect = pygame.Rect(random.randint(100, win_w - 100), -100, 100, 80)
            self.vel = np.array([0, random.uniform(18, 26)])
            self.color = (100, 100, 150)

    def update(self):
        # Apply velocity with sine wave for buzzsaw
        if self.type == "BUZZSAW":
            self.vel[1] = math.sin(time.time() * 5 + self.time_offset) * 8
            
        # Ensure self.rect exists before modifying
        if hasattr(self, 'rect'):
            self.rect.x += int(self.vel[0])
            self.rect.y += int(self.vel[1])
        
        # Kill if off screen
        if (self.rect.right < -200 or self.rect.left > self.win_w + 200 or 
            self.rect.top > self.win_h + 200 or self.rect.bottom > self.win_h + 200):
            self.active = False

    def draw(self, surface):
        if hasattr(self, 'rect'):
            pygame.draw.rect(surface, self.color, self.rect)
            if self.type == "LASER":
                pygame.draw.rect(surface, (255, 200, 200), self.rect.inflate(-10, -10))

# ---------------------------------------------------------
# MAIN GAME ENGINE
# ---------------------------------------------------------
class GameManager:
    def __init__(self):
        pygame.init()
        # Enable dynamic resizing
        self.screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("LivingRoom Ninja: AR Parkour Engine")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("impact", 36)
        
        # Hardware integrations
        self.cap = cv2.VideoCapture(0)
        self.audio = AudioManager()
        self.skeleton = PlayerSkeleton()
        
        # Game State
        self.running = True
        self.obstacles = []
        self.score = 0
        self.start_time = time.time()
        self.last_level_up = self.start_time
        self.next_spawn_time = time.time() + 2.0
        
        # FX State
        self.shake_end = 0
        self.flash_end = 0

    def calculate_render_rect(self, frame_w, frame_h, win_w, win_h):
        """Calculates scaling to maintain webcam aspect ratio on resize."""
        scale = min(win_w / frame_w, win_h / frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        offset_x = (win_w - new_w) // 2
        offset_y = (win_h - new_h) // 2
        return offset_x, offset_y, new_w, new_h

    def handle_collisions(self, segments):
        """Checks only the player's core (torso/hip) hitbox against obstacles."""
        core = self.skeleton.core_rect
        if core is None:
            return
        for obs in self.obstacles:
            if obs.active and obs.rect.colliderect(core):
                self.trigger_hit_fx()
                obs.active = False

    def trigger_hit_fx(self):
        self.audio.play("hit")
        self.score = max(0, self.score - 50)
        curr_time = time.time()
        self.shake_end = curr_time + SHAKE_DURATION
        self.flash_end = curr_time + FLASH_DURATION

    def process_logic(self, win_w, win_h):
        curr_time = time.time()
        
        # Spawn Logic
        if curr_time > self.next_spawn_time:
            self.obstacles.append(Obstacle(win_w, win_h))
            self.audio.play("spawn")
            # Difficulty scaling: spawn faster over time
            spawn_delay = max(0.4, 2.0 - ((curr_time - self.start_time) * 0.01))
            self.next_spawn_time = curr_time + spawn_delay

        # Level up logic
        if curr_time - self.last_level_up >= 30:
            self.audio.play("level_up")
            self.score += 500
            self.last_level_up = curr_time

        # Update entities
        for obs in self.obstacles:
            obs.update()
        self.obstacles = [o for o in self.obstacles if o.active]
        self.score += 1 # Passive score gain

    def run(self):
        while self.running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    self.running = False

            win_w, win_h = self.screen.get_size()

            # 2. Frame Extraction & Formatting
            success, frame = self.cap.read()
            if not success:
                continue

            # MIRROR THE CAMERA INSTANTLY (1 = horizontal flip)
            frame = cv2.flip(frame, 1)

            # Convert BGR (OpenCV) -> RGB -> Pygame Surface
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe processing
            render_rect = self.calculate_render_rect(frame.shape[1], frame.shape[0], win_w, win_h)
            segments = self.skeleton.process_frame(frame_rgb, render_rect)

            # Pygame Surface translation (Swap axes seamlessly aligns OpenCV to Pygame)
            cam_surf = pygame.surfarray.make_surface(np.swapaxes(frame_rgb, 0, 1))
            cam_surf = pygame.transform.scale(cam_surf, (render_rect[2], render_rect[3]))

            # 3. Game Logic & Math
            self.process_logic(win_w, win_h)
            self.handle_collisions(segments)

            # 4. Rendering
            self.screen.fill((10, 10, 15)) # Background borders if aspect ratio differs
            
            # Screen Shake modifier
            sx, sy = 0, 0
            if time.time() < self.shake_end:
                sx = random.randint(-15, 15)
                sy = random.randint(-15, 15)

            # Blit Base Camera
            self.screen.blit(cam_surf, (render_rect[0] + sx, render_rect[1] + sy))

            # Draw Skeletal Hitboxes
            self.skeleton.draw(self.screen)

            # Draw Obstacles
            for obs in self.obstacles:
                obs.draw(self.screen)

            # Red Flash FX
            if time.time() < self.flash_end:
                flash_surf = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
                flash_surf.fill((255, 0, 0, 100))
                self.screen.blit(flash_surf, (0, 0))

            # HUD
            fps = int(self.clock.get_fps())
            score_txt = self.font.render(f"SCORE: {self.score}", True, (255, 255, 255))
            fps_txt = self.font.render(f"FPS: {fps}", True, (0, 255, 255))
            
            self.screen.blit(score_txt, (20, 20))
            self.screen.blit(fps_txt, (win_w - 150, 20))

            # Display flip & tick
            pygame.display.flip()
            self.clock.tick(FPS_TARGET)

        self.cleanup()

    def cleanup(self):
        print("[SHUTDOWN] Releasing hardware gracefully...")
        self.cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = GameManager()
    game.run()