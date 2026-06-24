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
        # High confidence thresholds to prevent jitter
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.segments = [] # Holds active line segments as numpy arrays

    def process_frame(self, frame, render_rect):
        """
        Extracts pose, maps normalized coordinates to screen space, 
        and updates the dynamic NumPy line segments.
        """
        results = self.pose.process(frame)
        self.segments.clear()
        
        ox, oy, sw, sh = render_rect

        if not results.pose_landmarks:
            return []

        landmarks = results.pose_landmarks.landmark
        drawn_points = {}

        # Map landmarks to Pygame screen space
        for idx, lm in enumerate(landmarks):
            if lm.visibility > 0.5:
                # Direct mapping (frame is already mirrored in the main loop)
                screen_x = int(lm.x * sw) + ox
                screen_y = int(lm.y * sh) + oy
                drawn_points[idx] = np.array([screen_x, screen_y])

        # Create numpy line segments based on pose connections
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in drawn_points and end_idx in drawn_points:
                p1 = drawn_points[start_idx]
                p2 = drawn_points[end_idx]
                self.segments.append((p1, p2))

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
        
        # Spawn logic based on type
        if self.type == "LASER":
            self.rect = pygame.Rect(win_w, random.randint(100, win_h - 100), 150, 20)
            self.vel = np.array([-random.uniform(8, 15), 0])
            self.color = (255, 50, 50)
            
        elif self.type == "BUZZSAW":
            self.rect = pygame.Rect(-100, random.randint(200, win_h - 200), 80, 80)
            self.vel = np.array([random.uniform(5, 10), 0])
            self.color = (255, 150, 0)
            self.origin_y = self.rect.y
            self.time_offset = random.uniform(0, 10)
            
        elif self.type == "ANVIL":
            self.rect = pygame.Rect(random.randint(100, win_w - 100), -100, 100, 80)
            self.vel = np.array([0, random.uniform(10, 18)])
            self.color = (100, 100, 150)

    def update(self):
        # Apply velocity
        if self.type == "BUZZSAW":
            # Sine wave movement
            self.vel[1] = math.sin(time.time() * 5 + self.time_offset) * 8
            
        self.rect.x += int(self.vel[0])
        self.rect.y += int(self.vel[1])
        
        # Kill if off screen
        if (self.rect.right < -200 or self.rect.left > self.win_w + 200 or 
            self.rect.top > self.win_h + 200):
            self.active = False

    def draw(self, surface):
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
        """NumPy/Pygame integrated intersection math."""
        for obs in self.obstacles:
            for p1, p2 in segments:
                # Pygame's clipline natively handles fast AABB/Line intersection
                if obs.rect.clipline(p1[0], p1[1], p2[0], p2[1]):
                    self.trigger_hit_fx()
                    obs.active = False # Destroy on hit to prevent multi-trigger
                    break 

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