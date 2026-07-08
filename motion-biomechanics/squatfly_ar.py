import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import time
import random

# =====================================================================
# AUDIO SYNTHESIS & MANAGEMENT
# =====================================================================
class SoundManager:
    """Handles audio loading and procedural synthesis fallback."""
    def __init__(self):
        # Zero-delay audio configuration as requested
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        
        self.sounds = {}
        self._load_or_synthesize("flap", 400, 0.1, "sine")
        self._load_or_synthesize("point", 800, 0.15, "square")
        self._load_or_synthesize("crash", 150, 0.3, "sawtooth")

    def _load_or_synthesize(self, name, freq, duration, wave_type):
        """Attempts to load a .wav, falls back to generating a waveform via NumPy."""
        try:
            self.sounds[name] = pygame.mixer.Sound(f"{name}.wav")
        except FileNotFoundError:
            # Procedural synthesis fallback
            sample_rate = 44100
            n_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, n_samples, False)
            
            if wave_type == "sine":
                wave = np.sin(2 * np.pi * freq * t)
            elif wave_type == "square":
                wave = np.sign(np.sin(2 * np.pi * freq * t))
            elif wave_type == "sawtooth":
                wave = 2 * (t * freq - np.floor(t * freq + 0.5))
                
            # Fade out to prevent audio clicking
            envelope = np.linspace(1, 0, n_samples)
            wave = wave * envelope
            
            # Convert to 16-bit PCM integer format
            audio_data = np.int16(wave * 32767)
            # Create stereo array (2 channels)
            stereo_data = np.column_stack((audio_data, audio_data))
            
            self.sounds[name] = pygame.sndarray.make_sound(stereo_data)
            self.sounds[name].set_volume(0.3)

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()

# =====================================================================
# KINETIC TRACKING ENGINE (MediaPipe)
# =====================================================================
class PoseTracker:
    """Manages MediaPipe Pose and mathematical squat validation."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.baseline_hip_y = None
        self.is_squatting = False
        self.calibration_start = time.time()
        self.calibration_duration = 2.0  # 2 seconds baseline
        
        # We store history to validate a "rapid" stand up
        self.hip_history = []

    def process(self, frame_rgb):
        results = self.pose.process(frame_rgb)
        flap_triggered = False
        calibrating = (time.time() - self.calibration_start) < self.calibration_duration

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract Hips (23, 24) and Shoulders (11, 12)
            left_hip, right_hip = landmarks[23], landmarks[24]
            left_shoulder, right_shoulder = landmarks[11], landmarks[12]
            
            # Compute center points
            hip_y = (left_hip.y + right_hip.y) / 2.0
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
            
            # Mathematical Normalization
            torso_length = abs(hip_y - shoulder_y)
            if torso_length == 0: torso_length = 0.01 # Prevent Div-By-Zero
            
            if calibrating:
                # Accumulate moving average for a stable baseline
                if self.baseline_hip_y is None:
                    self.baseline_hip_y = hip_y
                else:
                    self.baseline_hip_y = (self.baseline_hip_y * 0.9) + (hip_y * 0.1)
            else:
                # 🛠️ SAFETY CATCH: If calibration ended but the camera never saw you,
                # set the baseline right now to prevent a crash.
                if self.baseline_hip_y is None:
                    self.baseline_hip_y = hip_y

                # Calculate Delta Displacement (positive means hips moved DOWN)
                delta_y = hip_y - self.baseline_hip_y
                normalized_delta = delta_y / torso_length
                
                # Squat Mechanics
                SQUAT_ENTER_THRESHOLD = 0.25 # Hips drop by 25% of torso length
                SQUAT_EXIT_THRESHOLD = 0.10  # Hips return near baseline
                
                if normalized_delta > SQUAT_ENTER_THRESHOLD and not self.is_squatting:
                    self.is_squatting = True
                elif normalized_delta < SQUAT_EXIT_THRESHOLD and self.is_squatting:
                    self.is_squatting = False
                    flap_triggered = True  # The upward impulse!

        return results, flap_triggered, calibrating

# =====================================================================
# ARCADE GAME PHYSICS & ENTITIES
# =====================================================================
class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.gravity = 0.6
        self.flap_power = -10
        self.radius = 20

    def flap(self):
        self.velocity = self.flap_power

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def draw(self, surface):
        # Draw a glowing neon bird
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, (0, 255, 255), (int(self.x), int(self.y)), self.radius + 4, 2)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

class PipeManager:
    def __init__(self, screen_width, screen_height):
        self.pipes = []
        self.width = screen_width
        self.height = screen_height
        self.pipe_width = 80
        self.gap_size = 250
        self.velocity = -6
        self.spawn_timer = 0
        self.spawn_delay = 90  # Frames between spawns

    def update(self):
        self.spawn_timer += 1
        if self.spawn_timer > self.spawn_delay:
            self.spawn_timer = 0
            gap_y = random.randint(150, self.height - 150 - self.gap_size)
            # Store [x, gap_y, passed_flag]
            self.pipes.append([self.width, gap_y, False])

        for pipe in self.pipes:
            pipe[0] += self.velocity

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p[0] + self.pipe_width > 0]

    def draw(self, surface):
        for pipe in self.pipes:
            x, gap_y, _ = pipe
            # Top Pipe
            top_rect = pygame.Rect(x, 0, self.pipe_width, gap_y)
            # Bottom Pipe
            bottom_rect = pygame.Rect(x, gap_y + self.gap_size, self.pipe_width, self.height - gap_y - self.gap_size)
            
            pygame.draw.rect(surface, (57, 255, 20), top_rect, border_radius=8)
            pygame.draw.rect(surface, (57, 255, 20), bottom_rect, border_radius=8)
            
            # Neon Inner Border
            pygame.draw.rect(surface, (255, 255, 255), top_rect, 3, border_radius=8)
            pygame.draw.rect(surface, (255, 255, 255), bottom_rect, 3, border_radius=8)

# =====================================================================
# MAIN APPLICATION ENGINE
# =====================================================================
class SquatFlyApp:
    def __init__(self):
        pygame.init()
        self.base_w, self.base_h = 1280, 720
        # Initialize Dynamic Window
        self.screen = pygame.display.set_mode((self.base_w, self.base_h), pygame.RESIZABLE)
        pygame.display.set_caption("SquatFly AR: Kinetic Arcade Engine")
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("impact", 90)
        self.font_small = pygame.font.SysFont("impact", 40)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.sound_manager = SoundManager()
        self.pose_tracker = PoseTracker()
        
        self.reset_game()

    def reset_game(self):
        self.bird = Bird(300, self.base_h // 2)
        self.pipe_manager = PipeManager(self.base_w, self.base_h)
        self.score = 0
        self.state = "CALIBRATING" # CALIBRATING, PLAYING, GAMEOVER
        self.pose_tracker.calibration_start = time.time()

    def process_collisions(self):
        bird_rect = self.bird.get_rect()
        
        # Ground / Ceiling Collision
        if self.bird.y >= self.base_h or self.bird.y <= 0:
            return True
            
        # Pipe Collisions
        for pipe in self.pipe_manager.pipes:
            x, gap_y, passed = pipe
            top_rect = pygame.Rect(x, 0, self.pipe_manager.pipe_width, gap_y)
            bottom_rect = pygame.Rect(x, gap_y + self.pipe_manager.gap_size, self.pipe_manager.pipe_width, self.base_h)
            
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return True
                
            # Score Tracking
            if x + self.pipe_manager.pipe_width < self.bird.x and not passed:
                pipe[2] = True  # Mark as passed
                self.score += 1
                self.sound_manager.play("point")
                
        return False

    def run(self):
        running = True
        while running:
            # 1. Hardware Events & Exit Hooks
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_SPACE and self.state == "GAMEOVER":
                        self.reset_game()

            # 2. Frame Extraction & AI Processing
            ret, frame = self.cap.read()
            if not ret: continue
            
            # Flip for mirror effect and convert color spaces
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run MediaPipe Analysis
            mp_results, flap_triggered, is_calibrating = self.pose_tracker.process(frame_rgb)

            # Draw Skeleton (Optional visual feedback on the frame)
            if mp_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_rgb, mp_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            # 3. Dynamic Screen Compositing
            # Rotate & Convert to Pygame Surface natively
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            
            # Handle Pygame.RESIZABLE seamlessly
            current_w, current_h = self.screen.get_size()
            scaled_bg = pygame.transform.scale(frame_surface, (current_w, current_h))
            self.screen.blit(scaled_bg, (0, 0))

            # Maintain a standardized logic surface overlay to prevent hitboxes from warping
            logic_surface = pygame.Surface((self.base_w, self.base_h), pygame.SRCALPHA)

            # 4. State Machine Logic
            if is_calibrating:
                self.state = "CALIBRATING"
                text = self.font_large.render("STAND STRAIGHT TO CALIBRATE...", True, (255, 255, 0))
                logic_surface.blit(text, (50, self.base_h // 2))
                
            elif self.state == "CALIBRATING":
                # Transition to playing
                self.state = "PLAYING"
                
            elif self.state == "PLAYING":
                if flap_triggered:
                    self.bird.flap()
                    self.sound_manager.play("flap")
                
                self.bird.update()
                self.pipe_manager.update()
                
                self.pipe_manager.draw(logic_surface)
                self.bird.draw(logic_surface)
                
                if self.process_collisions():
                    self.state = "GAMEOVER"
                    self.sound_manager.play("crash")
                    
                # Draw Neon Score Overlay
                score_text = self.font_large.render(f"SQUATS: {self.score}", True, (0, 255, 255))
                logic_surface.blit(score_text, (50, 50))
                
            elif self.state == "GAMEOVER":
                self.pipe_manager.draw(logic_surface)
                self.bird.draw(logic_surface)
                
                go_text = self.font_large.render("SYSTEM CRASH", True, (255, 50, 50))
                restart_text = self.font_small.render("PRESS SPACE TO REBOOT", True, (255, 255, 255))
                logic_surface.blit(go_text, (self.base_w//2 - 250, self.base_h//2 - 50))
                logic_surface.blit(restart_text, (self.base_w//2 - 180, self.base_h//2 + 50))

            # Scale and composite logic layer onto main resizable window
            scaled_logic = pygame.transform.scale(logic_surface, (current_w, current_h))
            self.screen.blit(scaled_logic, (0, 0))

            pygame.display.flip()
            self.clock.tick(60)

        # 5. Pristine Hardware Cleanup
        self.cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = SquatFlyApp()
    app.run()