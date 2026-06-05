import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
import pygame

class AudioSynthesizer:
    """Procedural zero-dependency audio generator producing synth channels directly in RAM."""
    @staticmethod
    def init_mixer():
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

    @staticmethod
    def generate_sound(frequency, duration_ms, type_wave="sine", noise_mix=0.0):
        sample_rate = 22050
        num_samples = int(sample_rate * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
        
        if type_wave == "sine":
            wave = np.sin(2 * np.pi * frequency * t)
        elif type_wave == "square":
            wave = np.sign(np.sin(2 * np.pi * frequency * t))
        else:
            wave = np.zeros(num_samples)
            
        if noise_mix > 0.0:
            noise = np.random.uniform(-1.0, 1.0, num_samples)
            wave = (1.0 - noise_mix) * wave + noise_mix * noise

        # Apply exponential decay to prevent speaker clicking
        decay = np.exp(-4 * t / (duration_ms / 1000.0))
        wave = wave * decay
        
        # Convert to 16-bit signed integers
        audio_data = np.int16(wave * 32767)
        # Duplicate to 2-D matrix layout to avoid OS Stereo-Mixer Driver Trap issues
        stereo_matrix = np.column_stack((audio_data, audio_data))
        return pygame.sndarray.make_sound(stereo_matrix)

class GameState:
    START = 0
    PLAYING = 1
    GAMEOVER = 2

class Projectile:
    def __init__(self, x, y, vy, is_player=False, color=(0, 255, 255)):
        self.x = x
        self.y = y
        self.vy = vy
        self.is_player = is_player
        self.color = color
        self.radius = 5 if is_player else 8

    def update(self):
        self.y += self.vy

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)

class Enemy:
    def __init__(self, x, y, speed, score_value=100):
        self.x = x
        self.y = y
        self.speed = speed
        self.radius = random.randint(15, 25)
        self.color = (0, 0, 255) # Crimson Threat Core
        self.score_value = score_value

    def update(self):
        self.y += self.speed

    def draw(self, frame):
        # Draw nested pulse boundaries
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius + 4, (0, 75, 180), 2)

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-6, 6)
        self.vy = random.uniform(-6, 6)
        self.life = 1.0
        self.decay = random.uniform(0.04, 0.08)
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay

    def draw(self, frame):
        if self.life > 0:
            r = max(1, int(6 * self.life))
            cv2.circle(frame, (int(self.x), int(self.y)), r, self.color, -1)

class AegisStrikeEngine:
    def __init__(self):
        # Window & Canvas properties
        self.window_name = "Project Aegis-Strike // Sayyam AI Lab"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.cap = cv2.VideoCapture(0)
        self.width = 1280
        self.height = 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Initialize tracking layers
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        
        # Audio Design Matrix Compilation
        AudioSynthesizer.init_mixer()
        self.snd_start = AudioSynthesizer.generate_sound(440, 400, "sine")
        self.snd_shoot = AudioSynthesizer.generate_sound(880, 80, "square")
        self.snd_hit = AudioSynthesizer.generate_sound(120, 150, "sine", noise_mix=0.4)
        self.snd_gameover = AudioSynthesizer.generate_sound(80, 600, "square", noise_mix=0.2)

        # Game Entities and State Properties
        self.current_state = GameState.START
        self.player_x = self.width // 2
        self.player_y = int(self.height * 0.85)
        self.player_radius = 25
        
        self.projectiles = []
        self.enemies = []
        self.particles = []
        
        self.score = 0
        self.combo = 0
        self.difficulty_tier = 1.0
        self.shoot_cooldown = 0
        self.screen_shake = 0
        self.shield_active = False
        
        # Telemetry tracking dictionaries
        self.hand_telemetry = {"Left": None, "Right": None}

    def reset_simulation(self):
        self.player_x = self.width // 2
        self.projectiles.clear()
        self.enemies.clear()
        self.particles.clear()
        self.score = 0
        self.combo = 0
        self.difficulty_tier = 1.0
        self.shoot_cooldown = 0
        self.screen_shake = 0
        self.shield_active = False

    def check_finger_extended(self, hand_lms, tip_idx, pip_idx):
        return hand_lms.landmark[tip_idx].y < hand_lms.landmark[pip_idx].y

    def classify_gesture(self, hand_lms):
        """Differentiates between an Open Palm and a Closed Fist via kinematic checking."""
        extended_fingers = [
            self.check_finger_extended(hand_lms, 8, 6),   # Index
            self.check_finger_extended(hand_lms, 12, 10), # Middle
            self.check_finger_extended(hand_lms, 16, 14), # Ring
            self.check_finger_extended(hand_lms, 20, 18)  # Pinky
        ]
        return "OPEN" if sum(extended_fingers) >= 3 else "FIST"

    def process_hand_telemetry(self, results):
        self.hand_telemetry = {"Left": None, "Right": None}
        self.shield_active = False
        
        if not results.multi_hand_landmarks:
            return

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            # Note: MediaPipe mirrorshandedness. OpenCV flip correction maps Right->Left and Left->Right
            actual_side = "Left" if handedness == "Right" else "Right"
            self.hand_telemetry[actual_side] = hand_landmarks

        # Left Hand Logic: Map X position to player coordinates
        if self.hand_telemetry["Left"]:
            wrist_x = self.hand_telemetry["Left"].landmark[0].x
            # Normalize and constrain inside camera viewport parameters
            self.player_x = int((1.0 - wrist_x) * self.width)
            self.player_x = max(self.player_radius, min(self.width - self.player_radius, self.player_x))

        # Right Hand Logic: Handle Weapon/Shield state switching matrix
        if self.hand_telemetry["Right"]:
            gesture = self.classify_gesture(self.hand_telemetry["Right"])
            if gesture == "OPEN":
                if self.shoot_cooldown == 0:
                    self.projectiles.append(Projectile(self.player_x, self.player_y - 30, -18, is_player=True))
                    self.snd_shoot.play()
                    self.shoot_cooldown = 8 # Framework execution step ticks
            elif gesture == "FIST":
                self.shield_active = True

    def check_start_trigger(self):
        """Validates if both wrists are raised above the center horizontal plane of the screen frame."""
        if self.hand_telemetry["Left"] and self.hand_telemetry["Right"]:
            # Coordinate tracking updates (Y goes from 0 at top to 1 at bottom)
            if self.hand_telemetry["Left"].landmark[0].y < 0.4 and self.hand_telemetry["Right"].landmark[0].y < 0.4:
                self.snd_start.play()
                self.reset_simulation()
                self.current_state = GameState.PLAYING

    def spawn_threats(self):
        # Non-linear probability curves scaling dynamically based on Difficulty metrics
        spawn_chance = 0.02 + (self.difficulty_tier * 0.01)
        if random.random() < min(0.08, spawn_chance):
            sx = random.randint(30, self.width - 30)
            speed = random.uniform(4, 7) * (1.0 + (self.difficulty_tier * 0.1))
            self.enemies.append(Enemy(sx, 0, speed))

    def trigger_explosion(self, x, y, color):
        for _ in range(12):
            self.particles.append(Particle(x, y, color))

    def process_physics_and_collisions(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.screen_shake > 0:
            self.screen_shake -= 1

        # Step updates on entities
        for p in self.projectiles[:]:
            p.update()
            if p.y < 0 or p.y > self.height:
                self.projectiles.remove(p)

        for e in self.enemies[:]:
            e.update()
            if e.y > self.height:
                self.enemies.remove(e)
                self.combo = 0 # Drop combo count if threat leaks past defense bounds

        for part in self.particles[:]:
            part.update()
            if part.life <= 0:
                self.particles.remove(part)

        # Core Collision Intersection Checks
        for e in self.enemies[:]:
            # Enemy vs Shield / Player check
            dist_to_player = math.hypot(e.x - self.player_x, e.y - self.player_y)
            if dist_to_player < (e.radius + self.player_radius + (15 if self.shield_active else 0)):
                self.enemies.remove(e)
                if self.shield_active:
                    self.trigger_explosion(e.x, e.y, (0, 215, 255)) # Shield impact color
                    self.score += int(e.score_value * 0.5)
                    self.snd_hit.play()
                else:
                    self.trigger_explosion(self.player_x, self.player_y, (0, 0, 255))
                    self.screen_shake = 20
                    self.snd_gameover.play()
                    self.current_state = GameState.GAMEOVER
                continue

            # Projectile vs Enemy check
            for p in self.projectiles[:]:
                if p.is_player:
                    dist_to_enemy = math.hypot(p.x - e.x, p.y - e.y)
                    if dist_to_enemy < (p.radius + e.radius):
                        if p in self.projectiles: self.projectiles.remove(p)
                        if e in self.enemies: self.enemies.remove(e)
                        self.trigger_explosion(e.x, e.y, (0, 255, 255))
                        self.combo += 1
                        self.score += e.score_value * self.combo
                        self.snd_hit.play()
                        break

    def render_graphics_pipeline(self, frame):
        # Create additive stark black overlay backing for dynamic neon visual contrast optimization
        black_mask = np.zeros_like(frame)
        
        # Apply screen shake matrix translations if threshold flags remain active
        if self.screen_shake > 0:
            dx = random.randint(-self.screen_shake, self.screen_shake)
            dy = random.randint(-self.screen_shake, self.screen_shake)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (self.width, self.height))

        # Render Active Entities
        for p in self.projectiles: p.draw(black_mask)
        for e in self.enemies: e.draw(black_mask)
        for part in self.particles: part.draw(black_mask)

        # Draw Player Core UI
        if self.current_state == GameState.PLAYING:
            player_color = (0, 215, 255) if self.shield_active else (255, 255, 255)
            cv2.circle(black_mask, (self.player_x, self.player_y), self.player_radius, player_color, -1)
            # Render Extended Structural Shield Ring
            if self.shield_active:
                cv2.circle(black_mask, (self.player_x, self.player_y), self.player_radius + 15, (255, 100, 0), 3)

        # Composite HUD tracking matrices back onto the base web camera layer
        frame = cv2.addWeighted(frame, 0.3, black_mask, 0.7, 0)

        # Context HUD Layer Logic Statements
        if self.current_state == GameState.START:
            cv2.putText(frame, "PROJECT AEGIS-STRIKE", (340, 260), cv2.FONT_HERSHEY_TRIPLEX, 1.8, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "RAISE BOTH HANDS ABOVE SHOULDERS TO COMPILATE ENGINE", (220, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "[Left Hand = Position Navigation | Right Hand: Open Palm = Laser, Fist = Aegis Shield]", (110, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1, cv2.LINE_AA)
        
        elif self.current_state == GameState.PLAYING:
            cv2.putText(frame, f"SCORE: {self.score}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"COMBO: x{self.combo}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"DIFFICULTY LEVEL: {self.difficulty_tier:.2f}", (860, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display tracking state debug lines
            if self.hand_telemetry["Right"]:
                r_gesture = self.classify_gesture(self.hand_telemetry["Right"])
                cv2.putText(frame, f"WEAPON STATE: {r_gesture}", (40, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        elif self.current_state == GameState.GAMEOVER:
            cv2.putText(frame, "SYSTEM CRASH: PLAYER TERMINATED", (240, 300), cv2.FONT_HERSHEY_TRIPLEX, 1.6, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"FINAL SCORE RESUME: {self.score}", (440, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "RAISE BOTH HANDS TO RUN SYSTEM REBOOT", (340, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)

        return frame

    def execute(self):
        prev_time = time.time()
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            # Mirror frame layout parameters to align physical tracking axes smoothly
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Ingest gesture states
            self.process_hand_telemetry(results)

            # Core Execution Router States
            if self.current_state == GameState.START:
                self.check_start_trigger()
            elif self.current_state == GameState.PLAYING:
                # Increment continuous exponential difficulty parameters
                self.difficulty_tier += 0.0008
                self.spawn_threats()
                self.process_physics_and_collisions()
            elif self.current_state == GameState.GAMEOVER:
                self.check_start_trigger()

            # Render UI
            output_frame = self.render_graphics_pipeline(frame)
            cv2.imshow(self.window_name, output_frame)

            # Intercept Escape Key to close stream
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    engine = AegisStrikeEngine()
    engine.execute()