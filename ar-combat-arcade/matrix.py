import cv2
import numpy as np
import mediapipe as mp
import multiprocessing as mp_lib
import math
import time
import random
from collections import deque

# Force headless initialization for pygame before importing components 
# to ensure zero OS GUI cross-thread contamination windows open.
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


class ProceduralAudioEngine:
    """
    Generates real-time, zero-dependency game audio layers straight into RAM
    bypassing slow local disk I/O read routines.
    """
    def __init__(self):
        # Initialize pygame audio subsystem at a standardized studio rate
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        self.sample_rate = 44100
        self.tic_sound = self._synthesize_balance_tic()
        self.crash_sound = self._synthesize_terminal_crash()
        
        # Throttling parameter to prevent audio buffer overlapping distortion
        self.last_tic_time = 0.0

    def _synthesize_balance_tic(self) -> pygame.mixer.Sound:
        """Generates a high-precision cybernetic radar blip sound."""
        duration = 0.04  # 40 milliseconds
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # High-pitched frequency sweep line
        frequency_series = np.linspace(880, 1400, len(t))
        wave = np.sin(2 * np.pi * frequency_series * t)
        
        # Exponential attenuation envelope
        envelope = np.exp(-t * 90)
        audio_ints = (wave * envelope * 12000).astype(np.int16)
        
        # Mirror mono wave layout into a 2D Stereo sample matrix
        stereo_matrix = np.column_stack((audio_ints, audio_ints))
        return pygame.sndarray.make_sound(stereo_matrix)

    def _synthesize_terminal_crash(self) -> pygame.mixer.Sound:
        """Synthesizes a crashing white-noise impact blast with low sub-bass."""
        duration = 0.6  # 600 milliseconds
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate randomized digital white noise array
        noise = np.random.uniform(-1.0, 1.0, num_samples)
        
        # Generate heavy low frequency bass impact line
        sub_bass = np.sin(2 * np.pi * np.linspace(90, 30, num_samples) * t)
        
        # Blend structures
        blended_wave = (noise * 0.45) + (sub_bass * 0.55)
        
        # Decay exponential window curve
        envelope = np.exp(-t * 6.5)
        audio_ints = (blended_wave * envelope * 22000).astype(np.int16)
        
        stereo_matrix = np.column_stack((audio_ints, audio_ints))
        return pygame.sndarray.make_sound(stereo_matrix)

    def trigger_tic(self, velocity_magnitude):
        """Dispatches dynamic tic sound modulated by rolling physical velocity."""
        now = time.time()
        # Scale blip frequency with velocity magnitude metrics
        cooldown = max(0.08, 0.45 - (velocity_magnitude * 0.025))
        
        if now - self.last_tic_time > cooldown:
            # Set dynamic gain volume mapping based on velocity excitement
            volume = min(1.0, 0.15 + (velocity_magnitude * 0.04))
            self.tic_sound.set_volume(volume)
            self.tic_sound.play()
            self.last_tic_time = now

    def trigger_crash(self):
        """Fires explosion audio on fallback."""
        self.crash_sound.set_volume(1.0)
        self.crash_sound.play()


class PoseWorker(mp_lib.Process):
    """Isolated OS Process running MediaPipe Pose inference to bypass the GIL."""
    def __init__(self, frame_queue, signal_queue, stop_event):
        super().__init__()
        self.frame_queue = frame_queue
        self.signal_queue = signal_queue
        self.stop_event = stop_event

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        while not self.stop_event.is_set():
            if not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                except Exception:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    l_sh = np.array([landmarks[11].x, landmarks[11].y])
                    r_sh = np.array([landmarks[12].x, landmarks[12].y])
                    l_hp = np.array([landmarks[23].x, landmarks[23].y])
                    r_hp = np.array([landmarks[24].x, landmarks[24].y])

                    shoulder_vector = l_sh - r_sh
                    horizontal_tilt = math.atan2(shoulder_vector[1], shoulder_vector[0])
                    torso_height = np.linalg.norm((l_sh + r_sh)/2 - (l_hp + r_hp)/2)
                    
                    if not self.signal_queue.full():
                        self.signal_queue.put_nowait((horizontal_tilt, torso_height))
                        
        pose.close()


class ControlSmoothingLayer:
    """Filters high-frequency structural jitter using rolling memory layouts."""
    def __init__(self, buffer_size=7):
        self.buffer = deque(maxlen=buffer_size)

    def process(self, raw_tilt):
        self.buffer.append(raw_tilt)
        return sum(self.buffer) / len(self.buffer)


class CustomPhysicsEngine:
    """Vector-force matrix calculation simulating true rolling kinematics."""
    def __init__(self, screen_w, screen_h):
        self.width = screen_w
        self.height = screen_h
        self.reset_ball()
        
        self.k_force = 0.45      
        self.friction = 0.985    
        self.max_velocity = 22.0 
        self.dt = 1.0            

    def reset_ball(self):
        self.ball_x = self.width // 2
        self.ball_y = self.height // 3
        self.vx = 0.0
        self.vy = 0.0
        self.radius = 20

    def update(self, smoothed_tilt, platform_y, platform_length):
        force_x = self.k_force * math.sin(smoothed_tilt)
        
        self.vx += force_x * self.dt
        self.vx *= self.friction
        self.vx = np.clip(self.vx, -self.max_velocity, self.max_velocity)
        
        self.ball_x += self.vx
        
        left_bound = (self.width - platform_length) // 2
        right_bound = (self.width + platform_length) // 2
        
        dx = self.ball_x - (self.width // 2)
        target_y = platform_y + dx * math.tan(smoothed_tilt) - self.radius
        
        if self.ball_x - self.radius < left_bound or self.ball_x + self.radius > right_bound:
            self.vy += 1.5
            self.ball_y += self.vy
            return False 
        else:
            self.ball_y = target_y
            self.vy = 0.0
            return True


class CyberneticGripEngine:
    """Main rendering loop, visual composition pipeline, and window frame blitter."""
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.platform_y = 500
        self.platform_length = 600

        self.audio = ProceduralAudioEngine()
        self.smoother = ControlSmoothingLayer()
        self.physics = CustomPhysicsEngine(self.width, self.height)
        self.trail_buffer = deque(maxlen=20)
        
        self.shake_intensity = 0.0
        self.score = 0
        self.game_active = True

    def render_screen_shake(self, frame):
        if self.shake_intensity > 0.1:
            dx = random.randint(-int(self.shake_intensity), int(self.shake_intensity))
            dy = random.randint(-int(self.shake_intensity), int(self.shake_intensity))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (self.width, self.height))
            self.shake_intensity *= 0.85  
        return frame

    def draw_glowing_lines(self, target_img, p1, p2, color, thickness, glow_radius=15):
        overlay = target_img.copy()
        cv2.line(overlay, p1, p2, color, thickness + glow_radius)
        cv2.GaussianBlur(overlay, (25, 25), 0, dst=overlay)
        cv2.line(overlay, p1, p2, (255, 255, 255), thickness)
        cv2.addWeighted(overlay, 0.6, target_img, 0.4, 0, dst=target_img)

    def execute(self):
        cv2.namedWindow("Sayyam AI Lab: Vector Balance Engine", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sayyam AI Lab: Vector Balance Engine", self.width, self.height)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        frame_q = mp_lib.Queue(maxsize=2)
        signal_q = mp_lib.Queue(maxsize=2)
        stop_event = mp_lib.Event()

        worker = PoseWorker(frame_q, signal_q, stop_event)
        worker.start()

        current_tilt = 0.0
        baseline_torso_height = None
        
        print("[ENGINE] Core booted successfully. Live processing operational with Procedural Audio.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))

            mini_frame = cv2.resize(frame, (320, 240))
            if not frame_q.full():
                frame_q.put(mini_frame)

            if not signal_q.empty():
                try:
                    raw_tilt, raw_height = signal_q.get_nowait()
                    current_tilt = self.smoother.process(-raw_tilt)
                    if baseline_torso_height is None:
                        baseline_torso_height = raw_height
                except Exception:
                    pass

            ui_layer = np.zeros_like(frame)
            frame = cv2.addWeighted(frame, 0.25, ui_layer, 0.75, 0)

            if self.game_active:
                is_balanced = self.physics.update(current_tilt, self.platform_y, self.platform_length)
                
                if is_balanced:
                    self.score += 1
                    
                    # Fire dynamic ticks if ball exhibits noticeable movement vector velocity
                    velocity_mag = abs(self.physics.vx)
                    if velocity_mag > 0.5:
                        self.audio.trigger_tic(velocity_mag)

                    edge_distance = min(
                        abs(self.physics.ball_x - ((self.width - self.platform_length) // 2)),
                        abs(((self.width + self.platform_length) // 2) - self.physics.ball_x)
                    )
                    if edge_distance < 90:
                        self.shake_intensity = max(self.shake_intensity, (90 - edge_distance) * 0.25)
                else:
                    # Capture initialization frame of fall sequence to play the crash sound once
                    if self.physics.vy <= 1.5:
                        self.audio.trigger_crash()

                    if self.physics.ball_y > self.height + 100:
                        self.game_active = False
                        self.shake_intensity = 35.0  

                self.trail_buffer.append((int(self.physics.ball_x), int(self.physics.ball_y)))

            for i in range(1, len(self.trail_buffer)):
                thickness = max(1, int((i / len(self.trail_buffer)) * 12))
                cv2.line(frame, self.trail_buffer[i-1], self.trail_buffer[i], (0, 215, 255), thickness)

            cos_t = math.cos(current_tilt)
            sin_t = math.sin(current_tilt)
            
            p1_x = int(self.width // 2 - (self.platform_length // 2) * cos_t)
            p1_y = int(self.platform_y - (self.platform_length // 2) * sin_t)
            p2_x = int(self.width // 2 + (self.platform_length // 2) * cos_t)
            p2_y = int(self.platform_y + (self.platform_length // 2) * sin_t)
            
            self.draw_glowing_lines(frame, (p1_x, p1_y), (p2_x, p2_y), color=(255, 0, 180), thickness=6, glow_radius=12)

            if self.game_active:
                cv2.circle(frame, (int(self.physics.ball_x), int(self.physics.ball_y)), self.physics.radius, (255, 255, 255), -1)
                cv2.circle(frame, (int(self.physics.ball_x), int(self.physics.ball_y)), self.physics.radius + 4, (0, 215, 255), 2)

            frame = self.render_screen_shake(frame)
            
            cv2.putText(frame, f"SCORE: {self.score}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"TILT: {math.degrees(current_tilt):.1f} DEG", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
            
            if not self.game_active:
                cv2.putText(frame, "CRITICAL SYSTEM FAILURE", (self.width // 2 - 320, self.height // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.putText(frame, "PRESS [R] TO REBOOT MATRIX", (self.width // 2 - 270, self.height // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Sayyam AI Lab: Vector Balance Engine", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and not self.game_active:
                self.physics.reset_ball()
                self.trail_buffer.clear()
                self.score = 0
                self.game_active = True

        stop_event.set()
        worker.join()
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    mp_lib.freeze_support()
    engine = CyberneticGripEngine()
    engine.execute()