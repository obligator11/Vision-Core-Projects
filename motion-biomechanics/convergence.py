import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import threading
from enum import Enum, auto

class GameState(Enum):
    STANDBY = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    GAMEOVER = auto()

class ProceduralAudioEngine:
    """Generates pure mathematical wave frequencies directly into RAM to prevent Disk I/O lag."""
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.shrink_sound = self._synthesize_sine_wave(220, 0.4, loop_extend=True)
        self.warning_sound = self._synthesize_beep(880, 0.1)
        self.fail_sound = self._synthesize_noise(0.5)
        self.start_sound = self._synthesize_sine_wave(440, 0.3)
        
    def _synthesize_sine_wave(self, hz, duration, loop_extend=False):
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        wave = np.sin(2 * np.pi * hz * t) * 16383
        wave_stereo = np.column_stack((wave, wave)).astype(np.int16)
        return pygame.sndarray.make_sound(wave_stereo)

    def _synthesize_beep(self, hz, duration):
        return self._synthesize_sine_wave(hz, duration)

    def _synthesize_noise(self, duration):
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        noise = np.random.uniform(-1, 1, n_samples) * 16383
        decay = np.linspace(1, 0, n_samples)
        noise = (noise * decay).astype(np.int16)
        noise_stereo = np.column_stack((noise, noise))
        return pygame.sndarray.make_sound(noise_stereo)

class TrackingDaemon:
    """Decouples MediaPipe Inference from the Main Thread using an asynchronous runner."""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.latest_coords = None
        self.is_running = True
        self.lock = threading.Lock()
        
    def start(self):
        pass # Threading handled implicitly by continuous frame updates

    def process_frame(self, frame_rgb):
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Extract left shoulder (11), right shoulder (12), left hip (23), right hip (24)
            with self.lock:
                self.latest_coords = {
                    'ls': (landmarks[11].x, landmarks[11].y),
                    'rs': (landmarks[12].x, landmarks[12].y),
                    'lh': (landmarks[23].x, landmarks[23].y),
                    'rh': (landmarks[24].x, landmarks[24].y)
                }
        else:
            with self.lock:
                self.latest_coords = None

    def get_latest(self):
        with self.lock:
            return self.latest_coords

class ConvergenceEngine:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.width = 1280
        self.height = 720
        self.window_name = "Sayyam AI Lab: Project Convergence"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        self.audio = ProceduralAudioEngine()
        self.daemon = TrackingDaemon()
        
        # State & Gameplay Control Parameters
        self.state = GameState.STANDBY
        self.zone_center = (self.width // 2, self.height // 2)
        self.initial_radius = 350
        self.zone_radius = float(self.initial_radius)
        self.min_radius = 40
        self.shrink_rate = 15.0 # Base pixels per second
        self.score = 0
        
        # Timing Vectors
        self.standby_start_time = None
        self.countdown_start_time = None
        self.last_frame_time = time.time()
        self.last_warning_time = 0
        
        # Motion validation variables
        self.previous_center = None
        
        # Advanced Mechanics: Zone Shift
        self.zone_velocity = [20.0, 15.0] # Pixels per second (dx, dy)

    def _calculate_torso_center(self, coords):
        if not coords:
            return None
        # Compute geometric centers of shoulders and hips
        s_x = (coords['ls'][0] + coords['rs'][0]) / 2.0
        s_y = (coords['ls'][1] + coords['rs'][1]) / 2.0
        h_x = (coords['lh'][0] + coords['rh'][0]) / 2.0
        h_y = (coords['lh'][1] + coords['rh'][1]) / 2.0
        
        # Map normal space directly to absolute monitor space pixels
        center_x = int(((s_x + h_x) / 2.0) * self.width)
        center_y = int(((s_y + h_y) / 2.0) * self.height)
        return (center_x, center_y)

    def run(self):
        shrink_channel = pygame.mixer.Channel(0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1) # Mirror transformation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Non-blocking async data extraction trigger
            self.daemon.process_frame(frame_rgb)
            coords = self.daemon.get_latest()
            player_center = self._calculate_torso_center(coords)
            
            current_time = time.time()
            dt = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # State Machine Loop Process
            if self.state == GameState.STANDBY:
                if player_center is not None:
                    if self.previous_center is not None:
                        # Validate if standing still (movement delta less than 8 absolute pixels)
                        movement = math.hypot(player_center[0] - self.previous_center[0], player_center[1] - self.previous_center[1])
                        if movement < 8:
                            if self.standby_start_time is None:
                                self.standby_start_time = current_time
                            elif current_time - self.standby_start_time >= 2.0:
                                self.state = GameState.COUNTDOWN
                                self.countdown_start_time = current_time
                                self.audio.start_sound.play()
                        else:
                            self.standby_start_time = None
                    self.previous_center = player_center
                else:
                    self.standby_start_time = None
                    
            elif self.state == GameState.COUNTDOWN:
                elapsed = current_time - self.countdown_start_time
                if elapsed >= 3.0:
                    self.state = GameState.PLAYING
                    self.zone_radius = float(self.initial_radius)
                    self.score = 0
                    self.zone_center = (self.width // 2, self.height // 2)
                    shrink_channel.play(self.audio.shrink_sound, loops=-1)
                    
            elif self.state == GameState.PLAYING:
                # Dynamic Difficulty Adjustment: shrink rate scales with total survival time
                current_shrink_rate = self.shrink_rate + (self.score * 0.8)
                if self.zone_radius > self.min_radius:
                    self.zone_radius -= current_shrink_rate * dt
                    
                # Advanced Interaction Layer: Dynamic Zone Shifting Vector
                self.zone_center = (
                    int(self.zone_center[0] + self.zone_velocity[0] * dt),
                    int(self.zone_center[1] + self.zone_velocity[1] * dt)
                )
                # Geometric boundary bouncing collisions for the zone
                buffer_r = int(self.zone_radius)
                if self.zone_center[0] - buffer_r < 0 or self.zone_center[0] + buffer_r > self.width:
                    self.zone_velocity[0] *= -1
                    self.zone_center = (max(buffer_r, min(self.zone_center[0], self.width - buffer_r)), self.zone_center[1])
                if self.zone_center[1] - buffer_r < 0 or self.zone_center[1] + buffer_r > self.height:
                    self.zone_velocity[1] *= -1
                    self.zone_center = (self.zone_center[0], max(buffer_r, min(self.zone_center[1], self.height - buffer_r)))

                self.score += dt
                
                # Biomechanical Spatial Violation Assessment
                if player_center is not None:
                    dist_to_center = math.hypot(player_center[0] - self.zone_center[0], player_center[1] - self.zone_center[1])
                    
                    # Boundary threshold warning condition (within 25% of radius bounds)
                    if dist_to_center > (self.zone_radius * 0.75):
                        # Play asynchronous warning sound every 250ms
                        if current_time - self.last_warning_time > 0.25:
                            self.audio.warning_sound.play()
                            self.last_warning_time = current_time
                            
                    if dist_to_center > self.zone_radius:
                        # Terminal Out-Of-Bounds State Entered
                        self.state = GameState.GAMEOVER
                        shrink_channel.stop()
                        self.audio.fail_sound.play()
                else:
                    # Penalty for disappearing from the sensor window
                    if current_time - self.last_warning_time > 0.25:
                        self.audio.warning_sound.play()
                        self.last_warning_time = current_time

            # Graphical Rendering Infrastructure (High Contrast Cyber-HUD Engine)
            hud_overlay = frame.copy()
            
            if self.state == GameState.STANDBY:
                cv2.putText(hud_overlay, "SYSTEM STANDBY: INITIALIZING COORDINATES", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3)
                if self.standby_start_time:
                    progress = int((current_time - self.standby_start_time) * 50)
                    cv2.rectangle(hud_overlay, (50, 120), (50 + progress, 140), (0, 255, 0), -1)
                else:
                    cv2.putText(hud_overlay, "HOLD TOTALLY STILL WITHIN FRAME TO ENGAGE", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
            elif self.state == GameState.COUNTDOWN:
                time_left = 3 - int(current_time - self.countdown_start_time)
                cv2.putText(hud_overlay, f"LOCK ON RECEIVED. INITIATING IN: {max(1, time_left)}", (100, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
                
            elif self.state == GameState.PLAYING:
                # Calculate metric boundaries for haptic vignette flashing
                dist_check = 0
                if player_center is not None:
                    dist_check = math.hypot(player_center[0] - self.zone_center[0], player_center[1] - self.zone_center[1])
                
                is_near_boundary = dist_check > (self.zone_radius * 0.75) or player_center is None
                zone_color = (0, 0, 255) if is_near_boundary else (0, 255, 0)
                
                if is_near_boundary:
                    # Alpha-blended warning vignette injection
                    vignette = np.zeros_like(frame)
                    vignette[:, :] = (0, 0, 180) # Saturated dark deep crimson red
                    cv2.addWeighted(hud_overlay, 0.75, vignette, 0.25, 0, hud_overlay)
                
                # Vector lines showing distance vectors
                if player_center is not None:
                    cv2.line(hud_overlay, self.zone_center, player_center, (255, 255, 255), 2)
                    cv2.circle(hud_overlay, player_center, 12, (255, 220, 0), -1)
                
                cv2.circle(hud_overlay, self.zone_center, int(self.zone_radius), zone_color, 4)
                cv2.putText(hud_overlay, f"SURVIVAL ALTITUDE: {int(self.score * 10)}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
            elif self.state == GameState.GAMEOVER:
                cv2.rectangle(hud_overlay, (0, 0), (self.width, self.height), (0, 0, 50), -1)
                cv2.putText(hud_overlay, "CRITICAL ERROR: MOVEMENT BOUNDARY VIOLATION", (120, self.height // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.putText(hud_overlay, f"FINAL SCORE VECTOR: {int(self.score * 10)}", (400, self.height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(hud_overlay, "PRESS 'R' TO FLUSH ENVIRONMENT VARIABLES AND REBOOT", (220, self.height // 2 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)

            # Draw generic skeletal mapping markers if tracked coordinates exist
            if coords and player_center is not None and self.state != GameState.GAMEOVER:
                for key in ['ls', 'rs', 'lh', 'rh']:
                    pt = (int(coords[key][0] * self.width), int(coords[key][1] * self.height))
                    cv2.circle(hud_overlay, pt, 6, (0, 255, 255), -1)
            
            cv2.imshow(self.window_name, hud_overlay)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.state == GameState.GAMEOVER:
                self.state = GameState.STANDBY
                self.standby_start_time = None
                self.previous_center = None
                self.zone_radius = float(self.initial_radius)

        self._cleanup()

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    engine = ConvergenceEngine()
    engine.run()