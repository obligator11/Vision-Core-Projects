import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time
import math
from collections import deque

class BodySwapEngine:
    def __init__(self):
        """
        SECTION 1: init()
        Initializes the CV capture matrices, structural memory arrays, 
        game parameters, and the procedural zero-dependency audio synthesizer.
        """
        self.window_name = "Sayyam AI Lab - Project Body-Swap (Hellfire Edition)"
        self.width = 1280
        self.height = 720
        
        # Ingestion Matrix Configuration
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # MediaPipe Inference Subgraph Initialization
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Kinematic Memory Mapping & Dual-Axis Filtering Buffers
        self.smoothing_buffer_x = deque(maxlen=3)  # Lower window = hyper-responsive raw input
        self.smoothing_buffer_y = deque(maxlen=3)
        self.jitter_threshold = 0.002
        self.prev_normalized_x = 0.5
        self.prev_normalized_y = 0.5
        
        # Operational State System Control
        self.STATE_STANDBY = 0
        self.STATE_PLAYING = 1
        self.STATE_GAMEOVER = 2
        self.current_state = self.STATE_STANDBY
        
        # Dynamic Body Core Mapping Logic (Extremely compressed for high difficulty)
        self.MODES = ["HEAD", "LEFT HAND", "RIGHT HAND", "TORSO"]
        self.current_mode = "HEAD"
        self.mode_switch_interval = 2.5  # Chaotic, rapid remapping pace
        self.last_mode_switch_time = 0.0
        
        # Custom Arcade Entities Configuration
        self.player_x = self.width // 2
        self.player_y = self.height // 2
        self.player_radius = 20  # Shrunk bounding hit radius for high stakes
        
        self.threats = []
        self.threat_spawn_rate = 0.5  # Heavy projectile onslaught matrix
        self.last_threat_spawn_time = 0.0
        self.threat_speed_base = 12.0  # Intense base vertical mapping speed
        self.game_score = 0
        self.difficulty_escalation = 1.0
        
        # Color Core Space (Stark-Tech Aesthetic)
        self.COLOR_PURE_WHITE = (255, 255, 255)
        self.COLOR_NEON_CYAN = (255, 255, 0)
        self.COLOR_NEON_MAGENTA = (255, 0, 255)
        self.COLOR_DEEP_RED = (0, 0, 180)
        self.COLOR_GLOW_GREEN = (0, 255, 0)
        self.COLOR_ORANGE = (0, 140, 255)
        self.COLOR_PURPLE = (240, 32, 160)
        
        # Procedural Audio Compilations
        self.init_audio_pipeline()
        
    def init_audio_pipeline(self):
        """
        Natively synthesizes audio wave matrices into local RAM registers
        to preserve 60+ FPS performance target and eliminate file I/O dependency.
        """
        pygame.mixer.init(frequency=22050, size=-16, channels=2)
        
        sample_rate = 22050
        duration_start = 0.3
        duration_swap = 0.20
        duration_fail = 0.6
        
        # 1. System Ignition Synthesizer (Ascending Sweep)
        num_samples_start = int(sample_rate * duration_start)
        buf_start = np.zeros((num_samples_start, 2), dtype=np.int16)
        for i in range(num_samples_start):
            t = i / sample_rate
            freq = 400 + (t / duration_start) * 600
            val = int(16383 * math.sin(2 * math.pi * freq * t))
            buf_start[i] = [val, val]
        self.sound_start = pygame.sndarray.make_sound(buf_start)
        
        # 2. Control Swap Chaos Frequency (Modulated Matrix Pulse)
        num_samples_swap = int(sample_rate * duration_swap)
        buf_swap = np.zeros((num_samples_swap, 2), dtype=np.int16)
        for i in range(num_samples_swap):
            t = i / sample_rate
            freq = 1100 if (i // 300) % 2 == 0 else 700  # Higher pitch for emergency tension
            val = int(14000 * (1.0 if math.sin(2 * math.pi * freq * t) > 0 else -1.0))
            buf_swap[i] = [val, val]
        self.sound_swap = pygame.sndarray.make_sound(buf_swap)
        
        # 3. System Collapse Buzzer (Decaying Low Pass White Noise Blast)
        num_samples_fail = int(sample_rate * duration_fail)
        buf_fail = np.zeros((num_samples_fail, 2), dtype=np.int16)
        for i in range(num_samples_fail):
            t = i / sample_rate
            envelope = (1.0 - (t / duration_fail)) ** 2
            noise = random.uniform(-1.0, 1.0)
            sub_bass = math.sin(2 * math.pi * 60 * t)
            val = int(16383 * (noise * 0.4 + sub_bass * 0.6) * envelope)
            buf_fail[i] = [val, val]
        self.sound_fail = pygame.sndarray.make_sound(buf_fail)

    def vision_processing(self, frame):
        """
        SECTION 2: vision_processing()
        Executes frame normalization, matrix orientation flips, color 
        conversions, and passes spatial arrays down the MediaPipe tensor framework.
        """
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return frame, results

    def gesture_detection(self, landmarks):
        """
        SECTION 3: gesture_detection()
        Evaluates structural skeletal landmarks using Euclidean spatial formulas
        to extract targeted trigger states (Right hand relative to shoulder).
        """
        if not landmarks:
            return False
            
        r_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        if r_wrist.visibility > 0.5 and r_shoulder.visibility > 0.5:
            if r_wrist.y < r_shoulder.y - 0.1:
                return True
        return False

    def game_logic(self, results):
        """
        SECTION 4 & 5: game_logic() & CONTROL LOGIC
        Extracts selected skeletal kinematics, processes filtering matrix models, 
        drives physics sandboxes, and maps object translation bounds.
        """
        current_time = time.time()
        
        # 1. Active Ingestion Matrix Tracking (Locked on 2D Coordinates)
        if self.current_state == self.STATE_PLAYING and results and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            target_x = None
            target_y = None
            
            if self.current_mode == "HEAD":
                node = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
                if node.visibility > 0.5:
                    target_x = node.x
                    target_y = node.y
                    
            elif self.current_mode == "LEFT HAND":
                node = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                if node.visibility > 0.5:
                    target_x = node.x
                    target_y = node.y
                    
            elif self.current_mode == "RIGHT HAND":
                node = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                if node.visibility > 0.5:
                    target_x = node.x
                    target_y = node.y
                    
            elif self.current_mode == "TORSO":
                sh_l = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                sh_r = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                if sh_l.visibility > 0.5 and sh_r.visibility > 0.5:
                    target_x = (sh_l.x + sh_r.x) / 2.0
                    target_y = (sh_l.y + sh_r.y) / 2.0
            
            # Coordinate Filtration Processing
            if target_x is not None and target_y is not None:
                self.smoothing_buffer_x.append(target_x)
                self.smoothing_buffer_y.append(target_y)
                
                averaged_x = sum(self.smoothing_buffer_x) / len(self.smoothing_buffer_x)
                averaged_y = sum(self.smoothing_buffer_y) / len(self.smoothing_buffer_y)
                
                if abs(averaged_x - self.prev_normalized_x) > self.jitter_threshold:
                    self.prev_normalized_x = averaged_x
                if abs(averaged_y - self.prev_normalized_y) > self.jitter_threshold:
                    self.prev_normalized_y = averaged_y
                    
                self.player_x = int(self.prev_normalized_x * self.width)
                self.player_y = int(self.prev_normalized_y * self.height)
                
                # Active screen coordinate boundary clamp constraints
                self.player_x = max(self.player_radius + 10, min(self.width - self.player_radius - 10, self.player_x))
                self.player_y = max(130, min(self.height - self.player_radius - 10, self.player_y))

        # 2. Asynchronous State Initialization Check Handlers
        if self.current_state in [self.STATE_STANDBY, self.STATE_GAMEOVER]:
            if results and results.pose_landmarks:
                if self.gesture_detection(results.pose_landmarks.landmark):
                    self.sound_start.play()
                    self.current_state = self.STATE_PLAYING
                    self.last_mode_switch_time = current_time
                    self.last_threat_spawn_time = current_time
                    self.current_mode = random.choice(self.MODES)
                    self.threats.clear()
                    self.smoothing_buffer_x.clear()
                    self.smoothing_buffer_y.clear()
                    self.game_score = 0
                    self.difficulty_escalation = 1.0

        # 3. Enhanced Dynamic Chaos Spawning & Physics Math Loops
        if self.current_state == self.STATE_PLAYING:
            # remap Trigger Remapping Cadence System
            if current_time - self.last_mode_switch_time > self.mode_switch_interval:
                available_modes = [m for m in self.MODES if m != self.current_mode]
                self.current_mode = random.choice(available_modes)
                self.sound_swap.play()
                self.last_mode_switch_time = current_time
                self.smoothing_buffer_x.clear()
                self.smoothing_buffer_y.clear()
                
            # Threat Projectile Generator Loops
            if current_time - self.last_threat_spawn_time > (self.threat_spawn_rate / self.difficulty_escalation):
                tx = random.randint(40, self.width - 40)
                ty = -30
                tr = random.randint(18, 38)
                
                # Highly challenging weighted distributions
                choices = ["standard", "swaying", "charger", "seeker"]
                weights = [0.25, 0.25, 0.25, 0.25]
                
                t_type = random.choices(choices, weights=weights, k=1)[0]
                
                # Dynamic speed assignment multipliers
                if t_type == "charger":
                    initial_speed = self.threat_speed_base * 0.25 * self.difficulty_escalation
                elif t_type == "seeker":
                    initial_speed = self.threat_speed_base * 0.85 * self.difficulty_escalation
                else:
                    initial_speed = self.threat_speed_base * self.difficulty_escalation
                
                self.threats.append({
                    'x': tx, 'y': ty, 'r': tr, 'type': t_type, 'base_x': tx,
                    'sway_amplitude': random.randint(70, 140),
                    'sway_speed': random.uniform(0.04, 0.09),
                    'current_speed': initial_speed
                })
                self.last_threat_spawn_time = current_time
                
            # Physics Calculus Space & Matrix Collision Detection Checks
            for threat in self.threats[:]:
                t_type = threat.get('type', 'standard')
                
                if t_type == 'standard':
                    threat['y'] += threat['current_speed']
                elif t_type == 'swaying':
                    threat['y'] += threat['current_speed']
                    threat['x'] = threat['base_x'] + math.sin(threat['y'] * threat['sway_speed']) * threat['sway_amplitude']
                    threat['x'] = max(30, min(self.width - 30, threat['x']))
                elif t_type == 'seeker':
                    # Upgraded multi-axis pursuit algorithms targeting active lock vectors
                    threat['y'] += threat['current_speed']
                    drift_step = 4.5 * self.difficulty_escalation
                    if self.player_x > threat['x']:
                        threat['x'] += min(drift_step, self.player_x - threat['x'])
                    elif self.player_x < threat['x']:
                        threat['x'] -= min(drift_step, threat['x'] - self.player_x)
                    threat['x'] = max(30, min(self.width - 30, threat['x']))
                elif t_type == 'charger':
                    # Non-linear acceleration calculation mapping curves
                    threat['current_speed'] += 0.95 * self.difficulty_escalation
                    threat['y'] += threat['current_speed']
                
                # Check Euclidean Circle Intersection Across Full 2D Workspace
                distance = math.sqrt((threat['x'] - self.player_x)**2 + (threat['y'] - self.player_y)**2)
                if distance < (threat['r'] + self.player_radius):
                    self.sound_fail.play()
                    self.current_state = self.STATE_GAMEOVER
                    
                # Out-of-bounds pruning metrics
                if threat['y'] > self.height + 60:
                    if threat in self.threats:
                        self.threats.remove(threat)
                    self.game_score += 1
                    self.difficulty_escalation += 0.08  # Severe velocity multiplier steps

    def rendering(self, frame, results):
        """
        SECTION 6 & 7: rendering() & VISUAL METRICS
        Performs alpha-blended overlay matrix compositing, displays mode telemetry, 
        and explicitly projects target highlights on active body parts.
        """
        glow_mask = np.zeros_like(frame)
        
        # 1. Overlay High-Intensity Targeting Rings Around Lock Coordinates
        if self.current_state == self.STATE_PLAYING:
            cv2.circle(glow_mask, (self.player_x, self.player_y), 55, self.COLOR_NEON_CYAN, -1)
            cv2.circle(frame, (self.player_x, self.player_y), self.player_radius, self.COLOR_GLOW_GREEN, -1)
            cv2.circle(frame, (self.player_x, self.player_y), self.player_radius + 5, self.COLOR_PURE_WHITE, 2)
            cv2.circle(frame, (self.player_x, self.player_y), 55, self.COLOR_NEON_CYAN, 2)
            
            # HUD connectivity matrix tracking raycast paths
            cv2.line(frame, (self.player_x, 100), (self.player_x, self.player_y - self.player_radius), (120, 120, 120), 1, cv2.LINE_AA)

        # Merge high-contrast illumination visual masks
        frame = cv2.addWeighted(frame, 1.0, glow_mask, 0.40, 0)

        # 2. Render Falling Projectile Vectors Across the Viewport Matrix
        for threat in self.threats:
            t_type = threat.get('type', 'standard')
            if t_type == 'swaying':
                core_color, outline_color = self.COLOR_NEON_CYAN, self.COLOR_NEON_MAGENTA
            elif t_type == 'seeker':
                core_color, outline_color = self.COLOR_ORANGE, self.COLOR_NEON_CYAN
            elif t_type == 'charger':
                core_color, outline_color = self.COLOR_PURPLE, self.COLOR_GLOW_GREEN
            else:
                core_color, outline_color = self.COLOR_DEEP_RED, self.COLOR_NEON_MAGENTA
                
            cv2.circle(frame, (int(threat['x']), int(threat['y'])), threat['r'], core_color, -1)
            cv2.circle(frame, (int(threat['x']), int(threat['y'])), threat['r'], outline_color, 2)
            
        # 3. Project Dynamic HUD Panel Layers
        cv2.rectangle(frame, (0, 0), (self.width, 100), (15, 15, 15), -1)
        cv2.line(frame, (0, 100), (self.width, 100), self.COLOR_NEON_CYAN, 2)
        
        if self.current_state == self.STATE_STANDBY:
            cv2.putText(frame, "SYSTEM DETACHED: STANDBY PROTOCOL", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.COLOR_PURE_WHITE, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (self.width//2 - 420, self.height//2 - 60), (self.width//2 + 420, self.height//2 + 40), (10, 10, 10), -1)
            cv2.rectangle(frame, (self.width//2 - 420, self.height//2 - 60), (self.width//2 + 420, self.height//2 + 40), self.COLOR_NEON_CYAN, 2)
            cv2.putText(frame, "RAISE RIGHT HAND ABOVE SHOULDER TO BOOT ENGINE", (self.width//2 - 390, self.height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_NEON_CYAN, 2, cv2.LINE_AA)
                        
        elif self.current_state == self.STATE_PLAYING:
            cv2.putText(frame, f"ACTIVE LOCK: {self.current_mode}", (30, 62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.COLOR_NEON_MAGENTA, 3, cv2.LINE_AA)
            cv2.putText(frame, f"SCORE: {self.game_score}", (self.width - 240, 62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.COLOR_GLOW_GREEN, 2, cv2.LINE_AA)
            
            # Precision progress interval tracking bar rendering
            time_passed = time.time() - self.last_mode_switch_time
            progress_ratio = max(0.0, min(1.0, (self.mode_switch_interval - time_passed) / self.mode_switch_interval))
            bar_width = int((self.width - 60) * progress_ratio)
            cv2.rectangle(frame, (30, 115), (self.width - 30, 125), (30, 30, 30), -1)
            cv2.rectangle(frame, (30, 115), (30 + bar_width, 125), self.COLOR_NEON_CYAN, -1)
            
        elif self.current_state == self.STATE_GAMEOVER:
            cv2.putText(frame, "CORE COLLAPSE: LOGIC TERMINATED", (30, 62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.COLOR_DEEP_RED, 3, cv2.LINE_AA)
            cv2.putText(frame, f"FINAL SCORE: {self.game_score}", (self.width - 320, 62), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.COLOR_PURE_WHITE, 2, cv2.LINE_AA)
            
            cv2.rectangle(frame, (self.width//2 - 420, self.height//2 - 60), (self.width//2 + 420, self.height//2 + 40), (10, 10, 10), -1)
            cv2.rectangle(frame, (self.width//2 - 420, self.height//2 - 60), (self.width//2 + 420, self.height//2 + 40), self.COLOR_DEEP_RED, 2)
            cv2.putText(frame, "RAISE RIGHT HAND ABOVE SHOULDER TO REBOOT SYSTEM", (self.width//2 - 390, self.height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_NEON_MAGENTA, 2, cv2.LINE_AA)

        return frame

    def main_loop(self):
        """
        SECTION 8: main_loop()
        Drives the operational pipeline loop context, tracking clock periods
        to maintain hardware speed requirements.
        """
        prev_time = time.time()
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
                
            frame, results = self.vision_processing(frame)
            self.game_logic(results)
            output_frame = self.rendering(frame, results)
            
            # FPS Calculation Context metrics
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
            prev_time = current_time
            cv2.putText(output_frame, f"FPS: {int(fps)}", (self.width - 140, 155), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_GLOW_GREEN, 1, cv2.LINE_AA)
            
            cv2.imshow(self.window_name, output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    engine = BodySwapEngine()
    engine.main_loop()