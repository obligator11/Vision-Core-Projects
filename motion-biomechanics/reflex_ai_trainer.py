import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time
import threading
import sys

# =====================================================================
# THREAD-ISOLATED VISION & MODEL INFERENCE WORKER
# =====================================================================
class AsynchronousVisionEngine:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.data_lock = threading.Lock()
        
        self.mp_hands = mp.solutions.hands
        self.tracker = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.5
        )
        
        # FIXED: unified attribute name used everywhere
        self.shared_hand_points = []
        self.shared_frame = None
        
        self.last_valid_points = []
        self.point_velocities = []
        self.dropout_frame_count = 0
        self.max_prediction_allowance = 12
        
        self.ema_cache = {}
        self.smoothing_factor = 0.5

    def start(self):
        if self.started:
            return self
        self.started = True
        self.worker_thread = threading.Thread(target=self._process_pipeline, args=(), daemon=True)
        self.worker_thread.start()
        return self

    def _process_pipeline(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed or frame is None:
                time.sleep(0.001)
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_matrix = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inference_results = self.tracker.process(rgb_matrix)
            
            computed_points = []
            
            if inference_results.multi_hand_landmarks:
                self.dropout_frame_count = 0
                for hand_idx, hand_lms in enumerate(inference_results.multi_hand_landmarks):
                    if hand_idx >= 2: break
                    for lm_idx, lm in enumerate(hand_lms.landmark):
                        joint_key = f"{hand_idx}_{lm_idx}"
                        
                        if joint_key in self.ema_cache:
                            px, py = self.ema_cache[joint_key]
                            smoothed_x = self.smoothing_factor * lm.x + (1.0 - self.smoothing_factor) * px
                            smoothed_y = self.smoothing_factor * lm.y + (1.0 - self.smoothing_factor) * py
                            self.ema_cache[joint_key] = (smoothed_x, smoothed_y)
                        else:
                            self.ema_cache[joint_key] = (lm.x, lm.y)
                            
                        computed_points.append(self.ema_cache[joint_key])
                
                if len(computed_points) == len(self.last_valid_points) and len(computed_points) > 0:
                    self.point_velocities = [(c[0] - p[0], c[1] - p[1]) for c, p in zip(computed_points, self.last_valid_points)]
                else:
                    self.point_velocities = [(0.0, 0.0)] * len(computed_points)
                self.last_valid_points = list(computed_points)
                
            else:
                self.dropout_frame_count += 1
                if self.dropout_frame_count <= self.max_prediction_allowance and len(self.last_valid_points) > 0:
                    predicted_points = []
                    for pt, vel in zip(self.last_valid_points, self.point_velocities):
                        decayed_vel_x = vel[0] * 0.85
                        decayed_vel_y = vel[1] * 0.85
                        predicted_points.append((pt[0] + decayed_vel_x, pt[1] + decayed_vel_y))
                    
                    computed_points = list(predicted_points)
                    self.last_valid_points = list(predicted_points)
                    self.point_velocities = [(v[0]*0.85, v[1]*0.85) for v in self.point_velocities]
                else:
                    self.last_valid_points = []
                    self.point_velocities = []

            # FIXED: write to self.shared_hand_points (was writing to mismatched name)
            with self.data_lock:
                self.shared_frame = frame.copy()
                self.shared_hand_points = computed_points
                
            time.sleep(0.002)

    def lock_state(self):
        return self.data_lock

    def get_synchronized_payload(self, render_w, render_h):
        # FIXED: reads self.shared_hand_points which is now properly initialized
        with self.data_lock:
            if self.shared_frame is None:
                return None, []
            
            resized_canvas = cv2.resize(self.shared_frame, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
            scaled_coordinates = [(int(pt[0] * render_w), int(pt[1] * render_h)) for pt in self.shared_hand_points]
            return resized_canvas, scaled_coordinates

    def stop(self):
        self.started = False
        if self.worker_thread.is_alive():
            self.worker_thread.join()
        self.cap.release()

# =====================================================================
# PROCEDURAL MATHEMATICAL AUDIO SYNTHESIZER
# =====================================================================
class RealTimeAudioSynthesizer:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sample_rate = 44100

    def create_sound_channel(self, freq_sequence, duration=0.2, wave_type='sine', volume=0.4):
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        if wave_type == 'sine':
            data = np.sin(2 * np.pi * freq_sequence * t)
        elif wave_type == 'square':
            data = np.sign(np.sin(2 * np.pi * freq_sequence * t))
        elif wave_type == 'sawtooth':
            data = 2 * (t * freq_sequence - np.floor(t * freq_sequence + 0.5))
        else:
            data = np.sin(2 * np.pi * freq_sequence * t)

        decay = np.exp(-4.5 * np.linspace(0, 1, num_samples))
        processed_data = (data * decay * 32767 * volume).astype(np.int16)
        
        stereo_matrix = np.ascontiguousarray(np.vstack((processed_data, processed_data)).T)
        return pygame.sndarray.make_sound(stereo_matrix)

    def trigger_hit(self):
        sound = self.create_sound_channel(980, duration=0.08, wave_type='sine', volume=0.45)
        sound.play()

    def trigger_miss(self):
        sound = self.create_sound_channel(140, duration=0.2, wave_type='sawtooth', volume=0.2)
        sound.play()

    def start_background_loop(self):
        num_samples = int(self.sample_rate * 2.0)
        t = np.linspace(0, 2.0, num_samples, False)
        wave = np.sin(2 * np.pi * 65.41 * t) + 0.15 * np.sin(2 * np.pi * 130.81 * t)
        processed_data = (wave * 0.05 * 32767).astype(np.int16)
        stereo_matrix = np.ascontiguousarray(np.vstack((processed_data, processed_data)).T)
        sound = pygame.sndarray.make_sound(stereo_matrix)
        sound.play(loops=-1)

# =====================================================================
# REFLEX GAME STATE INFRASTRUCTURE
# =====================================================================
class TargetInstance:
    def __init__(self, x, y, max_lifetime):
        self.x = x
        self.y = y
        self.base_radius = 45
        self.current_radius = 5
        self.max_lifetime = max_lifetime
        self.birth_time = time.time()
        self.is_dead = False
        self.pulse_phase = random.uniform(0, np.pi)

    def get_remaining_percentage(self):
        elapsed = time.time() - self.birth_time
        return max(0.0, 1.0 - (elapsed / self.max_lifetime))

    def update_animation(self):
        pct = self.get_remaining_percentage()
        if pct <= 0:
            self.is_dead = True
        if self.current_radius < self.base_radius:
            self.current_radius += (self.base_radius - self.current_radius) * 0.4
        self.pulse_phase += 0.15
        self.animated_radius = int(self.current_radius + np.sin(self.pulse_phase) * 3)

class HitFeedbackParticle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.radius = 12
        self.max_radius = 65
        self.alpha = 1.0
        self.color = color

    def update(self):
        self.radius += (self.max_radius - self.radius) * 0.3
        self.alpha -= 0.12
        return self.alpha > 0

# =====================================================================
# MAIN SYSTEM CORES & APPLICATION LOOP
# =====================================================================
class ReflexAITrainerEngine:
    def __init__(self):
        self.window_name = "Reflex AI Trainer - Unbreakable Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        self.synth = RealTimeAudioSynthesizer()
        self.synth.start_background_loop()

        self.mp_hands = mp.solutions.hands
        self.engine_worker = AsynchronousVisionEngine(src=0).start()
        
        self.game_state = "START_SCREEN"
        self.score = 0
        self.combo = 0
        self.high_score = 0
        self.last_reaction_time_ms = 0
        
        self.target_spawn_cooldown = 1.8
        self.target_max_lifetime = 1.5
        self.last_spawn_timestamp = time.time()
        
        self.active_targets = []
        self.active_particles = []
        
        self.current_hand_points = []
        self.previous_hand_points = []
        self.hand_velocity_magnitude = 0.0

        self.fps_frame_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 60

        self.countdown_start_time = 0
        self.game_start_time = 0

    def spawn_dynamic_target(self, width, height):
        padding_x = int(width * 0.12)
        padding_y = int(height * 0.15) + 75
        
        rx = random.randint(padding_x, max(padding_x + 10, width - padding_x))
        ry = random.randint(padding_y, max(padding_y + 10, height - padding_y))
        
        new_target = TargetInstance(rx, ry, self.target_max_lifetime)
        self.active_targets.append(new_target)
        self.last_spawn_timestamp = time.time()

    def evaluate_game_physics(self):
        current_time = time.time()
        
        elapsed_game_time = current_time - self.game_start_time
        speed_factor = min(1.0, elapsed_game_time / 90.0) 
        self.target_spawn_cooldown = max(0.65, 1.8 - (speed_factor * 1.15))
        self.target_max_lifetime = max(0.55, 1.5 - (speed_factor * 0.95))

        if current_time - self.last_spawn_timestamp > self.target_spawn_cooldown:
            if len(self.active_targets) < 3:
                try:
                    win_rect = cv2.getWindowImageRect(self.window_name)
                    w, h = win_rect[2], win_rect[3]
                except:
                    w, h = 1280, 720
                self.spawn_dynamic_target(w, h)

        velocity_bonus = min(30, int(self.hand_velocity_magnitude * 0.45))

        for target in self.active_targets[:]:
            target.update_animation()
            
            if target.is_dead:
                self.synth.trigger_miss()
                self.combo = 0 
                self.active_targets.remove(target)
                continue

            hit_detected = False
            adaptive_hit_radius = target.animated_radius + 15 + velocity_bonus
            
            for idx, curr_pt in enumerate(self.current_hand_points):
                dist = np.hypot(curr_pt[0] - target.x, curr_pt[1] - target.y)
                if dist < adaptive_hit_radius:
                    hit_detected = True
                    break
                
                if idx < len(self.previous_hand_points):
                    prev_pt = self.previous_hand_points[idx]
                    p_curr = np.array(curr_pt)
                    p_prev = np.array(prev_pt)
                    p_targ = np.array([target.x, target.y])
                    
                    line_vec = p_curr - p_prev
                    targ_vec = p_targ - p_prev
                    line_len_sq = np.sum(line_vec ** 2)
                    
                    if line_len_sq > 0:
                        t = max(0.0, min(1.0, np.dot(targ_vec, line_vec) / line_len_sq))
                        projection = p_prev + t * line_vec
                        sweep_dist = np.hypot(projection[0] - target.x, projection[1] - target.y)
                        
                        if sweep_dist < adaptive_hit_radius:
                            hit_detected = True
                            break
            
            if hit_detected:
                self.synth.trigger_hit()
                self.combo += 1
                multiplier = 1 + (self.combo // 5)
                self.score += 10 * multiplier
                if self.score > self.high_score:
                    self.high_score = self.score
                
                self.last_reaction_time_ms = int((current_time - target.birth_time) * 1000)
                self.active_particles.append(HitFeedbackParticle(target.x, target.y, (0, 255, 255)))
                self.active_particles.append(HitFeedbackParticle(target.x, target.y, (255, 0, 128)))
                
                self.active_targets.remove(target)

        self.active_particles = [p for p in self.active_particles if p.update()]

    def render_glow_circle(self, mask, center, radius, color, thickness=1, glow_radius=12):
        for r in range(glow_radius, 0, -3):
            cv2.circle(mask, center, radius + r, color, thickness + r, cv2.LINE_AA)

    def draw_user_interface(self, frame):
        h, w, _ = frame.shape
        ui_layer = frame.copy()
        overlay_mask = np.zeros_like(frame)

        self.fps_frame_counter += 1
        if time.time() - self.fps_timer > 1.0:
            self.current_fps = self.fps_frame_counter
            self.fps_frame_counter = 0
            self.fps_timer = time.time()

        if self.game_state == "START_SCREEN":
            frame[:] = cv2.GaussianBlur(frame, (25, 25), 0)
            cv2.putText(frame, "REFLEX AI TRAINER", (w//2 - 320, h//2 - 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, "Dead-Reckoning Predictive Kinematics Core Active", (w//2 - 340, h//2 - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            
            pulse = int(127 + 127 * np.sin(time.time() * 5))
            cv2.putText(frame, "PRESS [SPACEBAR] TO START", (w//2 - 260, h//2 + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, pulse, 255), 3, cv2.LINE_AA)

        elif self.game_state == "COUNTDOWN":
            frame[:] = cv2.GaussianBlur(frame, (15, 15), 0)
            elapsed = time.time() - self.countdown_start_time
            remaining = 3.9 - elapsed
            if remaining <= 0.9:
                self.game_state = "GAMEPLAY"
                self.game_start_time = time.time()
                self.last_spawn_timestamp = time.time()
            else:
                count_str = str(int(remaining))
                cv2.putText(frame, count_str, (w//2 - 50, h//2 + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 10, cv2.LINE_AA)

        elif self.game_state == "GAMEPLAY":
            for target in self.active_targets:
                pct = target.get_remaining_percentage()
                node_color = (0, 255, 128) if pct > 0.5 else ((0, 255, 255) if pct > 0.2 else (0, 0, 255))
                
                self.render_glow_circle(overlay_mask, (target.x, target.y), target.animated_radius, node_color)
                cv2.circle(ui_layer, (target.x, target.y), target.animated_radius, node_color, 3, cv2.LINE_AA)
                cv2.circle(ui_layer, (target.x, target.y), int(target.animated_radius * 0.35), node_color, -1, cv2.LINE_AA)
                
                end_angle = int(360 * pct)
                cv2.ellipse(ui_layer, (target.x, target.y), (target.animated_radius + 8, target.animated_radius + 8), 
                            -90, 0, end_angle, node_color, 2, cv2.LINE_AA)

            for particle in self.active_particles:
                p_mask = np.zeros_like(frame)
                cv2.circle(p_mask, (particle.x, particle.y), int(particle.radius), particle.color, 2, cv2.LINE_AA)
                cv2.addWeighted(frame, 1.0, p_mask, particle.alpha, 0, frame)

            if len(self.current_hand_points) > 0:
                num_hands = len(self.current_hand_points) // 21
                for h_idx in range(num_hands):
                    offset = h_idx * 21
                    hand_slice = self.current_hand_points[offset:offset+21]

                    for pt in hand_slice:
                        cv2.circle(ui_layer, pt, 2, (255, 230, 0), -1, cv2.LINE_AA)
                        cv2.circle(overlay_mask, pt, 5, (0, 255, 255), -1, cv2.LINE_AA)

                    for connection in self.mp_hands.HAND_CONNECTIONS:
                        p1, p2 = connection[0], connection[1]
                        if p1 < len(hand_slice) and p2 < len(hand_slice):
                            cv2.line(ui_layer, hand_slice[p1], hand_slice[p2], (0, 255, 255), 1, cv2.LINE_AA)

            cv2.addWeighted(ui_layer, 1.0, overlay_mask, 0.35, 0, frame)

            cv2.rectangle(frame, (0, 0), (w, 75), (15, 15, 15), -1)
            cv2.line(frame, (0, 75), (w, 75), (50, 50, 50), 1, cv2.LINE_AA)

            cv2.putText(frame, f"SCORE: {self.score}", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            combo_color = (0, 255, 255) if self.combo >= 5 else (180, 180, 180)
            cv2.putText(frame, f"COMBO: x{self.combo}", (250, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, combo_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"REACTION: {self.last_reaction_time_ms} ms", (490, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(frame, f"HIGH: {self.high_score}", (820, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 130, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {self.current_fps}", (w - 130, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 180, 255), 2, cv2.LINE_AA)

    def execute_pipeline(self):
        while True:
            try:
                win_rect = cv2.getWindowImageRect(self.window_name)
                win_w = win_rect[2] if win_rect[2] > 100 else 1280
                win_h = win_rect[3] if win_rect[3] > 100 else 720
            except:
                win_w, win_h = 1280, 720

            render_frame, points = self.engine_worker.get_synchronized_payload(win_w, win_h)
            
            if render_frame is None:
                time.sleep(0.01)
                continue

            self.previous_hand_points = list(self.current_hand_points)
            self.current_hand_points = points

            if len(self.current_hand_points) == len(self.previous_hand_points) and len(self.current_hand_points) > 0:
                deltas = [np.hypot(c[0] - p[0], c[1] - p[1]) for c, p in zip(self.current_hand_points, self.previous_hand_points)]
                self.hand_velocity_magnitude = float(np.mean(deltas))
            else:
                self.hand_velocity_magnitude = 0.0

            if self.game_state == "GAMEPLAY":
                self.evaluate_game_physics()

            self.draw_user_interface(render_frame)
            cv2.imshow(self.window_name, render_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: 
                break
            elif key == 32: 
                if self.game_state == "START_SCREEN":
                    self.countdown_start_time = time.time()
                    self.game_state = "COUNTDOWN"
                    self.score = 0
                    self.combo = 0

        self.engine_worker.stop()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        sys.exit(0)

if __name__ == "__main__":
    engine = ReflexAITrainerEngine()
    engine.execute_pipeline()