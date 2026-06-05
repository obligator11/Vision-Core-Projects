import cv2
import mediapipe as mp
import numpy as np
import pygame
import multiprocessing as mp_lib
import time
import random

# ==========================================
# BACKGROUND INFERENCE CONCURRENCY WORKER
# ==========================================
def vision_inference_worker(frame_queue, coord_queue, stop_event):
    """
    Isolated process dedicated strictly to executing the heavy MediaPipe 
    neural graph, preventing the main game thread from bottlenecking on GIL.
    """
    mp_hands = mp.solutions.hands
    # Low-confidence limits optimized for rapid, high-speed sweeping motion blurs
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # Flip and convert color space for MediaPipe graph Ingestion
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                # Isolate the dominant wrist node (Landmark 0)
                wrist = results.multi_hand_landmarks[0].landmark[0]
                coord_queue.put((wrist.x, wrist.y, time.time()))
            else:
                coord_queue.put(None)
        else:
            time.sleep(0.001)
            
    hands.close()

# ==========================================
# PROCEDURAL AUDIO SYNTHESIZER MODULE
# ==========================================
class ProceduralAudioEngine:
    """
    Compiles pure mathematical arrays directly into RAM sound cards,
    bypassing disk storage reads completely to support real-time latency loops.
    """
    @staticmethod
    def initialize_audio_subsystem():
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()

    @staticmethod
    def generate_crash_sound():
        """Synthesizes a low-pass digital white-noise distortion decay with sub-bass."""
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Exponentially decaying white noise combined with a heavy low sine wave
        noise = np.random.uniform(-1, 1, len(t))
        envelope = np.exp(-10 * t)
        sub_bass = np.sin(2 * np.pi * 60 * t) * envelope
        
        audio_arr = ((noise * 0.4 + sub_bass * 0.6) * envelope * 32767).astype(np.int16)
        stereo_arr = np.column_stack((audio_arr, audio_arr))
        return pygame.mixer.Sound(buffer=stereo_arr)

    @staticmethod
    def generate_engine_tone(frequency=220):
        """Generates a looping sine-wave chunk for dynamic pitch modifications."""
        sample_rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        wave = np.sin(2 * np.pi * frequency * t)
        audio_arr = (wave * 0.3 * 32767).astype(np.int16)
        stereo_arr = np.column_stack((audio_arr, audio_arr))
        return pygame.mixer.Sound(buffer=stereo_arr)


# ==========================================
# PRIMARY GAME MECHANICS & HUD VECTOR ENGINE
# ==========================================
class VelocityPulseEngine:
    def __init__(self):
        # Window Canvas properties explicitly configured for monitor scalability
        self.width, self.height = 1280, 720
        self.window_name = "Project Velocity-Pulse (S-Tier Engine)"
        
        # Game State Variable Flags
        self.is_running = False
        self.game_over = False
        self.score = 0
        
        # Kinematic State Vectors
        self.player_pos = np.array([self.width // 2, self.height // 2], dtype=np.float32)
        self.player_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.player_radius = 20
        self.base_difficulty_speed = 3.0
        
        # Motion Capture Tracking History Memory
        self.wrist_history = []  # Tuples of (x, y, timestamp)
        self.max_history_len = 5
        self.motion_trail = []   # Player position history for VFX rendering
        
        # Target Threat Matrix Generation boundaries
        self.target_pos = np.array([0, 0], dtype=np.float32)
        self.spawn_new_target()
        
        # Subsystem Audio Storage
        ProceduralAudioEngine.initialize_audio_subsystem()
        self.sound_crash = ProceduralAudioEngine.generate_crash_sound()
        self.active_channel = pygame.mixer.Channel(0)

    def spawn_new_target(self):
        """Spawns an interactive target far from borders to secure clear visibility."""
        self.target_pos[0] = random.randint(100, self.width - 100)
        self.target_pos[1] = random.randint(100, self.height - 100)

    def process_kinematic_velocity(self, raw_data):
        """
        Processes physical coordinate sequences over time intervals (dt) 
        to accurately measure velocity states independent of baseline position.
        """
        if raw_data is None:
            # Drop data mapping if hand goes out of bounds to maintain frame rate
            return
            
        x_mapped = raw_data[0] * self.width
        y_mapped = raw_data[1] * self.height
        current_time = raw_data[2]
        
        self.wrist_history.append((x_mapped, y_mapped, current_time))
        if len(self.wrist_history) > self.max_history_len:
            self.wrist_history.pop(0)
            
        if len(self.wrist_history) >= 2:
            # Extract historical boundary nodes to perform finite calculus steps
            start_node = self.wrist_history[0]
            end_node = self.wrist_history[-1]
            
            dt = end_node[2] - start_node[2]
            if dt > 0:
                dx = end_node[0] - start_node[0]
                dy = end_node[1] - start_node[1]
                
                # Calculate instantaneous physics vector
                v_x = dx / dt
                v_y = dy / dt
                
                # Check for Initial Start/Ignition state via rapid velocity surge trigger
                if not self.is_running:
                    velocity_magnitude = np.hypot(v_x, v_y)
                    if velocity_magnitude > 1200:  # Absolute velocity pixel trigger spike
                        self.is_running = True
                        self.sound_crash.play() # Feedback initialization indicator
                else:
                    # Non-Linear Scaling Mapping: translates vector speed limits into game acceleration
                    scaling_factor = 0.015
                    self.player_velocity[0] += v_x * scaling_factor
                    self.player_velocity[1] += v_y * scaling_factor

    def update_physics(self):
        """Executes custom vector calculations, applying non-linear damping parameters."""
        if not self.is_running or self.game_over:
            return
            
        # Non-Linear control check: Apply strict damping to prevent infinite kinetic expansion
        self.player_velocity *= 0.95
        
        # Enforce constant minimal velocity threshold (Player cannot stop moving)
        velocity_magnitude = np.hypot(self.player_velocity[0], self.player_velocity[1])
        minimum_speed = self.base_difficulty_speed + (self.score * 0.5)
        
        if velocity_magnitude < minimum_speed:
            if velocity_magnitude == 0:
                # Default direction fallback vector if completely static
                self.player_velocity = np.array([minimum_speed, 0.0], dtype=np.float32)
            else:
                self.player_velocity = (self.player_velocity / velocity_magnitude) * minimum_speed
        
        # Update Player Position matrix coordinates
        self.player_pos += self.player_velocity
        
        # Motion Trail tracking updates for high-end aesthetic bloom tracking
        self.motion_trail.append(tuple(self.player_pos.astype(int)))
        if len(self.motion_trail) > 25:
            self.motion_trail.pop(0)
            
        # Procedural Pitch Control modulation bound to current player velocity tracking
        if not self.active_channel.get_busy():
            mapped_freq = int(220 + clamp(velocity_magnitude * 10, 0, 600))
            engine_chunk = ProceduralAudioEngine.generate_engine_tone(mapped_freq)
            self.active_channel.play(engine_chunk)
            
        # Collision validation math against environmental screen borders
        if (self.player_pos[0] - self.player_radius < 0 or self.player_pos[0] + self.player_radius > self.width or
            self.player_pos[1] - self.player_radius < 0 or self.player_pos[1] + self.player_radius > self.height):
            self.game_over = True
            self.sound_crash.play()
            self.active_channel.stop()

        # Euclidean Distance evaluation against Target intercept fields
        distance_to_target = np.hypot(self.player_pos[0] - self.target_pos[0], self.player_pos[1] - self.target_pos[1])
        if distance_to_target < (self.player_radius + 25): # Radius of player + target bounds
            self.score += 1
            self.spawn_new_target()
            # Play a synthesized audio confirmation shift
            pygame.mixer.Sound(buffer=(np.sin(np.linspace(0, 0.1, 4410)) * 32767).astype(np.int16)).play()

    def render_graphics(self, camera_frame):
        """Assembles alpha-blended overlay pipelines and cybernetic HUD diagnostics."""
        # Darken base frame matrix to enforce an immersive dark cybernetic environment
        display_mask = cv2.flip(camera_frame, 1)
        display_mask = cv2.resize(display_mask, (self.width, self.height))
        display_mask = cv2.convertScaleAbs(display_mask, alpha=0.25, beta=0)
        
        # Render dynamic vector paths detailing player motion vector history
        if len(self.motion_trail) >= 2:
            for i in range(len(self.motion_trail) - 1):
                thickness = int(1 + (i / len(self.motion_trail)) * 6)
                # Neon Blue/Cyan gradient tracking bloom parameters
                cv2.line(display_mask, self.motion_trail[i], self.motion_trail[i+1], (255, 255, 0), thickness)
                
        # Draw Interactive Objective Target Matrix
        if self.is_running and not self.game_over:
            cv2.circle(display_mask, tuple(self.target_pos.astype(int)), 25, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(display_mask, tuple(self.target_pos.astype(int)), 30, (0, 165, 255), 2, cv2.LINE_AA)
            
        # Draw Main Player Node
        color_palette = (0, 0, 255) if not self.game_over else (0, 0, 50) # Crimson vector alert state
        cv2.circle(display_mask, tuple(self.player_pos.astype(int)), self.player_radius, color_palette, -1, cv2.LINE_AA)
        cv2.circle(display_mask, tuple(self.player_pos.astype(int)), self.player_radius + 5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Overlay System Text UI Elements
        self.render_hud_text(display_mask)
        
        cv2.imshow(self.window_name, display_mask)

    def render_hud_text(self, canvas):
        """Renders crisp high-tech metrics overlaying user actions."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        if not self.is_running:
            # Standby Initialization Phase Text UI Display
            cv2.putText(canvas, "SYSTEM: STANDBY_MODE_LOCKED", (40, 60), font, 0.8, (0, 255, 255), 2)
            cv2.putText(canvas, ">> SWIPE HAND AGGRESSIVELY TO IGNITE ENGINE <<", (320, 360), font, 0.9, (0, 100, 255), 3, cv2.LINE_AA)
        elif self.game_over:
            # Terminal State Termination Hud Overlays
            cv2.putText(canvas, "CRITICAL ERROR: SYSTEM_CRASH_COLLISION", (300, 320), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"FINAL SCORE RESIDUE: {self.score}", (480, 390), font, 0.9, (255, 255, 255), 2)
            cv2.putText(canvas, "PRESS 'R' TO FLUSH SYSTEM REGISTERS OR 'Q' TO RECLAIM MEMORY", (220, 460), font, 0.6, (0, 255, 255), 1)
        else:
            # Active Runtime Telemetry Engine Data HUD Mapping
            vel_mag = np.hypot(self.player_velocity[0], self.player_velocity[1])
            cv2.putText(canvas, f"SCORE: {self.score:03d}", (40, 60), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"VELOCITY TELEMETRY: {vel_mag:.2f} px/f", (40, 100), font, 0.7, (255, 255, 255), 1)
            cv2.putText(canvas, f"MINIMUM_SPEED_FLOOR: {self.base_difficulty_speed + (self.score * 0.5):.1f}", (40, 130), font, 0.5, (0, 255, 255), 1)

    def reset_engine_registers(self):
        """Flushes system variables back to tracking baselines."""
        self.game_over = False
        self.is_running = False
        self.score = 0
        self.player_pos = np.array([self.width // 2, self.height // 2], dtype=np.float32)
        self.player_velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.wrist_history.clear()
        self.motion_trail.clear()
        self.spawn_new_target()


def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

# ==========================================
# SYSTEM ORCHESTRATION PIPELINE
# ==========================================
if __name__ == '__main__':
    # Initialize high-speed capture configurations via native system ports
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Establish Inter-Process Communications Infrastructure (IPC)
    frame_in_q = mp_lib.Queue(maxsize=1)
    coord_out_q = mp_lib.Queue(maxsize=1)
    stop_signal = mp_lib.Event()
    
    # Boot child tracking engine background process daemon node
    process_worker = mp_lib.Process(
        target=vision_inference_worker, 
        args=(frame_in_q, coord_out_q, stop_signal)
    )
    process_worker.daemon = True
    process_worker.start()
    
    # Compile core engine components
    game_engine = VelocityPulseEngine()
    cv2.namedWindow(game_engine.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(game_engine.window_name, game_engine.width, game_engine.height)
    
    # Execute primary clock architecture execution loop
    while cap.isOpened():
        success, raw_frame = cap.read()
        if not success:
            break
            
        # Non-blocking injection mechanism pushing video arrays to the ML worker process
        if frame_in_q.empty():
            frame_in_q.put(raw_frame)
            
        # Intercept asynchronous coordinate payloads processed by background worker
        if not coord_out_q.empty():
            payload = coord_out_q.get()
            game_engine.process_kinematic_velocity(payload)
            
        # Step the local simulation forward 
        game_engine.update_physics()
        
        # Render visual composite 
        game_engine.render_graphics(raw_frame)
        
        # Check system register input interrupts
        key_strike = cv2.waitKey(1) & 0xFF
        if key_strike == ord('q') or key_strike == ord('Q'):
            break
        elif key_strike == ord('r') or key_strike == ord('R'):
            game_engine.reset_engine_registers()
            
    # Reclaim global allocations and shutdown concurrency streams cleanly
    stop_signal.set()
    process_worker.terminate()
    process_worker.join()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    pygame.quit()