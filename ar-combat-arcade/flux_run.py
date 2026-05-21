import cv2
import numpy as np
import mediapipe as mp
import time
import random
import threading
from collections import deque
import pygame

# --- GLOBAL CONFIGURATION MATRIX ---
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

COLOR_BG_DARK = (15, 10, 25)
COLOR_CEILING = (45, 15, 70)
COLOR_FLOOR = (20, 65, 45)
COLOR_NEON_CYAN = (255, 255, 0)
COLOR_NEON_MAGENTA = (180, 0, 255)
COLOR_NEON_GREEN = (0, 255, 150)
COLOR_SPIKE = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)

# --- PROCEDURAL SIGNAL SYNTHESIS MODULE ---
def generate_synth_sound(frequency, duration, sound_type="sine"):
    try:
        sample_rate = 22050
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        if sound_type == "sine":
            wave = np.sin(2 * np.pi * frequency * t) * 0.5
            decay = np.exp(-5 * t)
            wave = wave * decay
        elif sound_type == "noise":
            wave = np.random.uniform(-0.5, 0.5, num_samples)
            decay = np.exp(-3 * t)
            wave = wave * decay

        audio_buffer = (wave * 32767).astype(np.int16)
        stereo_buffer = np.column_stack((audio_buffer, audio_buffer))
        return pygame.sndarray.make_sound(stereo_buffer)
    except:
        return None

# --- OBSTACLE ENTITY VECTOR MODULE ---
class Obstacle:
    def __init__(self, x_start, type_id, speed):
        self.x = float(x_start)
        self.type = type_id  
        self.speed = speed
        self.width = 40
        self.height = 60
        self.y_offset = 0.0
        self.phase = random.uniform(0, 2 * np.pi)

    def update(self, dt):
        self.x -= self.speed * dt
        if self.type == 2:
            self.y_offset = np.sin(time.time() * 5 + self.phase) * 100

    def draw(self, target_buffer, floor_y, ceiling_y):
        if self.type == 0:
            pts = np.array([[int(self.x), int(floor_y)], [int(self.x + self.width), int(floor_y)], [int(self.x + self.width // 2), int(floor_y - self.height)]], np.int32)
            cv2.drawContours(target_buffer, [pts], 0, COLOR_SPIKE, -1)
            cv2.polylines(target_buffer, [pts], True, COLOR_NEON_MAGENTA, 2)
        elif self.type == 1:
            pts = np.array([[int(self.x), int(ceiling_y)], [int(self.x + self.width), int(ceiling_y)], [int(self.x + self.width // 2), int(ceiling_y + self.height)]], np.int32)
            cv2.drawContours(target_buffer, [pts], 0, COLOR_SPIKE, -1)
            cv2.polylines(target_buffer, [pts], True, COLOR_NEON_MAGENTA, 2)
        elif self.type == 2:
            center_y = int((floor_y + ceiling_y) // 2 + self.y_offset)
            cv2.rectangle(target_buffer, (int(self.x), center_y - 40), (int(self.x + self.width), center_y + 40), COLOR_NEON_MAGENTA, -1)
            cv2.rectangle(target_buffer, (int(self.x), center_y - 40), (int(self.x + self.width), center_y + 40), COLOR_WHITE, 2)

    def get_hitbox(self, floor_y, ceiling_y):
        if self.type == 0: return (self.x, floor_y - self.height, self.width, self.height)
        elif self.type == 1: return (self.x, ceiling_y, self.width, self.height)
        else: return (self.x, (floor_y + ceiling_y) // 2 + self.y_offset - 40, self.width, 80)

# --- CORE GRAPHICS ENGINE ---
class FluxEngine:
    def __init__(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2)
            self.sound_flip = generate_synth_sound(880, 0.15, "sine")
            self.sound_crash = generate_synth_sound(120, 0.6, "noise")
        except:
            self.sound_flip, self.sound_crash = None, None

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cv2.namedWindow("Project Flux-Run // Sayyam AI Lab", cv2.WINDOW_NORMAL)
        
        # Multithreading Synced Variable Payload
        self.shared_palm_detected = False
        self.shared_finger_direction = "NONE"  # "UP", "DOWN", "NONE"
        self.shared_landmarks = []
        self.thread_lock = threading.Lock()
        self.current_working_frame = None
        self.keep_running = True

        self.inference_thread = threading.Thread(target=self._async_vision_inference, daemon=True)
        self.inference_thread.start()

        self.current_state = "START"
        self.floor_y = int(FRAME_HEIGHT * 0.8)
        self.ceiling_y = int(FRAME_HEIGHT * 0.2)
        self.player_x = 150.0
        self.player_y = float(self.floor_y - 30)
        self.player_radius = 25
        
        self.gravity_state = 1.0  
        self.current_gravity_vector = 1.0
        
        self.trail_buffer = deque(maxlen=20)
        self.obstacles = []
        self.game_speed = 480.0
        self.score = 0
        self.cooldown_frames = 20
        self.bg_scroll_offset = 0.0  
        self.current_camera_tilt = 0.0
        self.target_camera_tilt = 0.0
        self.last_frame_time = time.time()
        
        # Local main thread cache layers
        self.local_palm_detected = False
        self.local_finger_direction = "NONE"
        self.local_landmarks = []

    def _async_vision_inference(self):
        """Processes geometry directly to match finger posture inputs."""
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55
        )

        while self.keep_running:
            frame_to_process = None
            with self.thread_lock:
                if self.current_working_frame is not None:
                    frame_to_process = self.current_working_frame.copy()
            
            if frame_to_process is None:
                time.sleep(0.01)
                continue

            try:
                img_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                palm_found = False
                direction = "NONE"
                serialized_landmarks = []
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    for lm in hand_landmarks.landmark:
                        serialized_landmarks.append((lm.x, lm.y))
                    
                    # Extract raw index fingertip and base tracking variables
                    tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                    
                    # Check for widespread open palm posture (used for starting/rebooting)
                    index_open = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                    middle_open = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
                    ring_open = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
                    pinky_open = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
                    
                    if index_open and middle_open and ring_open and pinky_open:
                        palm_found = True

                    # Directional logic relative to MCP knuckle anchor offset boundaries
                    if tip_y < mcp_y - 0.08:
                        direction = "UP"
                    elif tip_y > mcp_y + 0.08:
                        direction = "DOWN"

                with self.thread_lock:
                    self.shared_palm_detected = palm_found
                    self.shared_finger_direction = direction
                    self.shared_landmarks = serialized_landmarks

            except Exception as e:
                pass
            
            time.sleep(0.01)

    def reset_game_state(self):
        self.current_state = "PLAYING"
        self.player_y = float(self.floor_y - 30)
        self.gravity_state = 1.0
        self.current_gravity_vector = 1.0
        self.trail_buffer.clear()
        self.obstacles.clear()
        self.game_speed = 480.0
        self.score = 0
        self.current_camera_tilt = 0.0
        self.target_camera_tilt = 0.0
        self.cooldown_frames = 20

    def process_state(self, dt):
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

        if self.current_state == "START":
            if self.local_palm_detected and self.cooldown_frames == 0:
                self.reset_game_state()
            return

        if self.current_state == "GAMEOVER":
            if self.local_palm_detected and self.cooldown_frames == 0:
                self.reset_game_state()
            return

        # --- ACTIVE RUNNING CODE BLOCK ---
        self.bg_scroll_offset = (self.bg_scroll_offset + self.game_speed * dt) % 80
        self.score += int(dt * 100)
        self.game_speed += dt * 6.5  

        # Dynamic flip assignments based on matching finger directives
        if self.local_finger_direction == "UP" and self.gravity_state == 1.0 and self.cooldown_frames == 0:
            self.gravity_state = -1.0
            self.target_camera_tilt = -15.0
            self.cooldown_frames = 12
            if self.sound_flip: self.sound_flip.play()
            
        elif self.local_finger_direction == "DOWN" and self.gravity_state == -1.0 and self.cooldown_frames == 0:
            self.gravity_state = 1.0
            self.target_camera_tilt = 15.0
            self.cooldown_frames = 12
            if self.sound_flip: self.sound_flip.play()

        # Update smooth interpolation dynamics
        self.current_camera_tilt = self.current_camera_tilt * 0.82 + self.target_camera_tilt * 0.18
        self.target_camera_tilt *= 0.85
        self.current_gravity_vector = self.current_gravity_vector * 0.72 + self.gravity_state * 0.28
        
        target_y = self.floor_y - 30 if self.gravity_state == 1.0 else self.ceiling_y + 30
        self.player_y = self.player_y * 0.65 + target_y * 0.35
        self.trail_buffer.append((int(self.player_x), int(self.player_y)))

        if len(self.obstacles) == 0 or self.obstacles[-1].x < FRAME_WIDTH - random.randint(320, 680):
            type_id = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
            self.obstacles.append(Obstacle(FRAME_WIDTH, type_id, self.game_speed))

        for obs in self.obstacles:
            obs.update(dt)
            ox, oy, ow, oh = obs.get_hitbox(self.floor_y, self.ceiling_y)
            if np.sqrt((self.player_x - max(ox, min(self.player_x, ox + ow)))**2 + (self.player_y - max(oy, min(self.player_y, oy + oh)))**2) < self.player_radius:
                self.current_state = "GAMEOVER"
                self.cooldown_frames = 40
                if self.sound_crash: self.sound_crash.play()

        self.obstacles = [obs for obs in self.obstacles if obs.x > -100]

    def generate_cyber_hud(self, base_canvas):
        cv2.line(base_canvas, (0, self.floor_y), (FRAME_WIDTH, self.floor_y), COLOR_FLOOR, 4)
        cv2.line(base_canvas, (0, self.ceiling_y), (FRAME_WIDTH, self.ceiling_y), COLOR_CEILING, 4)
        for i in range(-1, int(FRAME_WIDTH // 80) + 2):
            curr_x = int(i * 80 - self.bg_scroll_offset)
            cv2.line(base_canvas, (curr_x, self.floor_y), (curr_x + 40, FRAME_HEIGHT), COLOR_FLOOR, 1)
            cv2.line(base_canvas, (curr_x, self.ceiling_y), (curr_x - 40, 0), COLOR_CEILING, 1)

        if self.current_state == "PLAYING":
            for idx, pt in enumerate(self.trail_buffer):
                if idx > 0: cv2.line(base_canvas, list(self.trail_buffer)[idx - 1], pt, COLOR_NEON_CYAN if self.gravity_state == 1.0 else COLOR_NEON_MAGENTA, int(1 + (idx / len(self.trail_buffer)) * 8), cv2.LINE_AA)
            cv2.circle(base_canvas, (int(self.player_x), int(self.player_y)), self.player_radius, COLOR_NEON_GREEN if self.cooldown_frames == 0 else COLOR_WHITE, -1)
            cv2.circle(base_canvas, (int(self.player_x), int(self.player_y)), self.player_radius - 5, COLOR_BG_DARK, -1)
            for obs in self.obstacles: obs.draw(base_canvas, self.floor_y, self.ceiling_y)

        # Contextual UI Latch Engine Overlay
        if self.current_state == "START":
            cv2.putText(base_canvas, "FLUX-RUN ENGINE GENERATION V3", (280, FRAME_HEIGHT // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_NEON_CYAN, 3, cv2.LINE_AA)
            cv2.putText(base_canvas, "SHOW SPREAD OPEN PALM TO START", (380, FRAME_HEIGHT // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
        elif self.current_state == "GAMEOVER":
            cv2.putText(base_canvas, "GEOMETRY RE-ALIGNMENT FAILURE", (280, FRAME_HEIGHT // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_SPIKE, 3, cv2.LINE_AA)
            cv2.putText(base_canvas, f"FINAL RECORDED SCORE: {self.score}", (460, FRAME_HEIGHT // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NEON_CYAN, 2, cv2.LINE_AA)
            cv2.putText(base_canvas, "SHOW OPEN PALM TO RESET ENGINE PIPELINE", (340, FRAME_HEIGHT // 2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
        else:
            cv2.putText(base_canvas, f"SCORE: {self.score}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_NEON_CYAN, 2, cv2.LINE_AA)
            dir_text = f"INDEX VECTOR: {self.local_finger_direction}"
            cv2.putText(base_canvas, dir_text, (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NEON_GREEN if self.local_finger_direction != "NONE" else COLOR_WHITE, 2, cv2.LINE_AA)

    def run(self):
        while self.cap.isOpened():
            current_timestamp = time.time()
            dt = current_timestamp - self.last_frame_time
            self.last_frame_time = current_timestamp
            
            success, frame = self.cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)

            with self.thread_lock:
                self.current_working_frame = frame.copy()
                self.local_palm_detected = self.shared_palm_detected
                self.local_finger_direction = self.shared_finger_direction
                self.local_landmarks = self.shared_landmarks

            self.process_state(dt)

            base_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            base_canvas[:] = COLOR_BG_DARK
            self.generate_cyber_hud(base_canvas)

            # Draw corner camera monitor pipeline details
            cam_preview = cv2.resize(frame, (240, 135))
            if self.local_landmarks:
                for pt in self.local_landmarks:
                    cv2.circle(cam_preview, (int(pt[0] * 240), int(pt[1] * 135)), 2, COLOR_NEON_GREEN, -1)
            cv2.rectangle(cam_preview, (0, 0), (240, 135), COLOR_NEON_CYAN, 2)
            base_canvas[20:155, FRAME_WIDTH - 260:FRAME_WIDTH - 20] = cam_preview

            if abs(self.current_camera_tilt) > 0.1:
                base_canvas = cv2.warpAffine(base_canvas, cv2.getRotationMatrix2D((FRAME_WIDTH // 2, FRAME_HEIGHT // 2), self.current_camera_tilt, 1.0), (FRAME_WIDTH, FRAME_HEIGHT), borderValue=COLOR_BG_DARK)

            cv2.imshow("Project Flux-Run // Sayyam AI Lab", base_canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.keep_running = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    FluxEngine().run()