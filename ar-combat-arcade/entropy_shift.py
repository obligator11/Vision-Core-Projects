import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time
import threading
import queue
from enum import Enum, auto

class GameState(Enum):
    STANDBY = auto()
    REMAP_FLASH = auto()
    PLAYING = auto()
    GAME_OVER = auto()

class Action(Enum):
    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    DEFEND = auto()
    SHOOT = auto()

def generate_tone(frequency, duration, volume=0.5, wave_type="sine"):
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, False)
    
    if wave_type == "sine":
        samples = np.sin(2 * np.pi * frequency * t)
    elif wave_type == "square":
        samples = np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == "noise":
        samples = random.uniform(-1.0, 1.0) * np.ones_like(t)
        decay = np.exp(-4 * t)
        samples = samples * decay
    else:
        samples = np.sin(2 * np.pi * frequency * t)
        
    audio_buffer = (samples * volume * 32767).astype(np.int16)
    stereo_buffer = np.column_stack((audio_buffer, audio_buffer))
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_buffer))

class AudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self.snd_start = generate_tone(587.33, 0.4, 0.4, "sine")
        self.snd_remap = generate_tone(880.0, 0.15, 0.5, "square")
        self.snd_shoot = generate_tone(1200.0, 0.08, 0.2, "sine")
        self.snd_defend = generate_tone(330.0, 0.2, 0.3, "sine")
        self.snd_left = generate_tone(440.0, 0.05, 0.25, "sine")
        self.snd_right = generate_tone(493.88, 0.05, 0.25, "sine")
        self.snd_hit = generate_tone(150.0, 0.25, 0.6, "noise")
        self.snd_fail = generate_tone(110.0, 0.6, 0.5, "square")

class Threat:
    def __init__(self, width, height, level_chaos):
        self.x = random.randint(50, width - 50)
        self.y = 0
        self.speed = random.uniform(4.0, 7.0) + (level_chaos * 1.5)
        self.radius = random.randint(15, 25)
        self.type = random.choice([Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.DEFEND, Action.SHOOT])

    def update(self):
        self.y += self.speed

    def draw(self, frame):
        color = (0, 0, 255)
        if self.type == Action.DEFEND:
            color = (0, 255, 255)
        elif self.type == Action.SHOOT:
            color = (255, 0, 255)
        elif self.type == Action.MOVE_LEFT:
            color = (255, 128, 0)
        elif self.type == Action.MOVE_RIGHT:
            color = (0, 128, 255)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius + 4, color, 2)

class EntropyEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width == 0:
            self.width, self.height = 1280, 720
            
        cv2.namedWindow("Sayyam AI Lab - Project Entropy-Shift", cv2.WINDOW_NORMAL)
        
        self.audio = AudioEngine()
        self.state = GameState.STANDBY
        
        self.frame_q = queue.Queue(maxsize=2)
        self.coord_q = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        
        self.gestures = ["FIST", "OPEN_PALM", "THUMBS_UP", "PEACE"]
        self.actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.DEFEND, Action.SHOOT]
        self.mapping = {}
        self.scramble_mappings()
        
        self.score = 0
        self.chaos_factor = 0.0
        self.threats = []
        
        self.last_remap_time = time.time()
        self.remap_interval = 9.0
        self.flash_start_time = 0.0
        
        self.active_vfx = []
        self.tracking_thread = threading.Thread(target=self._vision_pipeline, daemon=True)
        self.tracking_thread.start()

    def scramble_mappings(self):
        shuffled_actions = list(self.actions)
        random.shuffle(shuffled_actions)
        self.mapping = {self.gestures[i]: shuffled_actions[i] for i in range(len(self.gestures))}

    def _vision_pipeline(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_q.get(timeout=0.1)
            except queue.Empty:
                continue
                
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            payload = []
            
            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    landmarks = hand_lms.landmark
                    h_type = results.multi_handedness[idx].classification[0].label
                    
                    coords = {}
                    for i, lm in enumerate(landmarks):
                        coords[i] = (int(lm.x * self.width), int(lm.y * self.height))
                        
                    tip_ids = [4, 8, 12, 16, 20]
                    pip_ids = [3, 6, 10, 14, 18]
                    
                    is_extended = []
                    for t_id, p_id in zip(tip_ids[1:], pip_ids[1:]):
                        is_extended.append(coords[t_id][1] < coords[p_id][1])
                        
                    thumb_extended = coords[4][0] < coords[3][0] if h_type == "Right" else coords[4][0] > coords[3][0]
                    is_extended.insert(0, thumb_extended)
                    
                    detected_gesture = "UNKNOWN"
                    if sum(is_extended) == 0:
                        detected_gesture = "FIST"
                    elif sum(is_extended) == 5:
                        detected_gesture = "OPEN_PALM"
                    elif is_extended[0] and sum(is_extended[1:]) == 0:
                        detected_gesture = "THUMBS_UP"
                    elif is_extended[1] and is_extended[2] and sum(is_extended) == 2:
                        detected_gesture = "PEACE"
                        
                    payload.append({
                        "wrist": coords[0],
                        "index_tip": coords[8],
                        "gesture": detected_gesture,
                        "type": h_type
                    })
            if not self.coord_q.full():
                self.coord_q.put(payload)

    def execute_action(self, action):
        if action == Action.MOVE_LEFT:
            self.audio.snd_left.play()
            self.active_vfx.append(("WAVE_LEFT", time.time(), (255, 128, 0)))
        elif action == Action.MOVE_RIGHT:
            self.audio.snd_right.play()
            self.active_vfx.append(("WAVE_RIGHT", time.time(), (0, 128, 255)))
        elif action == Action.DEFEND:
            self.audio.snd_defend.play()
            self.active_vfx.append(("SHIELD", time.time(), (0, 255, 255)))
        elif action == Action.SHOOT:
            self.audio.snd_shoot.play()
            self.active_vfx.append(("BEAM", time.time(), (255, 0, 255)))

    def render_hud(self, frame, active_telemetry):
        overlay = frame.copy()
        
        if self.state == GameState.STANDBY:
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (20, 10, 10), -1)
            cv2.putText(overlay, "PROJECT ENTROPY-SHIFT", (160, self.height // 2 - 80), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 215, 255), 3, cv2.LINE_AA)
            cv2.putText(overlay, "CLAP BOTH HANDS TO INITIALIZE MATRIX", (140, self.height // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "SYSTEM ENGINE: ASYNCHRONOUS GIL BYPASS 60+ FPS", (240, self.height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
            
        elif self.state == GameState.REMAP_FLASH or self.state == GameState.PLAYING:
            cv2.rectangle(overlay, (0, 0), (self.width, 80), (15, 15, 15), -1)
            cv2.line(overlay, (0, 80), (self.width, 80), (0, 215, 255), 2)
            
            cv2.putText(overlay, f"SCORE: {self.score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"CHAOS: {int(self.chaos_factor * 100)}%", (240, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            
            time_since_scramble = time.time() - self.last_remap_time
            progress = max(0.0, min(1.0, time_since_scramble / self.remap_interval))
            bar_w = int(300 * (1.0 - progress))
            cv2.rectangle(overlay, (self.width - 340, 32), (self.width - 40, 48), (30, 30, 30), -1)
            cv2.rectangle(overlay, (self.width - 340, 32), (self.width - 340 + bar_w, 48), (0, 215, 255), -1)
            cv2.putText(overlay, "REMAP CYCLE", (self.width - 340, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            y_offset = 140
            cv2.rectangle(overlay, (20, y_offset - 30), (280, y_offset + 160), (10, 10, 10), -1)
            cv2.rectangle(overlay, (20, y_offset - 30), (280, y_offset + 160), (150, 150, 150), 1)
            cv2.putText(overlay, "CURRENT COGNITIVE MAP", (30, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
            
            for idx, (gest, act) in enumerate(self.mapping.items()):
                cv2.putText(overlay, f"{gest} -> {act.name}", (30, y_offset + 30 + (idx * 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if self.state == GameState.REMAP_FLASH:
                flash_elapsed = time.time() - self.flash_start_time
                alpha = max(0.0, 1.0 - (flash_elapsed / 0.6))
                flash_layer = frame.copy()
                cv2.rectangle(flash_layer, (0, 0), (self.width, self.height), (0, 165, 255), -1)
                cv2.addWeighted(flash_layer, alpha * 0.4, overlay, 1.0 - (alpha * 0.4), 0, overlay)
                cv2.putText(overlay, "!!! COGNITIVE REMAP TRIGGERED !!!", (140, self.height // 2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        elif self.state == GameState.GAME_OVER:
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (5, 5, 20), -1)
            cv2.putText(overlay, "CRITICAL COGNITIVE OVERLOAD", (110, self.height // 2 - 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(overlay, f"FINAL MATRIX SCORE: {self.score}", (240, self.height // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "CLAP BOTH HANDS TO REBOOT SIMULATION", (140, self.height // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

        for data in active_telemetry:
            wrist = data["wrist"]
            gesture = data["gesture"]
            cv2.circle(overlay, wrist, 12, (0, 215, 255), 2)
            cv2.putText(overlay, f"{data['type']} : {gesture}", (wrist[0] - 50, wrist[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        current_time = time.time()
        self.active_vfx = [vfx for vfx in self.active_vfx if current_time - vfx[1] < 0.4]
        for vfx_type, start_t, color in self.active_vfx:
            elapsed = current_time - start_t
            scale = elapsed / 0.4
            
            if vfx_type == "SHIELD":
                thick = max(1, int(8 * (1.0 - scale)))
                cv2.circle(overlay, (self.width // 2, self.height // 2 + 100), int(100 * scale), color, thick)
            elif vfx_type == "BEAM":
                y_pos = int((self.height // 2 + 100) * (1.0 - scale))
                thick = max(1, int(12 * (1.0 - scale)))
                cv2.line(overlay, (self.width // 2, self.height // 2 + 100), (self.width // 2, y_pos), color, thick)
            elif vfx_type == "WAVE_LEFT":
                x_pos = int(self.width // 2 - (200 * scale))
                thick = max(1, int(6 * (1.0 - scale)))
                cv2.ellipse(overlay, (x_pos, self.height // 2 + 100), (20, 60), 0, 0, 360, color, thick)
            elif vfx_type == "WAVE_RIGHT":
                x_pos = int(self.width // 2 + (200 * scale))
                thick = max(1, int(6 * (1.0 - scale)))
                cv2.ellipse(overlay, (x_pos, self.height // 2 + 100), (20, 60), 0, 0, 360, color, thick)

        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    def process_frame(self):
        success, frame = self.cap.read()
        if not success:
            return True

        frame = cv2.flip(frame, 1)
        
        if not self.frame_q.full():
            self.frame_q.put(frame.copy())
            
        telemetry = []
        try:
            telemetry = self.coord_q.get_nowait()
        except queue.Empty:
            pass

        if self.state == GameState.STANDBY or self.state == GameState.GAME_OVER:
            if len(telemetry) == 2:
                w1 = telemetry[0]["wrist"]
                w2 = telemetry[1]["wrist"]
                dist = np.linalg.norm(np.array(w1) - np.array(w2))
                if dist < 85:
                    self.audio.snd_start.play()
                    self.score = 0
                    self.chaos_factor = 0.0
                    self.threats = []
                    self.last_remap_time = time.time()
                    self.remap_interval = 9.0
                    self.scramble_mappings()
                    self.state = GameState.PLAYING
                    
        elif self.state == GameState.PLAYING or self.state == GameState.REMAP_FLASH:
            current_time = time.time()
            
            if self.state == GameState.REMAP_FLASH and current_time - self.flash_start_time > 0.6:
                self.state = GameState.PLAYING

            if current_time - self.last_remap_time > self.remap_interval:
                self.scramble_mappings()
                self.audio.snd_remap.play()
                self.last_remap_time = current_time
                self.flash_start_time = current_time
                self.state = GameState.REMAP_FLASH
                self.remap_interval = max(4.0, 9.0 - (self.chaos_factor * 6.0))

            self.chaos_factor = min(1.0, self.score / 35.0)
            
            spawn_chance = 0.02 + (self.chaos_factor * 0.04)
            if random.random() < spawn_chance and len(self.threats) < 5:
                self.threats.append(Threat(self.width, self.height, self.chaos_factor))

            current_actions = set()
            for hand in telemetry:
                gest = hand["gesture"]
                if gest in self.mapping:
                    act = self.mapping[gest]
                    current_actions.add(act)
                    self.execute_action(act)

            for threat in list(self.threats):
                threat.update()
                threat.draw(frame)
                
                if threat.y > self.height - 120:
                    if threat.type in current_actions:
                        self.score += 1
                        self.threats.remove(threat)
                    elif threat.y > self.height:
                        self.audio.snd_hit.play()
                        self.threats.remove(threat)
                        if threat.type != Action.SHOOT and threat.type != Action.DEFEND:
                            self.audio.snd_fail.play()
                            self.state = GameState.GAME_OVER

        self.render_hud(frame, telemetry)
        cv2.imshow("Sayyam AI Lab - Project Entropy-Shift", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        return True

    def clean_up(self):
        self.stop_event.set()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    engine = EntropyEngine()
    running = True
    while running:
        running = engine.process_frame()
    engine.clean_up()