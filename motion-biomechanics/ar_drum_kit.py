import cv2
import mediapipe as mp
from ultralytics import YOLO
import pygame
import numpy as np
import threading
import time
import sys
from collections import deque

# --- CONFIGURATION ---
CAM_WIDTH, CAM_HEIGHT = 640, 480
FPS_TARGET = 60
VELOCITY_THRESHOLD = 15  # Downward pixel velocity required to trigger a hit
HIT_COOLDOWN = 0.2       # Seconds

# COCO Dataset Mappings for YOLO
CLASS_MAP = {
    41: 'Snare',  # Cup
    67: 'HiHat',  # Cell phone
    73: 'Kick',   # Book
    63: 'Crash'   # Laptop
}

# --- AUDIO MANAGER ---
class AudioManager:
    def __init__(self):
        # Ultra-low latency Pygame mixer initialization
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)  # Allow overlapping hits
        
        self.sounds = {}
        self.load_or_synthesize()

    def load_or_synthesize(self):
        """Loads WAV files if present, otherwise synthesizes drum sounds using NumPy math."""
        sound_files = {'Kick': 'Kick.wav', 'Snare': 'Snare.wav', 'HiHat': 'HiHat.wav', 'Crash': 'Crash.wav'}
        
        for name, file in sound_files.items():
            try:
                self.sounds[name] = pygame.mixer.Sound(file)
                print(f"[AUDIO] Loaded {file}")
            except FileNotFoundError:
                print(f"[AUDIO] {file} not found. Synthesizing {name}...")
                self.sounds[name] = self.synthesize_drum(name)

    def synthesize_drum(self, drum_type):
        """Math-based drum synthesis for graceful fallback."""
        sample_rate = 44100
        if drum_type == 'Kick':
            duration = 0.3
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            freq = np.geomspace(150, 40, len(t))
            wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 15)
        elif drum_type == 'Snare':
            duration = 0.25
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 20)
            noise = np.random.uniform(-1, 1, len(t)) * np.exp(-t * 15)
            wave = (tone * 0.5 + noise * 0.5)
        elif drum_type == 'HiHat':
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            noise = np.random.uniform(-1, 1, len(t))
            wave = noise * np.exp(-t * 40)
        else: # Crash
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            noise = np.random.uniform(-1, 1, len(t))
            wave = noise * np.exp(-t * 3)
            
        # Convert to 16-bit stereo sound buffer
        audio_arr = np.int16(wave * 32767)
        stereo_arr = np.column_stack((audio_arr, audio_arr))
        return pygame.sndarray.make_sound(stereo_arr)

    def play(self, sound_name):
        if sound_name in self.sounds:
            pygame.mixer.find_channel(True).play(self.sounds[sound_name])

# --- VISION ENGINE (ASYNC THREAD) ---
class VisionEngine(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        # Load lightweight YOLOv8 nano
        print("[VISION] Booting YOLOv8n AI...")
        self.yolo = YOLO("yolov8n.pt") 
        
        self.running = True
        self.latest_frame = None
        self.latest_hands = []
        self.latest_objects = []
        self.frame_count = 0

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1) # Mirror for AR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. MediaPipe Hand Tracking (Every frame)
            hand_results = self.hands.process(rgb_frame)
            parsed_hands = []
            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    # Extract Index Finger Tip (Landmark 8)
                    idx_tip = hand_lms.landmark[8]
                    px, py = int(idx_tip.x * CAM_WIDTH), int(idx_tip.y * CAM_HEIGHT)
                    parsed_hands.append((px, py))
            
            # 2. YOLO Object Detection (Every 3 frames to save CPU)
            if self.frame_count % 3 == 0:
                results = self.yolo(frame, verbose=False, classes=list(CLASS_MAP.keys()))
                parsed_objects = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        drum_type = CLASS_MAP.get(cls_id, None)
                        if drum_type:
                            parsed_objects.append({'box': (x1, y1, x2, y2), 'type': drum_type})
                self.latest_objects = parsed_objects

            self.latest_hands = parsed_hands
            self.latest_frame = rgb_frame
            self.frame_count += 1

    def stop(self):
        self.running = False
        self.cap.release()

# --- MAIN GAME ENGINE ---
class DrumKitApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CAM_WIDTH, CAM_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Phantom Percussion: Spatial AR Drum Kit")
        self.clock = pygame.time.Clock()
        
        self.audio = AudioManager()
        self.vision = VisionEngine()
        self.vision.start()
        
        # State tracking
        self.finger_history = {} # Tracks previous Y positions for velocity
        self.cooldowns = {}      # Prevents multi-triggering
        self.ripples = []        # Visual VFX
        
        # Looper State
        self.is_recording = False
        self.is_playing = False
        self.record_start = 0
        self.play_start = 0
        self.recorded_track = []
        self.play_index = 0

    def calculate_velocity(self, finger_id, current_y):
        if finger_id not in self.finger_history:
            self.finger_history[finger_id] = current_y
            return 0
        velocity = current_y - self.finger_history[finger_id]
        self.finger_history[finger_id] = current_y
        return velocity # Positive means moving downwards

    def check_hit(self, px, py, objects):
        current_time = time.time()
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            drum_type = obj['type']
            
            # Check AABB Intersection
            if x1 < px < x2 and y1 < py < y2:
                if current_time - self.cooldowns.get(drum_type, 0) > HIT_COOLDOWN:
                    self.cooldowns[drum_type] = current_time
                    self.audio.play(drum_type)
                    self.ripples.append({'pos': (px, py), 'radius': 10, 'alpha': 255, 'color': (0, 255, 255)})
                    
                    # Log for recording loop
                    if self.is_recording:
                        self.recorded_track.append((current_time - self.record_start, drum_type))
                    return

    def update_looper(self):
        if self.is_playing and self.recorded_track:
            elapsed = time.time() - self.play_start
            if self.play_index < len(self.recorded_track):
                hit_time, drum_type = self.recorded_track[self.play_index]
                if elapsed >= hit_time:
                    self.audio.play(drum_type)
                    self.play_index += 1
            else:
                # Loop track
                self.play_start = time.time()
                self.play_index = 0

    def run(self):
        running = True
        font = pygame.font.SysFont('Arial', 24, bold=True)
        
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.is_recording = not self.is_recording
                        self.is_playing = False
                        if self.is_recording:
                            self.recorded_track.clear()
                            self.record_start = time.time()
                    elif event.key == pygame.K_p:
                        self.is_playing = not self.is_playing
                        self.is_recording = False
                        if self.is_playing:
                            self.play_start = time.time()
                            self.play_index = 0

            # 2. Frame Processing
            frame = self.vision.latest_frame
            if frame is not None:
                # Convert OpenCV RGB to Pygame Surface
                surf = pygame.surfarray.make_surface(np.rot90(frame))
                
                # Draw YOLO Boxes
                objects = self.vision.latest_objects
                for obj in objects:
                    x1, y1, x2, y2 = obj['box']
                    pygame.draw.rect(surf, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 3)
                    lbl = font.render(obj['type'], True, (0, 255, 0))
                    surf.blit(lbl, (x1, max(y1-25, 0)))

                # Process Hands & Hit Detection
                hands = self.vision.latest_hands
                for i, (px, py) in enumerate(hands):
                    pygame.draw.circle(surf, (255, 0, 0), (px, py), 8)
                    vy = self.calculate_velocity(i, py)
                    
                    if vy > VELOCITY_THRESHOLD:
                        self.check_hit(px, py, objects)

                # Render VFX Ripples
                for r in self.ripples[:]:
                    r['radius'] += 3
                    r['alpha'] -= 10
                    if r['alpha'] <= 0:
                        self.ripples.remove(r)
                    else:
                        # Pygame doesn't natively do alpha circles easily without a surface, doing a thin circle for now
                        pygame.draw.circle(surf, r['color'], r['pos'], r['radius'], 2)

                # UI Overlays
                ui_text = f"REC (R): {'ON' if self.is_recording else 'OFF'} | PLAY (P): {'ON' if self.is_playing else 'OFF'}"
                ui_color = (255, 50, 50) if self.is_recording else (50, 255, 50) if self.is_playing else (255, 255, 255)
                surf.blit(font.render(ui_text, True, ui_color), (10, 10))

                # Handle Resizable Window
                w, h = self.screen.get_size()
                scaled_surf = pygame.transform.smoothscale(surf, (w, h))
                self.screen.blit(scaled_surf, (0, 0))

            self.update_looper()
            pygame.display.flip()
            self.clock.tick(FPS_TARGET)

        # Cleanup
        self.vision.stop()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = DrumKitApp()
    app.run()