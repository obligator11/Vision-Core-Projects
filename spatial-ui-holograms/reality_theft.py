import cv2
import numpy as np
import pygame
import mediapipe as mp
from ultralytics import YOLO
import threading
import time
import math
import sys

# =====================================================================
# AUDIO SYNTHESIS & SOUND MANAGER (Zero-Latency Fallback)
# =====================================================================
class SoundManager:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self.sample_rate = 44100
        
        # Synthesize sounds to guarantee execution without missing .wav files
        self.snd_locked = self._create_sound(freq=880, duration=0.1, wave_type='sine')
        self.snd_snatch = self._create_sound(freq=220, duration=0.3, wave_type='sawtooth')
        self.snd_bounce = self._create_sound(freq=440, duration=0.1, wave_type='square')
        
    def _create_sound(self, freq, duration, wave_type='sine'):
        """Mathematically generates a sound wave array and returns a Pygame Sound."""
        frames = int(self.sample_rate * duration)
        t = np.linspace(0, duration, frames, False)
        
        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * freq * t)
        elif wave_type == 'square':
            wave = np.sign(np.sin(2 * np.pi * freq * t))
        elif wave_type == 'sawtooth':
            wave = 2 * (t * freq - np.floor(t * freq + 0.5))
            
        # Apply envelope (fade out)
        envelope = np.linspace(1.0, 0.0, frames)
        wave = wave * envelope
        
        # Convert to 16-bit stereo
        audio_array = np.zeros((frames, 2), dtype=np.int16)
        audio_array[:, 0] = (wave * 32767 * 0.3).astype(np.int16)
        audio_array[:, 1] = audio_array[:, 0]
        
        return pygame.sndarray.make_sound(audio_array)

    def play_locked(self):
        self.snd_locked.play()

    def play_snatch(self):
        self.snd_snatch.play()

    def play_bounce(self, velocity):
        """Scale volume dynamically based on impact velocity."""
        vol = min(1.0, max(0.1, abs(velocity) / 1000.0))
        self.snd_bounce.set_volume(vol)
        self.snd_bounce.play()


# =====================================================================
# 2D PHYSICS SPRITE ENGINE
# =====================================================================
class DigitalSprite:
    def __init__(self, surface, x, y):
        self.surface = surface
        self.rect = self.surface.get_rect(center=(x, y))
        self.vx = np.random.uniform(-300, 300)  # Initial lateral burst
        self.vy = np.random.uniform(-500, -200) # Initial upward burst
        
        self.gravity = 1800.0  # Pixels per second squared
        self.restitution = 0.7 # Bounciness (energy retained)
        self.x = float(self.rect.x)
        self.y = float(self.rect.y)

    def update(self, dt, bounds_w, bounds_h, sound_manager):
        # Apply Gravity
        self.vy += self.gravity * dt
        
        # Integrate Velocity
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Floor Collision
        if self.y + self.rect.height > bounds_h:
            self.y = bounds_h - self.rect.height
            if abs(self.vy) > 100:  # Only bounce if velocity is high enough
                sound_manager.play_bounce(self.vy)
            self.vy *= -self.restitution
            self.vx *= 0.98 # Friction
            
        # Wall Collisions (Left / Right)
        if self.x < 0:
            self.x = 0
            self.vx *= -self.restitution
        elif self.x + self.rect.width > bounds_w:
            self.x = bounds_w - self.rect.width
            self.vx *= -self.restitution
            
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

    def draw(self, screen):
        screen.blit(self.surface, self.rect)


# =====================================================================
# BACKGROUND YOLO INFERENCE THREAD (Prevents Frame Drops)
# =====================================================================
class YOLOWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.model = YOLO('yolov8n-seg.pt')
        self.current_frame = None
        self.result_mask = None
        self.result_box = None
        self.lock = threading.Lock()
        self.running = True

    def update_frame(self, frame):
        with self.lock:
            # Downscale frame for faster inference
            self.current_frame = cv2.resize(frame, (320, 240))

    def get_results(self):
        with self.lock:
            return self.result_mask, self.result_box

    def run(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()
                    self.current_frame = None
            
            if frame_to_process is not None:
                # Run inference
                results = self.model.predict(frame_to_process, verbose=False)
                
                best_mask = None
                best_box = None
                best_conf = 0.0
                
                if results and len(results[0].masks) > 0 if results[0].masks else False:
                    # Find highest confidence object
                    for i, mask_data in enumerate(results[0].masks.data):
                        # --- THE FIX: Ignore humans ---
                        cls_id = int(results[0].boxes.cls[i])
                        if cls_id == 0:  # Class 0 is 'person' in YOLO/COCO
                            continue
                        # ------------------------------
                        
                        conf = float(results[0].boxes.conf[i])
                        if conf > best_conf:
                            best_conf = conf
                            best_mask = mask_data.cpu().numpy()
                            # Map bounding box back to relative coordinates (0.0 to 1.0)
                            box = results[0].boxes.xyxyn[i].cpu().numpy()
                            best_box = box

                with self.lock:
                    self.result_mask = best_mask
                    self.result_box = best_box
            else:
                time.sleep(0.01)


# =====================================================================
# MAIN APPLICATION ENGINE
# =====================================================================
class RealityThiefApp:
    def __init__(self):
        pygame.init()
        self.sounds = SoundManager()
        
        # Display configuration
        self.BASE_W, self.BASE_H = 1280, 720
        self.screen = pygame.display.set_mode((self.BASE_W, self.BASE_H), pygame.RESIZABLE)
        pygame.display.set_caption("Reality Thief: Mixed-Reality Physics Engine")
        self.clock = pygame.time.Clock()

        # Camera pipeline
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.BASE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.BASE_H)

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7,
            max_num_hands=1
        )

        # Background Thread for YOLO Segmentation
        self.yolo_thread = YOLOWorker()
        self.yolo_thread.start()

        self.sprites = []
        
        # Inpainting / Illusion State
        self.frozen_patch = None
        self.frozen_mask = None
        self.freeze_end_time = 0
        
        self.pinch_threshold = 60  # Pixel distance to trigger a grab
        self.cooldown = 0

    def calculate_pinch(self, hand_landmarks, w, h):
        """Calculates distance between index (8) and thumb (4) and returns midpoint."""
        idx = hand_landmarks.landmark[8]
        thb = hand_landmarks.landmark[4]
        
        x8, y8 = int(idx.x * w), int(idx.y * h)
        x4, y4 = int(thb.x * w), int(thb.y * h)
        
        dist = math.hypot(x8 - x4, y8 - y4)
        mid_x, mid_y = (x8 + x4) // 2, (y8 + y4) // 2
        return dist, mid_x, mid_y

    def perform_extraction(self, frame, mask_full_res, bbox_pixels):
        """Erases object from feed and converts it to a Pygame physics sprite."""
        self.sounds.play_snatch()
        x1, y1, x2, y2 = bbox_pixels
        
        # 1. Create the Sprite (RGBA)
        sub_mask = mask_full_res[y1:y2, x1:x2]
        sub_frame = frame[y1:y2, x1:x2]
        
        rgba = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = (sub_mask * 255).astype(np.uint8) # Apply mask as alpha channel
        
        # Convert to Pygame Surface
        rgb_swapped = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
        surf = pygame.image.frombuffer(rgb_swapped.tobytes(), rgb_swapped.shape[1::-1], "RGBA")
        
        # Calculate center to spawn the sprite right where it was grabbed
        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2
        self.sprites.append(DigitalSprite(surf, center_x, center_y))

        # 2. Perform the Inpainting Illusion
        # Dilate mask to ensure edges are fully covered
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(mask_full_res, kernel, iterations=1)
        
        # Inpaint to erase the object
        inpainted_frame = cv2.inpaint(frame, dilated_mask, 5, cv2.INPAINT_TELEA)
        
        # Freeze the inpainted region to hold the illusion
        self.frozen_patch = inpainted_frame
        self.frozen_mask = dilated_mask.astype(bool)
        self.freeze_end_time = time.time() + 3.0 # Hold for 3 seconds

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS Delta time
            
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            # Capture Video
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1) # Mirror naturally
            h, w, _ = frame.shape
            
            # 1. Dispatch low-res frame to YOLO worker
            self.yolo_thread.update_frame(frame)
            
            # 2. Retrieve latest asynchronous YOLO data
            y_mask, y_box = self.yolo_thread.get_results()
            
            mask_full_res = None
            bbox_pixels = None

            if y_mask is not None and y_box is not None:
                # Resize YOLO mask (320x240) back up to Full Camera Res
                mask_full_res = cv2.resize(y_mask, (w, h))
                mask_full_res = (mask_full_res > 0.5).astype(np.uint8)
                
                x1 = max(0, int(y_box[0] * w))
                y1 = max(0, int(y_box[1] * h))
                x2 = min(w, int(y_box[2] * w))
                y2 = min(h, int(y_box[3] * h))
                bbox_pixels = (x1, y1, x2, y2)

                # Draw glowing highlight over target
                contours, _ = cv2.findContours(mask_full_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay = frame.copy()
                cv2.drawContours(overlay, contours, -1, (255, 255, 0), 4) # Neon Cyan
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # 3. Maintain Inpainting Illusion
            if self.frozen_patch is not None and time.time() < self.freeze_end_time:
                # Apply the frozen background over the current frame where the mask was
                np.copyto(frame, self.frozen_patch, where=np.expand_dims(self.frozen_mask, axis=2))
            else:
                self.frozen_patch = None

            # 4. MediaPipe Hand Tracking & Interaction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = self.hands.process(rgb_frame)
            
            if mp_results.multi_hand_landmarks:
                for hand_landmarks in mp_results.multi_hand_landmarks:
                    # Draw a cool minimal UI dot on the fingertips
                    idx_x, idx_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                    cv2.circle(rgb_frame, (idx_x, idx_y), 8, (0, 255, 255), -1)

                    dist, mid_x, mid_y = self.calculate_pinch(hand_landmarks, w, h)
                    
                    if self.cooldown > 0:
                        self.cooldown -= 1

                    # Trigger Extraction: Pinch is tight + cooldown is 0 + YOLO found something
                    if dist < self.pinch_threshold and self.cooldown == 0 and bbox_pixels is not None:
                        x1, y1, x2, y2 = bbox_pixels
                        
                        # Check if pinch is inside the object's overall Bounding Box
                        # This guarantees a catch even if your hand occludes the YOLO mask
                        if x1 <= mid_x <= x2 and y1 <= mid_y <= y2:
                            self.perform_extraction(frame, mask_full_res, bbox_pixels)
                            self.cooldown = 60 # 1 second cooldown

                            
            # 5. Render Base Frame onto Pygame Surface
            frame_surface = pygame.image.frombuffer(rgb_frame.tobytes(), (w, h), "RGB")
            
            # Dynamic Resizing & Scaling constraints
            window_w, window_h = self.screen.get_size()
            scale_ratio = min(window_w / w, window_h / h)
            new_w = int(w * scale_ratio)
            new_h = int(h * scale_ratio)
            
            scaled_bg = pygame.transform.scale(frame_surface, (new_w, new_h))
            
            # Center the camera frame in the resizable window
            offset_x = (window_w - new_w) // 2
            offset_y = (window_h - new_h) // 2
            
            self.screen.fill((10, 10, 15)) # Dark border
            self.screen.blit(scaled_bg, (offset_x, offset_y))

            # 6. Update and Draw Physical Sprites
            for sprite in self.sprites:
                sprite.update(dt, w, h, self.sounds)
                
                # Scale sprite dynamically before drawing to match current window aspect
                scaled_sprite_surf = pygame.transform.scale(
                    sprite.surface, 
                    (int(sprite.rect.width * scale_ratio), int(sprite.rect.height * scale_ratio))
                )
                
                # Draw at scaled coordinates
                draw_x = offset_x + int(sprite.rect.x * scale_ratio)
                draw_y = offset_y + int(sprite.rect.y * scale_ratio)
                self.screen.blit(scaled_sprite_surf, (draw_x, draw_y))

            pygame.display.flip()

        self.cleanup()

    def cleanup(self):
        self.yolo_thread.running = False
        self.cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = RealityThiefApp()
    app.run()