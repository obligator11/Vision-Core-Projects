import cv2
import numpy as np
import pygame
import threading
import time
from ultralytics import YOLO

# --- Configuration & Tuning ---
WINDOW_W, WINDOW_H = 1280, 720
NUM_PARTICLES = 3000
GRAVITY = 0.5
BOUNCE_FRICTION = 0.6
MAX_VELOCITY = 15.0
GRAVITY_WELL_STRENGTH = 0.05
OBJECT_AREA_THRESHOLD = 5000  # Minimum area to trigger a gravity well

class ParticleEngine:
    def __init__(self, count, width, height):
        self.count = count
        self.width = width
        self.height = height
        
        # Vectorized particle states
        self.pos_x = np.random.uniform(0, width, count)
        self.pos_y = np.random.uniform(-height, 0, count)
        self.vel_x = np.zeros(count)
        self.vel_y = np.random.uniform(1, 3, count)
        self.colors = np.full((count, 3), (0, 255, 255), dtype=np.uint8)

    def update(self, collision_mask, gravity_wells):
        # 1. Apply global downward gravity
        self.vel_y += GRAVITY
        
        # 2. Apply localized Gravity Wells (from heavy physical objects)
        for well in gravity_wells:
            cx, cy, mass = well
            dx = cx - self.pos_x
            dy = cy - self.pos_y
            dist_sq = dx**2 + dy**2 + 1e-5 
            
            force = (GRAVITY_WELL_STRENGTH * mass) / np.sqrt(dist_sq)
            
            self.vel_x += force * (dx / np.sqrt(dist_sq))
            self.vel_y += force * (dy / np.sqrt(dist_sq))

        # Clamp velocities
        self.vel_x = np.clip(self.vel_x, -MAX_VELOCITY, MAX_VELOCITY)
        self.vel_y = np.clip(self.vel_y, -MAX_VELOCITY, MAX_VELOCITY)

        # 3. Update Positions
        self.pos_x += self.vel_x
        self.pos_y += self.vel_y

        # 4. Handle Screen Boundaries
        self.pos_x = self.pos_x % self.width
        
        fallen = self.pos_y > self.height
        self.pos_y[fallen] = np.random.uniform(-50, 0, np.sum(fallen))
        self.vel_y[fallen] = np.random.uniform(1, 3, np.sum(fallen))
        self.vel_x[fallen] = 0

        # 5. Fast Mask-Based Collisions
        on_screen = (self.pos_y >= 0) & (self.pos_y < self.height)
        valid_y = self.pos_y[on_screen].astype(int)
        valid_x = self.pos_x[on_screen].astype(int)
        
        hits = collision_mask[valid_y, valid_x] > 0
        
        if np.any(hits):
            hit_indices = np.where(on_screen)[0][hits]
            self.vel_y[hit_indices] *= -BOUNCE_FRICTION
            self.vel_x[hit_indices] += np.random.uniform(-1, 1, len(hit_indices))
            self.pos_y[hit_indices] -= 2 
            
        return np.sum(hits) 

class VisionThread:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.model = YOLO('yolov8n.pt') # Auto-downloads tiny weights safely
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.running = True
        self.current_frame = None
        self.collision_mask = np.zeros((height, width), dtype=np.uint8)
        self.gravity_wells = []
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1) 
            frame = cv2.resize(frame, (self.width, self.height))
            
            new_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            new_wells = []
            
            if frame_count % 3 == 0:
                results = self.model(frame, stream=True, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0: continue
                        
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 500:
                                contour += np.array([x1, y1])
                                cv2.drawContours(new_mask, [contour], -1, 255, thickness=cv2.FILLED)
                                
                                if area > OBJECT_AREA_THRESHOLD:
                                    M = cv2.moments(contour)
                                    if M["m00"] != 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        new_wells.append((cx, cy, area))
            
            with self.lock:
                self.current_frame = frame.copy()
                if frame_count % 3 == 0:
                    self.collision_mask = new_mask
                    self.gravity_wells = new_wells
            
            frame_count += 1
            time.sleep(0.01)

    def get_data(self):
        with self.lock:
            return self.current_frame, self.collision_mask.copy(), list(self.gravity_wells)

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class AudioSynthesizer:
    """Generates procedural sound waves entirely in memory."""
    @staticmethod
    def create_ambient_drone():
        sample_rate = 44100
        duration = 1.0 # 1 second looped
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Deep 55Hz base with a 110Hz overtone
        wave = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.25 * np.sin(2 * np.pi * 110 * t)
        audio_data = np.int16(wave * 32767)
        return pygame.sndarray.make_sound(np.column_stack((audio_data, audio_data)))

    @staticmethod
    def create_sand_collision():
        sample_rate = 44100
        duration = 0.05 # Ultra-short snap
        samples = int(sample_rate * duration)
        # White noise
        noise = np.random.uniform(-0.3, 0.3, samples)
        # Fade out envelope
        envelope = np.linspace(1.0, 0.0, samples)
        audio_data = np.int16((noise * envelope) * 32767)
        return pygame.sndarray.make_sound(np.column_stack((audio_data, audio_data)))

class GravityWellGame:
    def __init__(self):
        # 1. Advanced Audio Setup (Low Latency)
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        pygame.display.set_caption("GravityWell AR (Synthesized Audio)")
        self.clock = pygame.time.Clock()
        
        # 2. Procedural Audio Assets (No downloads required)
        self.ambient_hum = AudioSynthesizer.create_ambient_drone()
        self.sand_sound = AudioSynthesizer.create_sand_collision()
        
        self.hum_channel = self.ambient_hum.play(loops=-1)
        self.hum_channel.set_volume(0.1)
        self.sand_channel = pygame.mixer.Channel(1)

        # 3. Engines
        self.vision = VisionThread(WINDOW_W, WINDOW_H)
        self.particles = ParticleEngine(NUM_PARTICLES, WINDOW_W, WINDOW_H)
        
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            frame, collision_mask, gravity_wells = self.vision.get_data()
            if frame is None:
                continue

            hit_count = self.particles.update(collision_mask, gravity_wells)

            # Dynamic Audio Modulation
            if self.hum_channel:
                target_hum = min(0.8, 0.1 + (len(gravity_wells) * 0.2))
                current_hum = self.hum_channel.get_volume()
                self.hum_channel.set_volume(current_hum + (target_hum - current_hum) * 0.1)
                
            if self.sand_channel:
                if hit_count > 10 and not self.sand_channel.get_busy():
                    self.sand_channel.play(self.sand_sound)
                self.sand_channel.set_volume(min(1.0, hit_count / 200.0))

            # Rendering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            frame_surface = pygame.transform.flip(frame_surface, True, False)
            
            win_w, win_h = self.screen.get_size()
            scaled_bg = pygame.transform.scale(frame_surface, (win_w, win_h))
            self.screen.blit(scaled_bg, (0, 0))

            for well in gravity_wells:
                cx, cy, mass = well
                draw_x = int(cx * (win_w / WINDOW_W))
                draw_y = int(cy * (win_h / WINDOW_H))
                radius = min(100, int(mass / 500))
                pygame.draw.circle(self.screen, (255, 0, 255), (draw_x, draw_y), radius, 2)

            px = (self.particles.pos_x * (win_w / WINDOW_W)).astype(int)
            py = (self.particles.pos_y * (win_h / WINDOW_H)).astype(int)
            
            valid = (py >= 0) & (py < win_h) & (px >= 0) & (px < win_w)
            
            for x, y in zip(px[valid], py[valid]):
                pygame.draw.rect(self.screen, (0, 255, 255), (x, y, 2, 2))

            pygame.display.flip()
            self.clock.tick(60)

        self.vision.stop()
        pygame.quit()

if __name__ == "__main__":
    game = GravityWellGame()
    game.run()