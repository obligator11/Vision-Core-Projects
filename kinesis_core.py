import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
from collections import deque
import threading
import pygame
import os
import wave
import struct

def generate_synthetic_hit_sound(filename="hit.wav"):
    if os.path.exists(filename):
        return
    sample_rate = 44100
    duration = 0.2
    num_samples = int(sample_rate * duration)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(num_samples):
            t = float(i) / sample_rate
            freq = 800 * math.exp(-20 * t)
            envelope = math.exp(-15 * t)
            tone = math.sin(2 * math.pi * freq * t)
            noise = random.uniform(-1, 1) * 0.4
            sample = (tone + noise) * envelope
            val = int(sample * 32767.0)
            val = max(-32768, min(32767, val))
            wav_file.writeframes(struct.pack('h', val))

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-25, 25)
        self.vy = random.uniform(-25, 25)
        self.life = 1.0
        self.color = color

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 1.2  
        self.life -= 0.05 

    def draw(self, frame):
        if self.life > 0:
            alpha = max(0, self.life)
            c = (int(self.color[0] * alpha), int(self.color[1] * alpha), int(self.color[2] * alpha))
            radius = max(1, int(6 * alpha))
            cv2.circle(frame, (int(self.x), int(self.y)), radius + 2, (0, 0, 0), -1)
            cv2.circle(frame, (int(self.x), int(self.y)), radius, c, -1)

class Target:
    def __init__(self, w, h):
        if random.choice([True, False]):
            self.x = random.randint(int(w * 0.05), int(w * 0.35))
        else:
            self.x = random.randint(int(w * 0.65), int(w * 0.95))
        self.y = random.randint(int(h * 0.1), int(h * 0.5))
        self.radius = 40
        self.color = (0, 215, 255)
        self.active = True
        self.spawn_time = time.time()

    def draw(self, frame):
        if not self.active: return
        
        elapsed = time.time() - self.spawn_time
        pulse = int(8 * math.sin(elapsed * 8))
        angle = int(elapsed * 120) % 360
        axes = (self.radius + pulse, self.radius + pulse)
        
        cv2.ellipse(frame, (self.x, self.y), axes, angle, 0, 100, (0,0,0), 7)
        cv2.ellipse(frame, (self.x, self.y), axes, angle, 180, 280, (0,0,0), 7)
        
        cv2.ellipse(frame, (self.x, self.y), axes, angle, 0, 100, self.color, 3)
        cv2.ellipse(frame, (self.x, self.y), axes, angle, 180, 280, self.color, 3)
        
        pts = []
        for i in range(6):
            theta = math.radians(60 * i + (angle * -0.6)) 
            px = int(self.x + (self.radius - 12) * math.cos(theta))
            py = int(self.y + (self.radius - 12) * math.sin(theta))
            pts.append([px, py])
        
        pts_array = np.array(pts, np.int32)
        cv2.polylines(frame, [pts_array], isClosed=True, color=(0, 0, 0), thickness=6)
        cv2.polylines(frame, [pts_array], isClosed=True, color=(255, 255, 255), thickness=2)
        
        cv2.circle(frame, (self.x, self.y), 6, (0, 0, 0), -1)
        cv2.circle(frame, (self.x, self.y), 4, (0, 0, 255), -1)

class KinesisEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        self.is_running = True
        
        self.history_left = deque(maxlen=15)
        self.history_right = deque(maxlen=15)
        
        self.targets = [Target(self.w, self.h)]
        self.particles = []
        
        self.last_impact_time = 0
        self.last_score = 0
        self.last_speed = 0
        
        generate_synthetic_hit_sound("hit.wav")
        pygame.mixer.init()
        self.sound_enabled = False
        if os.path.exists("hit.wav"):
            self.hit_sound = pygame.mixer.Sound("hit.wav")
            self.hit_sound.set_volume(0.8) 
            self.sound_enabled = True

        self.frame_to_render = None
        self.lock = threading.Lock()
        self.inference_thread = threading.Thread(target=self.inference_loop, daemon=True)

    def calculate_speed(self, history):
        if len(history) < 3: return 0.0
        p1, t1 = history[-3]
        p2, t2 = history[-1]
        
        dt = t2 - t1
        if dt <= 0: return 0.0
        
        dist_pixels = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        dist_meters = dist_pixels * (1.5 / self.h)
        return (dist_meters / dt) * 2.23694

    def check_collision(self, wrist_pos, speed):
        hit = False
        for t in self.targets:
            if not t.active: continue
            dist = math.sqrt((wrist_pos[0] - t.x)**2 + (wrist_pos[1] - t.y)**2)
            
            if dist < t.radius + 30:
                t.active = False
                hit = True
                
                if self.sound_enabled:
                    pygame.mixer.find_channel(True).play(self.hit_sound)
                
                for _ in range(45):
                    self.particles.append(Particle(t.x, t.y, t.color))
                
                self.last_score = speed * 8.5 * 10 
                self.last_speed = speed
                self.last_impact_time = time.time()
                
        if hit:
            self.targets = [t for t in self.targets if t.active]
            if len(self.targets) == 0:
                self.targets.append(Target(self.w, self.h))

    def inference_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1) 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.pose.process(rgb)
            current_time = time.time()
            
            with self.lock:
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    lx, ly = int(lm[15].x * self.w), int(lm[15].y * self.h)
                    rx, ry = int(lm[16].x * self.w), int(lm[16].y * self.h)
                    
                    self.history_left.append(((lx, ly), current_time))
                    self.history_right.append(((rx, ry), current_time))
                    
                    l_speed = self.calculate_speed(self.history_left)
                    r_speed = self.calculate_speed(self.history_right)
                    
                    self.check_collision((lx, ly), l_speed)
                    self.check_collision((rx, ry), r_speed)
                
                self.frame_to_render = frame.copy()

    def draw_shadow_trail(self, frame, history, color):
        if len(history) < 2: return
        
        pts = [h[0] for h in history]
        for i in range(1, len(pts)):
            thickness = int(np.interp(i, [1, len(pts)], [2, 18]))
            cv2.line(frame, pts[i-1], pts[i], (0, 0, 0), thickness + 6)
            cv2.line(frame, pts[i-1], pts[i], color, thickness)
            
        cv2.circle(frame, pts[-1], 20, (0, 0, 0), -1)
        cv2.circle(frame, pts[-1], 15, (255, 255, 255), -1)
        cv2.circle(frame, pts[-1], 18, color, 3)

    def draw_text_with_shadow(self, frame, text, pos, scale, color, thickness):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def run(self):
        self.inference_thread.start()
        print("[SYSTEM] Project Kinesis Auto-Audio Engine Online. Press 'q' to abort.")
        
        window_name = "Project Kinesis: AI Shadow-Box"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while self.is_running:
            start_t = time.time()
            
            with self.lock:
                if self.frame_to_render is not None:
                    display_frame = cv2.addWeighted(self.frame_to_render, 0.6, np.zeros_like(self.frame_to_render), 0.4, 0)
                else:
                    display_frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                    
                for t in self.targets:
                    t.draw(display_frame)
                    
                self.draw_shadow_trail(display_frame, self.history_left, (255, 200, 0))
                self.draw_shadow_trail(display_frame, self.history_right, (0, 165, 255))
                    
                for p in self.particles:
                    p.update()
                    p.draw(display_frame)
                self.particles = [p for p in self.particles if p.life > 0]
                
                if time.time() - self.last_impact_time < 1.5:
                    self.draw_text_with_shadow(display_frame, f"IMPACT: {int(self.last_score)}", 
                                              (30, 80), 1.8, (0, 0, 255), 5)
                    self.draw_text_with_shadow(display_frame, f"VELOCITY: {int(self.last_speed)} MPH", 
                                              (30, 140), 1.2, (0, 255, 255), 3)

            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                
            elapsed = time.time() - start_t
            time.sleep(max(0, (1.0/60.0) - elapsed))
            
        self.cap.release()
        pygame.quit()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = KinesisEngine()
    app.run()