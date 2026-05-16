import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import math
import random
from collections import deque
import pygame

# ==============================================================================
# SAYYAM AI LAB: PROJECT 'SHINOBI-SLICE' ENGINE V6 (BALANCED ARCADE)
# Upgrades: Tuned Physics, Fair Spawning, Cinematic Gravity
# ==============================================================================

class ProceduralAudio:
    """Synthesizes zero-latency game audio using pure mathematics."""
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        sample_rate = 44100
        
        # 1. Synthesize 'Fruit Slice'
        t_slice = np.linspace(0, 0.15, int(sample_rate * 0.15), False)
        slice_wave = np.sin(2 * np.pi * 1200 * t_slice) * np.exp(-t_slice * 25)
        slice_16bit = (slice_wave * 32767).astype(np.int16)
        self.snd_slice = pygame.sndarray.make_sound(np.column_stack((slice_16bit, slice_16bit)))
        
        # 2. Synthesize 'Bomb Explosion' 
        t_bomb = np.linspace(0, 0.5, int(sample_rate * 0.5), False)
        noise = np.random.uniform(-1, 1, len(t_bomb)) * np.exp(-t_bomb * 15)
        boom = np.sin(2 * np.pi * 60 * t_bomb) * np.exp(-t_bomb * 5)
        bomb_wave = (noise * 0.5) + boom
        bomb_16bit = (bomb_wave * 32767).astype(np.int16)
        self.snd_bomb = pygame.sndarray.make_sound(np.column_stack((bomb_16bit, bomb_16bit)))

class HalfObject:
    def __init__(self, x, y, radius, color, split_dir):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.angle = random.uniform(0, 360)
        self.rot_speed = split_dir * 10.0
        self.vx = split_dir * random.uniform(4.0, 12.0) 
        self.vy = random.uniform(-8.0, -2.0)

    def update(self, gravity):
        self.vy += gravity
        self.x += self.vx
        self.y += self.vy
        self.angle += self.rot_speed

class GameObject:
    def __init__(self, w, h):
        # BALANCED: 92% Fruit, 8% Bomb
        self.type = random.choices(['FRUIT', 'BOMB'], weights=[92, 8], k=1)[0]
        
        # BALANCED: Fair sizes so you don't miss due to camera resolution
        self.radius = random.randint(30, 50)
        
        self.x = random.randint(150, w - 150)
        self.y = h + self.radius
        
        # BALANCED: Controlled Arcs so they "float" slightly at the top
        self.vx = random.uniform(-6.0, 6.0)  
        self.vy = random.uniform(-22.0, -32.0) 
        self.angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(-10.0, 10.0)
        
        if self.type == 'FRUIT':
            self.color = random.choice([(0, 255, 150), (255, 215, 0), (0, 255, 255), (255, 50, 255)])
        else:
            self.color = (0, 0, 255)

    def update(self, gravity):
        self.vy += gravity
        self.x += self.vx
        self.y += self.vy
        self.angle += self.rot_speed
        
class ShinobiSliceEngine:
    def __init__(self):
        self.audio = ProceduralAudio()
        self.frame_queue = queue.Queue(maxsize=2)
        self.state_queue = queue.Queue(maxsize=2)
        self.running = True
        
        self.state = "STANDBY"
        self.score = 0
        self.lives = 3
        self.combo = 0          
        self.gravity = 0.75 # Floatier gravity for better slicing windows
        
        self.objects = []
        self.split_objects = [] 
        self.blade_trail = deque(maxlen=20) 
        self.particles = [] 
        self.trail_sparks = []  
        self.shake_frames = 0
        
        self.ai_thread = threading.Thread(target=self.vision_worker, daemon=True)
        self.ai_thread.start()

    def vision_worker(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                state_data = {'is_dual_palm': False, 'index_tip': None}
                
                if results.multi_hand_landmarks:
                    open_palms = 0
                    for hand_landmarks in results.multi_hand_landmarks:
                        palm = hand_landmarks.landmark[0]
                        tips = [8, 12, 16, 20]
                        extended = sum(1 for t in tips if math.hypot((hand_landmarks.landmark[t].x - palm.x)*w, (hand_landmarks.landmark[t].y - palm.y)*h) > h * 0.2)
                        if extended >= 3: open_palms += 1
                            
                        lm_8 = hand_landmarks.landmark[8]
                        state_data['index_tip'] = (int(lm_8.x * w), int(lm_8.y * h))
                    if open_palms == 2: state_data['is_dual_palm'] = True
                        
                if not self.state_queue.full():
                    self.state_queue.put(state_data)

    def check_intersection(self, p1, p2, circle_center, radius):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = circle_center
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: return math.hypot(cx - x1, cy - y1) <= radius
        t = max(0, min(1, ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)))
        return math.hypot(cx - (x1 + t * dx), cy - (y1 + t * dy)) <= radius

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Project Shinobi-Slice V6 (Arcade Edition)", cv2.WINDOW_NORMAL)
        last_spawn_time = time.time()
        spatial_data = {'is_dual_palm': False, 'index_tip': None}
        
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            if not self.state_queue.empty():
                spatial_data = self.state_queue.get()

            overlay = frame.copy()
            
            if self.state in ["STANDBY", "GAMEOVER"]:
                cv2.putText(overlay, f"SYSTEM {self.state}", (w//2 - 200, h//2 - 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 215, 255), 4, cv2.LINE_AA)
                cv2.putText(overlay, "Raise Dual 'Open Palms' To Ignite", (w//2 - 300, h//2 + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                if spatial_data['is_dual_palm']:
                    self.state = "PLAYING"
                    self.score = 0; self.lives = 3; self.combo = 0
                    self.objects.clear(); self.split_objects.clear()
                    self.blade_trail.clear(); self.particles.clear(); self.trail_sparks.clear()
                    
            elif self.state == "PLAYING":
                # ==========================================================
                # 1. KINEMATICS & PLASMA RIBBON VFX
                # ==========================================================
                if spatial_data['index_tip']:
                    tip = spatial_data['index_tip']
                    self.blade_trail.append(tip)
                    for _ in range(2):
                        self.trail_sparks.append({'pos': [tip[0], tip[1]], 
                                                  'vx': random.uniform(-3, 3), 
                                                  'vy': random.uniform(-3, 3), 
                                                  'life': random.randint(10, 25)})
                elif len(self.blade_trail) >= 2:
                    p1, p2 = self.blade_trail[-2], self.blade_trail[-1]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    if math.hypot(dx, dy) > 20: 
                        self.blade_trail.append((p2[0] + dx, p2[1] + dy))

                trail_len = len(self.blade_trail)
                if trail_len > 2:
                    for i in range(1, trail_len):
                        ratio = i / trail_len
                        aura = max(1, int(40 * ratio))
                        core = max(1, int(10 * ratio))
                        pt1, pt2 = self.blade_trail[i-1], self.blade_trail[i]
                        cv2.line(overlay, pt1, pt2, (255, 0, 255), aura, cv2.LINE_AA)
                        cv2.line(overlay, pt1, pt2, (255, 255, 255), core, cv2.LINE_AA)

                for spark in self.trail_sparks[:]:
                    spark['pos'][0] += spark['vx']
                    spark['pos'][1] += spark['vy']
                    spark['life'] -= 1
                    cv2.circle(overlay, (int(spark['pos'][0]), int(spark['pos'][1])), 
                               max(1, int(spark['life']/5)), (255, 200, 255), -1, cv2.LINE_AA)
                    if spark['life'] <= 0: self.trail_sparks.remove(spark)
                
                # ==========================================================
                # 2. BALANCED SPAWNER
                # ==========================================================
                # Spawns every ~1 second. Spawns 2 to 5 items max based on score.
                if time.time() - last_spawn_time > max(0.6, 1.8 - (self.score * 0.002)):
                    burst_amount = random.randint(2, min(5, 2 + self.score // 150))
                    for _ in range(burst_amount):
                        self.objects.append(GameObject(w, h))
                    last_spawn_time = time.time()

                for half in self.split_objects[:]:
                    half.update(self.gravity)
                    cv2.ellipse(overlay, (int(half.x), int(half.y)), (half.radius, half.radius), 
                                half.angle, 0, 180, half.color, -1, cv2.LINE_AA)
                    if half.y > h + half.radius:
                        self.split_objects.remove(half)

                # ==========================================================
                # 3. COLLISION & THE COMBO ENGINE
                # ==========================================================
                for obj in self.objects[:]:
                    obj.update(self.gravity)
                    
                    if obj.type == 'FRUIT':
                        pts = np.array([[int(obj.x + obj.radius * math.cos(math.radians(obj.angle + j * 60))),
                                         int(obj.y + obj.radius * math.sin(math.radians(obj.angle + j * 60)))] 
                                         for j in range(6)], np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], obj.color, cv2.LINE_AA)
                        cv2.polylines(overlay, [pts], True, (255, 255, 255), 3, cv2.LINE_AA)
                    elif obj.type == 'BOMB':
                        pulse = int(math.sin(time.time() * 20) * 8)
                        cv2.circle(overlay, (int(obj.x), int(obj.y)), obj.radius + pulse, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(overlay, (int(obj.x), int(obj.y)), int(obj.radius * 0.4), (255, 255, 255), -1, cv2.LINE_AA)

                    sliced = False
                    if trail_len >= 2:
                        if self.check_intersection(self.blade_trail[-2], self.blade_trail[-1], (obj.x, obj.y), obj.radius):
                            sliced = True

                    if sliced:
                        if obj.type == 'FRUIT':
                            self.audio.snd_slice.play()
                            self.combo += 1
                            points_earned = 10 + (self.combo * 5)
                            self.score += points_earned
                            
                            self.particles.append({'pos': (int(obj.x), int(obj.y)), 'life': 20, 'color': obj.color})
                            self.split_objects.append(HalfObject(obj.x, obj.y, obj.radius, obj.color, -1))
                            self.split_objects.append(HalfObject(obj.x, obj.y, obj.radius, obj.color, 1))
                            
                        elif obj.type == 'BOMB':
                            self.audio.snd_bomb.play()
                            self.combo = 0 
                            self.lives -= 1
                            self.shake_frames = 20
                            self.objects.clear(); self.split_objects.clear()
                            if self.lives <= 0: self.state = "GAMEOVER"
                            
                        if obj in self.objects: self.objects.remove(obj)
                    
                    elif obj.y > h + obj.radius:
                        if obj.type == 'FRUIT':
                            self.combo = 0 
                        self.objects.remove(obj)

                for p in self.particles[:]:
                    cv2.circle(overlay, p['pos'], 100 - p['life']*4, p['color'], max(1, p['life']), cv2.LINE_AA)
                    p['life'] -= 1
                    if p['life'] <= 0: self.particles.remove(p)

                # UI
                cv2.putText(overlay, f"SCORE: {self.score}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, f"LIVES: {'X ' * self.lives}", (30, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                if self.combo >= 3:
                    combo_color = (0, 255, 255) if self.combo < 10 else (0, 100, 255)
                    pulse = int(math.sin(time.time() * 10) * 5)
                    cv2.putText(overlay, f"{self.combo}x COMBO!", (w - 350, 80 + pulse), cv2.FONT_HERSHEY_DUPLEX, 1.5, combo_color, 4, cv2.LINE_AA)

            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            
            if self.shake_frames > 0:
                M = np.float32([[1, 0, random.randint(-40, 40)], [0, 1, random.randint(-40, 40)]])
                frame = cv2.warpAffine(frame, M, (w, h))
                red_layer = np.zeros_like(frame); red_layer[:, :] = (0, 0, 255)
                frame = cv2.addWeighted(frame, 0.6, red_layer, 0.4, 0)
                self.shake_frames -= 1

            cv2.imshow("Project Shinobi-Slice V6 (Arcade Edition)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ShinobiSliceEngine()
    app.run()