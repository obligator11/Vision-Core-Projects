import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import pygame
from multiprocessing import Process, Queue

class SynthEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        self.sample_rate = 44100

    def play_shoot(self):
        duration = 0.1
        n_samples = int(round(duration * self.sample_rate))
        t = np.linspace(0, duration, n_samples, False)
        wave = 8192 * np.sin(2 * np.pi * np.linspace(1500, 400, n_samples) * t)
        envelope = np.linspace(1.0, 0.0, n_samples)
        stereo_wave = np.column_stack((wave * envelope, wave * envelope)).astype(np.int16)
        try: pygame.sndarray.make_sound(stereo_wave).play()
        except: pass

    def play_shatter(self, is_boss=False):
        duration = 0.4 if is_boss else 0.25
        n_samples = int(round(duration * self.sample_rate))
        noise = np.random.uniform(-10000, 10000, n_samples)
        envelope = np.exp(-15 * np.linspace(0, 1, n_samples))
        stereo_wave = np.column_stack((noise * envelope, noise * envelope)).astype(np.int16)
        try: pygame.sndarray.make_sound(stereo_wave).play()
        except: pass

    def play_emp(self):
        duration = 0.8
        n_samples = int(round(duration * self.sample_rate))
        t = np.linspace(0, duration, n_samples, False)
        wave = 16000 * np.sin(2 * np.pi * np.linspace(300, 20, n_samples) * t)
        envelope = np.linspace(1.0, 0.0, n_samples)
        stereo_wave = np.column_stack((wave * envelope, wave * envelope)).astype(np.int16)
        try: pygame.sndarray.make_sound(stereo_wave).play()
        except: pass

def vision_worker(data_queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    
    def is_pointing_forward(l, tip_idx, mcp_idx):
        return l[tip_idx].z < l[mcp_idx].z - 0.025
        
    def is_extended_2d(l, tip_idx, mcp_idx):
        tip_dist = math.hypot(l[tip_idx].x - l[0].x, l[tip_idx].y - l[0].y)
        mcp_dist = math.hypot(l[mcp_idx].x - l[0].x, l[mcp_idx].y - l[0].y)
        return tip_dist > mcp_dist * 1.2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue
            
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        hands_data = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for idx, hand_lm in enumerate(res.multi_hand_landmarks):
                l = hand_lm.landmark
                wx, wy = int(l[9].x * w), int(l[9].y * h) 
                wrist_x, wrist_y = int(l[0].x * w), int(l[0].y * h)
                
                hand_label = res.multi_handedness[idx].classification[0].label 
                color_id = 0 if hand_label == 'Right' else 1 
                
                idx_up = is_extended_2d(l, 8, 5) or is_pointing_forward(l, 8, 5)
                mid_up = is_extended_2d(l, 12, 9) or is_pointing_forward(l, 12, 9)
                ring_up = is_extended_2d(l, 16, 13) or is_pointing_forward(l, 16, 13)
                pinky_up = is_extended_2d(l, 20, 17) or is_pointing_forward(l, 20, 17)
                
                is_shooting = idx_up and mid_up and not ring_up and not pinky_up
                is_waving = idx_up and mid_up and ring_up and pinky_up
                
                dx = l[8].x - l[5].x
                dy = l[8].y - l[5].y
                
                hands_data.append({
                    'x': wx, 'y': wy,
                    'wrist_x': wrist_x, 'wrist_y': wrist_y,
                    'is_shooting': is_shooting,
                    'is_waving': is_waving,
                    'dx': dx, 'dy': dy,
                    'color_id': color_id
                })
        
        while not data_queue.empty():
            try: data_queue.get_nowait()
            except: pass
        data_queue.put((frame, hands_data))

class NeonShooter:
    def __init__(self):
        self.q = Queue(maxsize=1)
        self.worker = Process(target=vision_worker, args=(self.q,))
        self.worker.daemon = True
        self.worker.start()
        
        self.synth = SynthEngine()
        self.width, self.height = 1280, 720
        self.focal_length = 300 
        
        self.state = "START" 
        self.score = 0
        self.last_shot = {0: 0, 1: 0} 
        
        self.orbs = []
        self.bullets = []
        self.particles = []
        
        self.emp_charge = 0.0
        self.emp_active = False
        self.emp_radius = 0
        
        self.PINK = (255, 50, 255)
        self.CYAN = (255, 255, 50) 
        self.RED = (50, 50, 255)
        
    def project_3d(self, x, y, z, is_environment=False):
        if z <= -self.focal_length: z = -self.focal_length + 1
        scale = self.focal_length / (self.focal_length + z)
        
        if is_environment:
            cx, cy = self.width // 2, self.height // 2
            x_proj = int(cx + (x - cx) * scale)
            y_proj = int(cy + (y - cy) * scale)
        else:
            x_proj, y_proj = int(x), int(y)
            
        return x_proj, y_proj, scale

    def spawn_orb(self):
        color_type = random.choice([0, 1])
        x = random.randint(200, 1080)
        y = random.randint(100, 500)
        z = random.randint(200, 700) 
        phase = random.uniform(0, 10) 
        
        is_boss = random.random() < 0.15
        hp = 3 if is_boss else 1
        
        self.orbs.append({
            'x': x, 'y': y, 'z': z, 
            'type': color_type, 
            'phase': phase,
            'is_boss': is_boss,
            'hp': hp,
            'flash': 0
        })

    def fire_bullet(self, hand_x, hand_y, color_type, vx, vy):
        self.bullets.append({
            'x': hand_x, 'y': hand_y, 'z': 0, 
            'vx': vx, 'vy': vy, 'vz': 50, 
            'type': color_type
        })
        self.synth.play_shoot()

    def trigger_shatter(self, x, y, color, is_boss=False):
        count = 50 if is_boss else 25
        for _ in range(count):
            self.particles.append({
                'x': x, 'y': y,
                'vx': random.uniform(-25, 25) if is_boss else random.uniform(-15, 15),
                'vy': random.uniform(-25, 25) if is_boss else random.uniform(-15, 15),
                'life': 20 if is_boss else 12,
                'color': (255,255,255) if is_boss else color
            })
        self.synth.play_shatter(is_boss)

    def trigger_emp(self):
        self.emp_active = True
        self.emp_radius = 0
        self.emp_charge = 0.0
        self.synth.play_emp()
        
        for orb in self.orbs:
            self.score += 500 if orb['is_boss'] else 100
            self.trigger_shatter(orb['x'], orb['y'], self.CYAN, orb['is_boss'])
            
        self.orbs.clear()
        self.bullets.clear()

    def run(self):
        cv2.namedWindow('Neon Shooter', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Neon Shooter', self.width, self.height)
        
        last_spawn = time.time()
        
        while True:
            if not self.q.empty():
                raw_frame, hands = self.q.get()
            else: continue
                
            frame = cv2.resize(raw_frame, (self.width, self.height))
            frame = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
            
            cx, cy = self.width // 2, self.height // 2
            scroll = int((time.time() * 200) % 50)
            for z in range(0, 100, 10):
                y_line = self.project_3d(cx, self.height + 50, z - scroll/10, is_environment=True)[1]
                cv2.line(frame, (0, y_line), (self.width, y_line), (100, 0, 100), 1)

            if self.emp_active:
                self.emp_radius += 60
                cv2.circle(frame, (cx, cy), self.emp_radius, (255, 255, 255), -1)
                if self.emp_radius > self.width:
                    self.emp_active = False

            if self.state == "START":
                cv2.putText(frame, "WAVE HAND TO START", (250, 360), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 4)
                for h in hands:
                    if h['is_waving']:
                        self.state = "PLAYING"
                        self.score = 0
                        self.emp_charge = 0
                        self.orbs.clear()
                        self.bullets.clear()

            elif self.state == "PLAYING" and not self.emp_active:
                if self.emp_charge < 100:
                    self.emp_charge += 0.2 
                    
                cv2.putText(frame, f"SCORE: {self.score}", (50, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                
                bar_color = (255, 255, 255) if self.emp_charge >= 100 else (100, 100, 100)
                cv2.rectangle(frame, (50, 90), (50 + int(self.emp_charge * 3), 110), bar_color, -1)
                cv2.rectangle(frame, (50, 90), (350, 110), (255, 255, 255), 2)
                if self.emp_charge >= 100:
                    cv2.putText(frame, "EMP READY: CROSS ARMS", (50, 140), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                
                if len(self.orbs) < 6 and (time.time() - last_spawn > 0.4):
                    self.spawn_orb()
                    last_spawn = time.time()
                
                if len(hands) == 2 and self.emp_charge >= 100:
                    wrist_dist = math.hypot(hands[0]['wrist_x'] - hands[1]['wrist_x'], hands[0]['wrist_y'] - hands[1]['wrist_y'])
                    if wrist_dist < 120:
                        self.trigger_emp()
                        
                for h in hands:
                    color_type = h['color_id']
                    color = self.CYAN if color_type == 0 else self.PINK
                    
                    mag = math.hypot(h['dx'], h['dy'])
                    if mag < 0.001: mag = 0.001
                    dir_x = h['dx'] / mag
                    dir_y = h['dy'] / mag
                    
                    locked_orb = None
                    min_dist = 120 
                    
                    for orb in self.orbs:
                        if orb['type'] == color_type or orb['is_boss']:
                            ox = orb['x'] - h['x']
                            oy = orb['y'] - h['y']
                            t = ox * dir_x + oy * dir_y
                            if t > 0: 
                                px = h['x'] + t * dir_x
                                py = h['y'] + t * dir_y
                                perp_dist = math.hypot(orb['x'] - px, orb['y'] - py)
                                
                                if perp_dist < min_dist:
                                    min_dist = perp_dist
                                    locked_orb = orb
                    
                    if locked_orb:
                        target_x, target_y = int(locked_orb['x']), int(locked_orb['y'])
                        cv2.line(frame, (h['x'], h['y']), (target_x, target_y), color, 2)
                        cv2.drawMarker(frame, (target_x, target_y), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
                    else:
                        end_x = int(h['x'] + dir_x * 1500)
                        end_y = int(h['y'] + dir_y * 1500)
                        cv2.line(frame, (h['x'], h['y']), (end_x, end_y), color, 1)
                    
                    if h['is_shooting'] and (time.time() - self.last_shot[color_type] > 0.15):
                        if locked_orb:
                            frames_to_impact = locked_orb['z'] / 50.0
                            if frames_to_impact <= 0: frames_to_impact = 1
                            vx = (locked_orb['x'] - h['x']) / frames_to_impact
                            vy = (locked_orb['y'] - h['y']) / frames_to_impact
                        else:
                            vx, vy = dir_x * 30, dir_y * 30
                            
                        self.fire_bullet(h['x'], h['y'], color_type, vx, vy)
                        self.last_shot[color_type] = time.time()

                for b in self.bullets[:]:
                    b['x'] += b['vx']
                    b['y'] += b['vy']
                    b['z'] += b['vz'] 
                    
                    x1, y1, scale1 = self.project_3d(b['x'], b['y'], b['z'])
                    x2 = int(x1 - (b['vx'] * 1.5))
                    y2 = int(y1 - (b['vy'] * 1.5))
                    
                    color = self.CYAN if b['type'] == 0 else self.PINK
                    thickness = max(2, int(5 * scale1)) 
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness + 2) 
                    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)               
                    
                    if b['z'] > 2000 or b['x'] < -1000 or b['x'] > 2500 or b['y'] < -1000 or b['y'] > 2000: 
                        if b in self.bullets: self.bullets.remove(b)

                for orb in self.orbs[:]:
                    wobble_speed = 4 if orb['is_boss'] else 2
                    orb['x'] += math.sin(time.time() * wobble_speed + orb['phase']) * (3 if orb['is_boss'] else 1.5)
                    orb['y'] += math.cos(time.time() * (wobble_speed*0.75) + orb['phase']) * (2 if orb['is_boss'] else 1.0)
                    
                    if orb['flash'] > 0: orb['flash'] -= 1
                    
                    x, y, scale = self.project_3d(orb['x'], orb['y'], orb['z'])
                    
                    base_r = 80 if orb['is_boss'] else 45
                    if orb['is_boss']: base_r -= (3 - orb['hp']) * 15 
                    
                    r = int(base_r * scale)
                    
                    if orb['flash'] > 0:
                        render_color = (255, 255, 255)
                    elif orb['is_boss']:
                        render_color = self.RED
                    else:
                        render_color = self.CYAN if orb['type'] == 0 else self.PINK
                    
                    pulse = math.sin(time.time() * 8 + orb['z']) * (5 * scale)
                    cv2.circle(frame, (int(x), int(y)), int(r + pulse), render_color, 2)
                    
                    angle = math.degrees(time.time() * 3 + orb['z'])
                    cv2.ellipse(frame, (int(x), int(y)), (r, int(r*0.3)), angle, 0, 360, render_color, 2)
                    cv2.ellipse(frame, (int(x), int(y)), (int(r*0.3), r), angle, 0, 360, render_color, 2)
                    cv2.circle(frame, (int(x), int(y)), int(r * 0.3), (255, 255, 255), -1) 
                        
                    for b in self.bullets[:]:
                        if orb['is_boss'] or b['type'] == orb['type']:
                            dist = math.hypot(orb['x'] - b['x'], orb['y'] - b['y'])
                            z_crossed = (b['z'] >= orb['z']) and ((b['z'] - b['vz']) <= orb['z'] + 15)
                            
                            hit_threshold = 120 if orb['is_boss'] else 80
                            
                            if dist < hit_threshold and z_crossed: 
                                if b in self.bullets: self.bullets.remove(b) 
                                
                                orb['hp'] -= 1
                                orb['flash'] = 3 
                                
                                if orb['hp'] <= 0:
                                    self.trigger_shatter(int(x), int(y), render_color, orb['is_boss'])
                                    self.score += 500 if orb['is_boss'] else 100
                                    if orb in self.orbs: self.orbs.remove(orb)
                                else:
                                    self.synth.play_shatter(is_boss=False)
                                break

            if self.particles:
                particle_layer = np.zeros_like(frame)
                for p in self.particles[:]:
                    p['x'] += p['vx']
                    p['y'] += p['vy']
                    p['life'] -= 1
                    if p['life'] <= 0:
                        self.particles.remove(p)
                    else:
                        cv2.circle(particle_layer, (int(p['x']), int(p['y'])), p['life'], p['color'], -1)
                frame = cv2.add(frame, cv2.GaussianBlur(particle_layer, (15, 15), 0))

            cv2.imshow('Neon Shooter', frame)
            if cv2.waitKey(1) & 0xFF == 27: break
                
        self.worker.terminate()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == '__main__':
    engine = NeonShooter()
    engine.run()