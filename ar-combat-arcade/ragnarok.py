import cv2
import numpy as np
import mediapipe as mp
import multiprocessing
import queue
import time
import random
import math
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
from typing import List, Tuple, Dict, Optional

class GameState:
    STANDBY = 0
    PLAYING = 1
    GAME_OVER = 2

class Projectile:
    def __init__(self, x: float, y: float, vx: float, vy: float, radius: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt

class Laser:
    def __init__(self, start_y: float, target_y: float, speed: float, width: float):
        self.current_y = start_y
        self.target_y = target_y
        self.speed = speed
        self.width = width
        self.is_active = False
        self.charge_timer = 1.5

    def update(self, dt: float):
        if self.charge_timer > 0:
            self.charge_timer -= dt
            if self.charge_timer <= 0:
                self.is_active = True
        else:
            direction = 1 if self.target_y > self.current_y else -1
            self.current_y += direction * self.speed * dt
            if abs(self.current_y - self.target_y) < 5:
                return True
        return False

def pose_inference_worker(frame_queue: multiprocessing.Queue, coord_queue: multiprocessing.Queue, shutdown_event: multiprocessing.Event):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while not shutdown_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        data = {}
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            l_pinky = landmarks[mp_pose.PoseLandmark.LEFT_PINKY]
            l_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
            r_pinky = landmarks[mp_pose.PoseLandmark.RIGHT_PINKY]
            r_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

            if nose.visibility > 0.5:
                data['head'] = (int(nose.x * w), int(nose.y * h))
            if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
                data['shoulder_left'] = (int(l_sh.x * w), int(l_sh.y * h))
                data['shoulder_right'] = (int(r_sh.x * w), int(r_sh.y * h))
            if l_wrist.visibility > 0.5:
                data['left_wrist'] = (int(l_wrist.x * w), int(l_wrist.y * h))
            if r_wrist.visibility > 0.5:
                data['right_wrist'] = (int(r_wrist.x * w), int(r_wrist.y * h))
                
            left_palm_span = math.hypot(l_pinky.x - l_index.x, l_pinky.y - l_index.y)
            left_wrist_to_index = math.hypot(l_wrist.x - l_index.x, l_wrist.y - l_index.y)
            data['left_palm_open'] = left_palm_span > 0.04 and left_wrist_to_index > 0.05
            
            right_palm_span = math.hypot(r_pinky.x - r_index.x, r_pinky.y - r_index.y)
            right_wrist_to_index = math.hypot(r_wrist.x - r_index.x, r_wrist.y - r_index.y)
            data['right_palm_open'] = right_palm_span > 0.04 and right_wrist_to_index > 0.05

            if l_hip.visibility > 0.5 and r_hip.visibility > 0.5:
                data['torso_center'] = (
                    int(((l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4) * w),
                    int(((l_sh.y + r_sh.y + l_hip.y + r_hip.y) / 4) * h)
                )
                data['torso_box'] = {
                    'xmin': int(min(l_sh.x, r_sh.x, l_hip.x, r_hip.x) * w),
                    'xmax': int(max(l_sh.x, r_sh.x, l_hip.x, r_hip.x) * w),
                    'ymin': int(min(l_sh.y, r_sh.y) * h),
                    'ymax': int(max(l_hip.y, r_hip.y) * h)
                }
        
        while not coord_queue.empty():
            try:
                coord_queue.get_nowait()
            except queue.Empty:
                break
        coord_queue.put(data)

    pose.close()

class RagnarokEngine:
    def __init__(self):
        self.win_name = "SAYYAM AI LAB - PROJECT RAGNAROK"
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.width = 1280
        self.height = 720
        
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.coord_queue = multiprocessing.Queue(maxsize=1)
        self.shutdown_event = multiprocessing.Event()
        
        self.worker_process = multiprocessing.Process(
            target=pose_inference_worker, 
            args=(self.frame_queue, self.coord_queue, self.shutdown_event)
        )
        
        self.state = GameState.STANDBY
        self.projectiles: List[Projectile] = []
        self.active_laser: Optional[Laser] = None
        
        self.score = 0.0
        self.high_score = 0.0
        self.spawn_timer = 0.0
        self.laser_cooldown = 5.0
        
        self.current_coords: Dict = {}
        self.prev_time = time.time()
        self.near_miss_trigger = 0.0
        self.time_dilation = 1.0
        
        self.boss_pulse_radius = 50
        self.boss_pulse_dir = 1
        self.boss_pos = (640, 100)
        self.boss_health = 100.0
        
        self._init_audio_matrix()

    def _init_audio_matrix(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        sample_rate = 44100

        t = np.linspace(0, 0.1, int(sample_rate * 0.1), False)
        wave = np.sin(2 * np.pi * 880 * t) * 0.3
        sound_arr = np.vstack((wave, wave)).T.copy(order='C')
        self.snd_laser = pygame.mixer.Sound(buffer=(sound_arr * 32767).astype(np.int16))

        t = np.linspace(0, 0.1, int(sample_rate * 0.1), False)
        wave = np.sin(2 * np.pi * 1200 * t) * (1 - t / 0.1) * 0.4
        sound_arr = np.vstack((wave, wave)).T.copy(order='C')
        self.snd_hit = pygame.mixer.Sound(buffer=(sound_arr * 32767).astype(np.int16))

        t = np.linspace(0, 0.5, int(sample_rate * 0.5), False)
        noise = np.random.uniform(-1, 1, len(t))
        wave = noise * np.exp(-6 * t) * 0.6
        sound_arr = np.vstack((wave, wave)).T.copy(order='C')
        self.snd_explosion = pygame.mixer.Sound(buffer=(sound_arr * 32767).astype(np.int16))

    def _reset_sandbox(self):
        self.projectiles.clear()
        self.active_laser = None
        self.score = 0.0
        self.spawn_timer = 0.0
        self.laser_cooldown = 4.0
        self.time_dilation = 1.0
        self.near_miss_trigger = 0.0
        self.boss_health = 100.0

    def _check_collision_circle_rect(self, cx: float, cy: float, r: float, box: Dict) -> bool:
        closest_x = max(box['xmin'], min(cx, box['xmax']))
        closest_y = max(box['ymin'], min(cy, box['ymax']))
        dist_x = cx - closest_x
        dist_y = cy - closest_y
        return (dist_x ** 2 + dist_y ** 2) < (r ** 2)

    def _process_game_logic(self, dt: float):
        left_open = self.current_coords.get('left_palm_open', False)
        right_open = self.current_coords.get('right_palm_open', False)

        if self.state == GameState.STANDBY or self.state == GameState.GAME_OVER:
            if left_open or right_open:
                self._reset_sandbox()
                self.state = GameState.PLAYING
            return

        self.score += dt
        self.spawn_timer -= dt
        self.laser_cooldown -= dt
        
        if self.near_miss_trigger > 0:
            self.near_miss_trigger -= dt
            self.time_dilation = 0.25
        else:
            self.time_dilation = 1.0

        effective_dt = dt * self.time_dilation

        if 'left_wrist' in self.current_coords and not left_open:
            lw = self.current_coords['left_wrist']
            if math.hypot(lw[0] - self.boss_pos[0], lw[1] - self.boss_pos[1]) < self.boss_pulse_radius + 30:
                self.boss_health -= 15.0 * dt
                self.snd_hit.play()
        if 'right_wrist' in self.current_coords and not right_open:
            rw = self.current_coords['right_wrist']
            if math.hypot(rw[0] - self.boss_pos[0], rw[1] - self.boss_pos[1]) < self.boss_pulse_radius + 30:
                self.boss_health -= 15.0 * dt
                self.snd_hit.play()

        if self.boss_health <= 0:
            self.boss_health = 0
            self.state = GameState.GAME_OVER
            self.snd_explosion.play()
            return

        if self.spawn_timer <= 0:
            angle = random.uniform(0.3, math.pi - 0.3)
            speed = random.uniform(250, 450)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            self.projectiles.append(Projectile(self.boss_pos[0], self.boss_pos[1], vx, vy, random.uniform(15, 25)))
            self.spawn_timer = max(0.4, 1.2 - (self.score * 0.01))
            self.snd_laser.play()

        if self.laser_cooldown <= 0 and not self.active_laser:
            start_y = random.choice([200.0, 650.0])
            target_y = 650.0 if start_y == 200.0 else 200.0
            self.active_laser = Laser(start_y, target_y, random.uniform(200, 350), random.uniform(15, 30))
            self.laser_cooldown = random.uniform(6.0, 10.0)

        for p in self.projectiles[:]:
            p.update(effective_dt)
            if p.y > self.height + 50 or p.x < -50 or p.x > self.width + 50:
                self.projectiles.remove(p)

        if self.active_laser:
            completed = self.active_laser.update(effective_dt)
            if completed:
                self.active_laser = None

        if 'torso_box' in self.current_coords:
            box = self.current_coords['torso_box']
            head = self.current_coords.get('head', None)
            
            for p in self.projectiles:
                if self._check_collision_circle_rect(p.x, p.y, p.radius, box):
                    self.state = GameState.GAME_OVER
                    self.snd_explosion.play()
                    return
                if head:
                    dist = math.hypot(p.x - head[0], p.y - head[1])
                    if dist < p.radius + 20:
                        self.state = GameState.GAME_OVER
                        self.snd_explosion.play()
                        return
                    elif p.radius + 20 <= dist <= p.radius + 60:
                        self.near_miss_trigger = 0.4

            if self.active_laser and self.active_laser.is_active:
                ly = self.active_laser.current_y
                lw = self.active_laser.width
                
                if box['ymin'] <= ly <= box['ymax'] or abs(box['ymin'] - ly) < lw or abs(box['ymax'] - ly) < lw:
                    self.state = GameState.GAME_OVER
                    self.snd_explosion.play()
                    return
                if head:
                    if abs(head[1] - ly) < lw + 15:
                        self.state = GameState.GAME_OVER
                        self.snd_explosion.play()
                        return
                    elif lw + 15 <= abs(head[1] - ly) <= lw + 50:
                        self.near_miss_trigger = 0.4

    def _render_layer(self, frame: np.ndarray) -> np.ndarray:
        hud = frame.copy()
        
        self.boss_pulse_radius += self.boss_pulse_dir * 1
        if self.boss_pulse_radius > 65 or self.boss_pulse_radius < 45:
            self.boss_pulse_dir *= -1
            
        if self.boss_health > 0:
            cv2.circle(hud, self.boss_pos, self.boss_pulse_radius, (0, 0, 255), -1)
            cv2.circle(hud, self.boss_pos, self.boss_pulse_radius + 10, (0, 140, 255), 2)
            cv2.putText(hud, "NEXUS CORE", (self.boss_pos[0] - 60, self.boss_pos[1] - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            hw = 200
            hh = 15
            hx = self.boss_pos[0] - hw // 2
            hy = self.boss_pos[1] - 60
            cv2.rectangle(hud, (hx, hy), (hx + hw, hy + hh), (0, 0, 50), -1)
            current_w = int(hw * (self.boss_health / 100.0))
            cv2.rectangle(hud, (hx, hy), (hx + current_w, hy + hh), (0, 0, 255), -1)

        if 'shoulder_left' in self.current_coords and 'shoulder_right' in self.current_coords:
            sl = self.current_coords['shoulder_left']
            sr = self.current_coords['shoulder_right']
            cv2.line(hud, sl, sr, (0, 255, 255), 3)

        if 'left_wrist' in self.current_coords:
            color = (0, 255, 0) if self.current_coords.get('left_palm_open', False) else (0, 255, 255)
            cv2.circle(hud, self.current_coords['left_wrist'], 15, color, -1)
        if 'right_wrist' in self.current_coords:
            color = (0, 255, 0) if self.current_coords.get('right_palm_open', False) else (0, 255, 255)
            cv2.circle(hud, self.current_coords['right_wrist'], 15, color, -1)

        if 'torso_box' in self.current_coords:
            box = self.current_coords['torso_box']
            cv2.rectangle(hud, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 255, 0), 2)
            
        if 'head' in self.current_coords:
            cv2.circle(hud, self.current_coords['head'], 25, (0, 255, 0), 2)

        for p in self.projectiles:
            cv2.circle(hud, (int(p.x), int(p.y)), int(p.radius), (0, 0, 255), -1)

        if self.active_laser:
            ly = int(self.active_laser.current_y)
            lw = int(self.active_laser.width)
            if not self.active_laser.is_active:
                if int(time.time() * 15) % 2 == 0:
                    cv2.line(hud, (0, ly), (self.width, ly), (0, 165, 255), 2)
            else:
                cv2.rectangle(hud, (0, ly - lw // 2), (self.width, ly + lw // 2), (0, 0, 255), -1)

        if self.state == GameState.STANDBY:
            overlay = hud.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, hud, 0.3, 0, hud)
            cv2.putText(hud, "PROJECT RAGNAROK", (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3)
            cv2.putText(hud, "SHOW OPEN PALM TO CAMERA TO START", (390, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif self.state == GameState.PLAYING:
            cv2.putText(hud, f"SURVIVAL TIME: {self.score:.2f}s", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(hud, f"PUNCH THE CORE (CLOSE YOUR FISTS)", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        elif self.state == GameState.GAME_OVER:
            if self.score > self.high_score:
                self.high_score = self.score
            overlay = hud.copy()
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, hud, 0.6, 0, hud)
            
            if self.boss_health <= 0:
                cv2.putText(hud, "VICTORY: NEXUS CORE DESTROYED!", (320, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)
            else:
                cv2.putText(hud, "SYSTEM FAILURE: PLAYER ELIMINATED", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                
            cv2.putText(hud, f"SURVIVAL TIME: {self.score:.2f}s", (480, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(hud, "SHOW OPEN PALM TO REBOOT ENGINE OR PRESS 'Q' TO QUIT", (340, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.near_miss_trigger > 0 and self.state == GameState.PLAYING:
            edge_flash = cv2.copyMakeBorder(hud, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 255])
            hud = cv2.resize(edge_flash, (self.width, self.height))

        return hud

    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        self.worker_process.start()

        while self.cap.isOpened():
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if self.frame_queue.empty():
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

            try:
                self.current_coords = self.coord_queue.get_nowait()
            except queue.Empty:
                pass

            self._process_game_logic(dt)
            output_frame = self._render_layer(frame)

            cv2.imshow(self.win_name, output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.shutdown_event.set()
        self.worker_process.join()
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    engine = RagnarokEngine()
    engine.run()