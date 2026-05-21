import cv2
import mediapipe as mp
import numpy as np
import multiprocessing as mp_lib
import math
import random
import collections
import pygame

def vision_worker(frame_q, coord_q):
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    while True:
        frame = frame_q.get()
        if frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            y_val = lm[8].y
            
            dy = math.hypot(lm[0].x - lm[12].x, lm[0].y - lm[12].y)
            dx = math.hypot(lm[4].x - lm[20].x, lm[4].y - lm[20].y)
            is_palm = (dy > 0.2) and (dx > 0.2)
            
            coord_q.put((y_val, is_palm))
        else:
            coord_q.put((None, False))

class AvianDashEngineV4:
    def __init__(self):
        self.w = 1280
        self.h = 720
        self.avatar_x = 300
        self.avatar_r = 15
        self.pipe_w = 100
        self.gap_h = 180
        self.state = "STANDBY"
        
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.snd_start = self.create_tone(800, 0.3)
        self.snd_score = self.create_tone(1200, 0.1)
        self.snd_crash = self.create_noise(0.4)
        
        self.frame_q = mp_lib.Queue(maxsize=1)
        self.coord_q = mp_lib.Queue()
        self.worker = mp_lib.Process(target=vision_worker, args=(self.frame_q, self.coord_q))
        self.worker.daemon = True
        self.worker.start()
        
        self.reset_game()

    def create_tone(self, freq, duration, vol=0.3):
        t = np.linspace(0, duration, int(44100 * duration), False)
        wave = np.sin(freq * t * 2 * np.pi)
        audio = np.int16(wave * vol * 32767)
        return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

    def create_noise(self, duration, vol=0.3):
        noise = np.random.uniform(-1, 1, int(44100 * duration))
        audio = np.int16(noise * vol * 32767)
        return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

    def reset_game(self):
        self.avatar_y = self.h // 2
        self.pipes = []
        self.enemies = []
        self.particles = []
        self.trail = collections.deque(maxlen=15)
        self.score = 0
        self.speed = 10.0
        self.spawn_pipe()

    def spawn_pipe(self):
        max_amp = 140
        min_y = max_amp + 20
        max_y = self.h - self.gap_h - max_amp - 20
        base_y = random.randint(min_y, max_y)
        
        self.pipes.append({
            "x": self.w,
            "base_y": base_y,
            "y": base_y,
            "passed": False,
            "phase": random.uniform(0, 6.28),
            "amp": random.uniform(60, max_amp),
            "freq": random.uniform(0.04, 0.09)
        })

    def spawn_enemy(self):
        self.enemies.append({
            "x": self.w + 100,
            "base_y": random.randint(100, self.h - 100),
            "phase": random.uniform(0, 6.28),
            "speed": random.uniform(16.0, 24.0),
            "freq": random.uniform(0.1, 0.3),
            "amp": random.uniform(80, 200)
        })

    def get_euclidean_collision(self, px, py, rx, ry, rw, rh):
        cx = max(rx, min(px, rx + rw))
        cy = max(ry, min(py, ry + rh))
        return math.hypot(px - cx, py - cy) < self.avatar_r + 8

    def spawn_explosion(self):
        for _ in range(150):
            self.particles.append({
                "x": self.avatar_x,
                "y": self.avatar_y,
                "vx": random.uniform(-40, 40),
                "vy": random.uniform(-40, 40),
                "life": 255,
                "color": random.choice([(0, 255, 255), (0, 100, 255), (0, 0, 255), (255, 255, 255)])
            })

    def draw_cyber_bird(self, overlay):
        body = np.array([
            [self.avatar_x + 25, self.avatar_y],
            [self.avatar_x - 15, self.avatar_y - 10],
            [self.avatar_x - 5, self.avatar_y],
            [self.avatar_x - 15, self.avatar_y + 10]
        ], np.int32)
        cv2.fillPoly(overlay, [body], (255, 120, 0))
        cv2.polylines(overlay, [body], True, (255, 255, 255), 2)
        
        wing1 = np.array([
            [self.avatar_x + 5, self.avatar_y - 5],
            [self.avatar_x - 15, self.avatar_y - 30],
            [self.avatar_x - 5, self.avatar_y - 5]
        ], np.int32)
        cv2.fillPoly(overlay, [wing1], (0, 255, 255))
        cv2.polylines(overlay, [wing1], True, (255, 255, 255), 1)

        wing2 = np.array([
            [self.avatar_x + 5, self.avatar_y + 5],
            [self.avatar_x - 15, self.avatar_y + 30],
            [self.avatar_x - 5, self.avatar_y + 5]
        ], np.int32)
        cv2.fillPoly(overlay, [wing2], (0, 255, 255))
        cv2.polylines(overlay, [wing2], True, (255, 255, 255), 1)

        cv2.circle(overlay, (int(self.avatar_x + 12), int(self.avatar_y - 2)), 3, (0, 0, 255), -1)

    def draw_enemy(self, overlay, ex, ey):
        pts = np.array([
            [ex, ey - 12],
            [ex + 35, ey],
            [ex, ey + 12],
            [ex - 20, ey]
        ], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 0, 200))
        cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)
        cv2.circle(overlay, (int(ex + 10), int(ey)), 4, (0, 255, 255), -1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Avian-Dash: NIGHTMARE PROTOCOL", cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.w, self.h))
            
            if self.frame_q.empty():
                self.frame_q.put(cv2.resize(frame, (320, 240)))
                
            y_val, is_palm = None, False
            while not self.coord_q.empty():
                y_val, is_palm = self.coord_q.get()
                
            display = cv2.addWeighted(frame, 0.15, np.zeros_like(frame), 0.85, 0)
            overlay = np.zeros_like(display)

            if self.state == "STANDBY":
                cv2.putText(overlay, "NIGHTMARE PROTOCOL", (self.w // 2 - 320, self.h // 2 - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                cv2.putText(overlay, "RAISE OPEN PALM TO INITIATE", (self.w // 2 - 320, self.h // 2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                if is_palm:
                    self.reset_game()
                    self.snd_start.play()
                    self.state = "PLAYING"

            elif self.state == "PLAYING":
                if y_val is not None:
                    target_y = int(y_val * self.h)
                    self.avatar_y += (target_y - self.avatar_y) * 0.45
                    
                self.trail.append((self.avatar_x, self.avatar_y))
                for i in range(1, len(self.trail)):
                    thickness = max(1, int((i / len(self.trail)) * 6))
                    cv2.line(overlay, (int(self.trail[i-1][0] - 15), int(self.trail[i-1][1])),
                             (int(self.trail[i][0] - 15), int(self.trail[i][1])), (0, 255, 255), thickness)
                
                for p in self.pipes:
                    p["x"] -= int(self.speed)
                    p["phase"] += p["freq"]
                    p["y"] = int(p["base_y"] + math.sin(p["phase"]) * p["amp"])
                    
                    cv2.rectangle(overlay, (p["x"], 0), (p["x"] + self.pipe_w, p["y"]), (40, 40, 40), -1)
                    cv2.rectangle(overlay, (p["x"], 0), (p["x"] + self.pipe_w, p["y"]), (0, 0, 255), 3)
                    
                    cv2.rectangle(overlay, (p["x"], p["y"] + self.gap_h), (p["x"] + self.pipe_w, self.h), (40, 40, 40), -1)
                    cv2.rectangle(overlay, (p["x"], p["y"] + self.gap_h), (p["x"] + self.pipe_w, self.h), (0, 0, 255), 3)
                    
                    for step_y in range(0, p["y"], 30):
                        cv2.line(overlay, (p["x"], step_y), (p["x"] + self.pipe_w, step_y), (100, 0, 100), 1)
                    for step_y in range(p["y"] + self.gap_h, self.h, 30):
                        cv2.line(overlay, (p["x"], step_y), (p["x"] + self.pipe_w, step_y), (100, 0, 100), 1)
                    
                    if self.get_euclidean_collision(self.avatar_x, self.avatar_y, p["x"], 0, self.pipe_w, p["y"]) or \
                       self.get_euclidean_collision(self.avatar_x, self.avatar_y, p["x"], p["y"] + self.gap_h, self.pipe_w, self.h - p["y"] - self.gap_h):
                        self.snd_crash.play()
                        self.state = "GAME_OVER"
                        self.spawn_explosion()
                        
                    if p["x"] + self.pipe_w < self.avatar_x and not p["passed"]:
                        p["passed"] = True
                        self.snd_score.play()
                        self.score += 1
                        self.speed += 0.6
                        self.spawn_enemy()
                        
                self.pipes = [p for p in self.pipes if p["x"] + self.pipe_w > 0]
                if self.pipes[-1]["x"] < self.w - 600:
                    self.spawn_pipe()

                for e in self.enemies:
                    e["x"] -= int(e["speed"])
                    e["phase"] += e["freq"]
                    ey = int(e["base_y"] + math.sin(e["phase"]) * e["amp"])
                    self.draw_enemy(overlay, e["x"], ey)
                    
                    if math.hypot(self.avatar_x - e["x"], self.avatar_y - ey) < self.avatar_r + 18:
                        self.snd_crash.play()
                        self.state = "GAME_OVER"
                        self.spawn_explosion()
                        
                self.enemies = [e for e in self.enemies if e["x"] > -100]
                self.draw_cyber_bird(overlay)
                
                hud_overlay = display.copy()
                cv2.rectangle(hud_overlay, (self.w // 2 - 120, 20), (self.w // 2 + 120, 80), (10, 10, 10), -1)
                cv2.rectangle(hud_overlay, (self.w // 2 - 120, 20), (self.w // 2 + 120, 80), (0, 0, 255), 2)
                cv2.putText(hud_overlay, f"SYS.SCORE: {self.score}", (self.w // 2 - 100, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                display = cv2.addWeighted(hud_overlay, 0.9, display, 0.1, 0)
                
            elif self.state == "GAME_OVER":
                for pt in self.particles:
                    pt["x"] += pt["vx"]
                    pt["y"] += pt["vy"]
                    pt["vy"] += 1.8
                    pt["life"] -= 8
                    if pt["life"] > 0:
                        cv2.circle(overlay, (int(pt["x"]), int(pt["y"])), random.randint(2, 6), pt["color"], -1)
                self.particles = [pt for pt in self.particles if pt["life"] > 0]
                
                cv2.putText(overlay, "CRITICAL FAILURE", (self.w // 2 - 280, self.h // 2 - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                cv2.putText(overlay, "RAISE OPEN PALM TO REBOOT", (self.w // 2 - 300, self.h // 2 + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

                if is_palm:
                    self.reset_game()
                    self.snd_start.play()
                    self.state = "PLAYING"
                
            cv2.addWeighted(overlay, 1.0, display, 1.0, 0, display)
            cv2.imshow("Avian-Dash: NIGHTMARE PROTOCOL", display)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        self.frame_q.put(None)
        self.worker.join()
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == '__main__':
    app = AvianDashEngineV4()
    app.run()