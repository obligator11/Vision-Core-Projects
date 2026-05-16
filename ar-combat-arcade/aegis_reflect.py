import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import random
import pygame

class AudioEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.deflect_snd = self._generate_pew(44100, 0.1, 800, 200)
        self.emp_snd = self._generate_boom(44100, 1.0)
        self.hit_snd = self._generate_kill(44100, 0.4)
        
    def _generate_pew(self, sample_rate, duration, start_freq, end_freq):
        n_samples = int(sample_rate * duration)
        buf = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            t = float(i) / sample_rate
            freq = start_freq * ((end_freq / start_freq) ** (t / duration))
            buf[i] = np.sin(2.0 * math.pi * freq * t) * (1.0 - t/duration)
        noise = np.random.uniform(-0.2, 0.2, n_samples).astype(np.float32)
        buf = buf + noise
        buf = np.int16(buf * 32767)
        return pygame.sndarray.make_sound(np.column_stack((buf, buf)))

    def _generate_boom(self, sample_rate, duration):
        n_samples = int(sample_rate * duration)
        buf = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            t = float(i) / sample_rate
            freq = 150.0 * math.exp(-4.0 * t)
            buf[i] = np.sin(2.0 * math.pi * freq * t) * math.exp(-3.0 * t)
        noise = np.random.uniform(-0.5, 0.5, n_samples).astype(np.float32) * math.exp(-5.0 * t)
        buf = buf + noise
        buf = np.int16(buf * 32767)
        return pygame.sndarray.make_sound(np.column_stack((buf, buf)))

    def _generate_kill(self, sample_rate, duration):
        n_samples = int(sample_rate * duration)
        buf = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            t = float(i) / sample_rate
            freq = 50.0 + np.random.uniform(-30, 30)
            buf[i] = np.sin(2.0 * math.pi * freq * t) * math.exp(-2.0 * t)
        buf = np.int16(buf * 32767)
        return pygame.sndarray.make_sound(np.column_stack((buf, buf)))

    def play_deflect(self):
        threading.Thread(target=self.deflect_snd.play, daemon=True).start()

    def play_emp(self):
        threading.Thread(target=self.emp_snd.play, daemon=True).start()
        
    def play_hit(self):
        threading.Thread(target=self.hit_snd.play, daemon=True).start()

class VectorMath:
    @staticmethod
    def get_line_segment_intersection(p0, p1, p2, p3):
        s10_x = p1[0] - p0[0]
        s10_y = p1[1] - p0[1]
        s32_x = p3[0] - p2[0]
        s32_y = p3[1] - p2[1]
        denom = s10_x * s32_y - s32_x * s10_y
        if denom == 0: return False
        s02_x = p0[0] - p2[0]
        s02_y = p0[1] - p2[1]
        s_numer = s10_x * s02_y - s10_y * s02_x
        if (s_numer < 0) == (denom > 0): return False
        t_numer = s32_x * s02_y - s32_y * s02_x
        if (t_numer < 0) == (denom > 0): return False
        if ((s_numer > denom) == (denom > 0)) or ((t_numer > denom) == (denom > 0)): return False
        return True

    @staticmethod
    def reflect_velocity(vel, p1, p2):
        v_line = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=np.float32)
        n = np.array([-v_line[1], v_line[0]])
        norm = np.linalg.norm(n)
        if norm == 0: return vel
        n = n / norm
        v = np.array(vel, dtype=np.float32)
        dot = np.dot(v, n)
        v_out = v - 2 * dot * n
        return v_out.tolist()

    @staticmethod
    def point_to_segment_dist(pt, v, w):
        l2 = (v[0]-w[0])**2 + (v[1]-w[1])**2
        if l2 == 0: return np.linalg.norm(np.array(pt)-np.array(v))
        t = max(0, min(1, np.dot(np.array(pt)-np.array(v), np.array(w)-np.array(v)) / l2))
        proj = np.array(v) + t * (np.array(w)-np.array(v))
        return np.linalg.norm(np.array(pt)-proj)

class AegisEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.audio = AudioEngine()
        self.state = "START"
        self.projectiles = []
        self.emp_charge = 0
        self.emp_active = False
        self.emp_radius = 0
        self.MAX_CHARGE = 60
        self.lives = 5
        self.speed_mult = 1.0

    def reset_game(self):
        self.lives = 5
        self.speed_mult = 1.0
        self.emp_charge = 0
        self.emp_active = False
        self.emp_radius = 0
        self.projectiles.clear()
        self.state = "PLAYING"

    def is_palm_gesture_active(self, lms):
        l_w, l_i = lms[15], lms[19]
        r_w, r_i = lms[16], lms[20]
        l_shoulder, r_shoulder = lms[11], lms[12]
        
        l_dist = math.hypot(l_w.x - l_i.x, l_w.y - l_i.y)
        r_dist = math.hypot(r_w.x - r_i.x, r_w.y - r_i.y)
        
        if l_dist > 0.04 and r_dist > 0.04 and l_w.y < l_shoulder.y and r_w.y < r_shoulder.y:
            return True
        return False

    def spawn_projectile(self, target_coord, w, h):
        start = random.choice([(0, 0), (w, 0), (0, h), (w, h)])
        dx = target_coord[0] - start[0]
        dy = target_coord[1] - start[1]
        dist = math.hypot(dx, dy)
        speed = random.uniform(25.0, 45.0) * self.speed_mult
        self.speed_mult += 0.05
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed
        self.projectiles.append({'pos': list(start), 'vel': [vx, vy], 'active': True, 'color': (0, 0, 255)})

    def render_plasma_shields(self, frame, p1, p2, color):
        overlay = frame.copy()
        cv2.line(overlay, p1, p2, (255, 255, 255), 15)
        cv2.line(overlay, p1, p2, color, 45)
        blur = cv2.GaussianBlur(overlay, (31, 31), 0)
        cv2.addWeighted(blur, 0.6, frame, 0.4, 0, frame)

    def draw_centered_text(self, frame, text, y_offset, color, scale, thickness):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, y_offset), font, scale, color, thickness)

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Aegis-Reflect: HUD', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Aegis-Reflect: HUD', 1280, 720)
        frame_count = 0
        flash_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            nose_coords = (w//2, h//2)
            left_arm = right_arm = None

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                l_elbow = (int(lms[13].x * w), int(lms[13].y * h))
                l_wrist = (int(lms[15].x * w), int(lms[15].y * h))
                r_elbow = (int(lms[14].x * w), int(lms[14].y * h))
                r_wrist = (int(lms[16].x * w), int(lms[16].y * h))
                nose_coords = (int(lms[0].x * w), int(lms[0].y * h))
                
                left_arm = (l_elbow, l_wrist)
                right_arm = (r_elbow, r_wrist)

                self.render_plasma_shields(frame, l_elbow, l_wrist, (255, 215, 0))
                self.render_plasma_shields(frame, r_elbow, r_wrist, (255, 215, 0))
                
                if self.state in ["START", "GAMEOVER"]:
                    if self.is_palm_gesture_active(lms):
                        self.reset_game()

                if self.state == "PLAYING":
                    if VectorMath.get_line_segment_intersection(l_elbow, l_wrist, r_elbow, r_wrist):
                        self.emp_charge += 2
                        cv2.circle(frame, (w//2, h//2), self.emp_charge * 2, (0, 255, 255), -1)
                        if self.emp_charge > self.MAX_CHARGE and not self.emp_active:
                            self.emp_active = True
                            self.emp_radius = 10
                            self.projectiles.clear()
                            self.audio.play_emp()
                    else:
                        self.emp_charge = max(0, self.emp_charge - 1)

            if self.state == "START":
                self.draw_centered_text(frame, "SYSTEM STANDBY", h//2 - 50, (0, 215, 255), 2.5, 5)
                self.draw_centered_text(frame, "RAISE OPEN PALMS TO START", h//2 + 50, (255, 255, 255), 1.0, 2)
                
            elif self.state == "GAMEOVER":
                self.draw_centered_text(frame, "SYSTEM FAILURE", h//2 - 50, (0, 0, 255), 3.0, 6)
                self.draw_centered_text(frame, "RAISE OPEN PALMS TO REBOOT", h//2 + 50, (255, 255, 255), 1.0, 2)

            elif self.state == "PLAYING":
                if self.lives <= 0:
                    self.state = "GAMEOVER"
                    continue

                if self.emp_active:
                    self.emp_radius += 50
                    overlay = frame.copy()
                    cv2.circle(overlay, (w//2, h//2), self.emp_radius, (255, 255, 255), -1)
                    cv2.addWeighted(overlay, max(0, 1.0 - (self.emp_radius/2000)), frame, 1.0, 0, frame)
                    if self.emp_radius > 2000:
                        self.emp_active = False
                        self.emp_charge = 0

                spawn_rate = max(8, int(30 / self.speed_mult))
                if frame_count % spawn_rate == 0 and not self.emp_active:
                    self.spawn_projectile(nose_coords, w, h)

                for p in self.projectiles:
                    if not p['active']: continue
                    
                    p['pos'][0] += p['vel'][0]
                    p['pos'][1] += p['vel'][1]
                    pt = p['pos']
                    
                    if pt[0] < -50 or pt[0] > w+50 or pt[1] < -50 or pt[1] > h+50:
                        p['active'] = False
                        continue

                    if math.hypot(pt[0] - nose_coords[0], pt[1] - nose_coords[1]) < 40:
                        p['active'] = False
                        self.lives -= 1
                        self.audio.play_hit()
                        flash_frames = 5
                        continue

                    hit = False
                    if left_arm and VectorMath.point_to_segment_dist(pt, left_arm[0], left_arm[1]) < 30:
                        p['vel'] = VectorMath.reflect_velocity(p['vel'], left_arm[0], left_arm[1])
                        hit = True
                    elif right_arm and VectorMath.point_to_segment_dist(pt, right_arm[0], right_arm[1]) < 30:
                        p['vel'] = VectorMath.reflect_velocity(p['vel'], right_arm[0], right_arm[1])
                        hit = True

                    if hit:
                        self.audio.play_deflect()
                        p['color'] = (0, 255, 0)
                        p['vel'][0] *= 1.5
                        p['vel'][1] *= 1.5

                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), -1)
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 15, p['color'], 4)

                cv2.putText(frame, f"LIVES: {self.lives}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.putText(frame, f"THREAT LEVEL: {self.speed_mult:.2f}x", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 215, 255), 2)

            if flash_frames > 0:
                overlay = frame.copy()
                overlay[:] = (0, 0, 255)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                flash_frames -= 1

            frame_count += 1
            cv2.imshow('Aegis-Reflect: HUD', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AegisEngine()
    app.run()