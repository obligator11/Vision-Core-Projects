import cv2
import numpy as np
import mediapipe as mp
import multiprocessing as mp_os
import math
import random
import pygame

def make_synth_sound(freq, duration, wave_type='sine'):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if wave_type == 'sine':
        wave = np.sin(freq * t * 2 * np.pi)
    elif wave_type == 'square':
        wave = np.sign(np.sin(freq * t * 2 * np.pi))
    elif wave_type == 'noise':
        wave = np.random.uniform(-1, 1, len(t)) * np.exp(-t * 5)
        
    sound_array = np.int16(wave * 0.3 * 32767)
    
    # Stereo Patch: Duplicate the mono wave into a 2D stereo wave
    stereo_array = np.column_stack((sound_array, sound_array)) 
    
    return pygame.sndarray.make_sound(stereo_array)

def vision_worker(q_in, q_out):
    mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while True:
        data = q_in.get()
        if data is None:
            break
        h_res = mp_hands.process(data)
        p_res = mp_pose.process(data)
        head_pos = None
        if h_res.multi_hand_landmarks:
            lm = h_res.multi_hand_landmarks[0].landmark[8]
            head_pos = (lm.x, lm.y)
        
        boxes = []
        if p_res.pose_landmarks:
            lms = p_res.pose_landmarks.landmark
            def gb(idx, pad=0.08):
                xs = [lms[i].x for i in idx]
                ys = [lms[i].y for i in idx]
                return (min(xs)-pad, min(ys)-pad, max(xs)+pad, max(ys)+pad)
            
            # CORE FIX: Only map the Head and Torso. Arms are immune.
            boxes.append(gb([0,1,2,3,4,5,6,7,8,9,10])) # Head
            boxes.append(gb([11,12,23,24]))            # Torso
            
        q_out.put((head_pos, boxes))

class OuroborosEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        self.snd_eat = make_synth_sound(600, 0.1, 'sine')
        self.snd_gold = make_synth_sound(1200, 0.3, 'sine')
        self.snd_die = make_synth_sound(100, 0.8, 'noise')
        
        self.q_in = mp_os.Queue()
        self.q_out = mp_os.Queue()
        self.worker = mp_os.Process(target=vision_worker, args=(self.q_in, self.q_out))
        self.worker.daemon = True
        self.worker.start()
        
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow("Ouroboros", cv2.WINDOW_NORMAL)
        
        self.state = "START"
        self.reset_game()
        self.head_pos = None
        self.boxes = []
        self.wave_history = []

    def reset_game(self):
        self.history = []
        self.max_len = 25
        self.score = 0
        self.spawn_apple()
        self.emp_radius = 0

    def spawn_apple(self):
        x = random.uniform(0.2, 0.8)
        y = random.uniform(0.2, 0.8)
        z = random.uniform(0.5, 1.5)
        is_gold = random.random() < 0.15
        self.apple = (x, y, z, is_gold)

    def detect_wave(self, w):
        if not self.head_pos:
            return False
        self.wave_history.append(self.head_pos[0] * w)
        
        # Relaxed Wave Threshold
        if len(self.wave_history) > 50:
            self.wave_history.pop(0)
            if max(self.wave_history) - min(self.wave_history) > w * 0.15:
                self.wave_history.clear()
                return True
        return False

    def execute(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            if self.q_in.empty():
                small = cv2.resize(frame, (320, 240))
                self.q_in.put(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            
            while not self.q_out.empty():
                self.head_pos, self.boxes = self.q_out.get_nowait()
            
            frame = cv2.convertScaleAbs(frame, alpha=0.25, beta=0)
            mask = np.zeros_like(frame)
            
            if self.state == "START":
                cv2.putText(frame, "WAVE HAND TO START", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                if self.detect_wave(w):
                    self.state = "PLAYING"
                    self.reset_game()
                    
            elif self.state == "GAMEOVER":
                self.emp_radius += 30
                cv2.circle(mask, (w//2, h//2), self.emp_radius, (0, 0, 255), -1)
                frame = cv2.addWeighted(frame, 1.0, mask, 0.7, 0)
                cv2.putText(frame, "GAME OVER - WAVE TO RESTART", (w//2 - 300, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                if self.detect_wave(w):
                    self.state = "PLAYING"
                    self.reset_game()
                    
            elif self.state == "PLAYING":
                # Endless Mode: Removed the "Hide to Win" logic here.
                
                for b in self.boxes:
                    x1, y1 = int(b[0]*w), int(b[1]*h)
                    x2, y2 = int(b[2]*w), int(b[3]*h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 20, 20), 2)
                
                if self.head_pos:
                    hx, hy = int(self.head_pos[0]*w), int(self.head_pos[1]*h)
                    self.history.append((hx, hy))
                    if len(self.history) > self.max_len:
                        self.history.pop(0)
                    
                    ax, ay = int(self.apple[0]*w), int(self.apple[1]*h)
                    ar = int(15 * self.apple[2])
                    
                    if math.hypot(hx - ax, hy - ay) < ar + 20:
                        if self.apple[3]:
                            self.snd_gold.play()
                            half = max(10, len(self.history) // 2)
                            self.history = self.history[-half:]
                            self.max_len = max(25, self.max_len // 2)
                        else:
                            self.snd_eat.play()
                            self.max_len += 15
                        self.score += 1
                        self.spawn_apple()
                    
                    if len(self.history) > 30:
                        for px, py in self.history[:-25]:
                            for b in self.boxes:
                                if int(b[0]*w) < px < int(b[2]*w) and int(b[1]*h) < py < int(b[3]*h):
                                    self.state = "GAMEOVER"
                                    self.snd_die.play()
                                    break
                
                if len(self.history) > 1:
                    pts = np.array(self.history, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(mask, [pts], False, (0, 255, 0), 12)
                    cv2.circle(mask, self.history[-1], 18, (0, 255, 100), -1)
                
                ax, ay = int(self.apple[0]*w), int(self.apple[1]*h)
                ar = int(15 * self.apple[2])
                ac = (0, 215, 255) if self.apple[3] else (0, 150, 255)
                cv2.circle(mask, (ax, ay), ar, ac, -1)
                cv2.circle(mask, (ax, ay), int(ar*0.3), (255, 255, 255), -1)
                
                glow = cv2.GaussianBlur(mask, (35, 35), 0)
                frame = cv2.addWeighted(frame, 1.0, glow, 1.5, 0)
                frame = cv2.add(frame, mask)
                
                cv2.putText(frame, f"APPLES: {self.score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Ouroboros", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        self.q_in.put(None)
        self.cap.release()
        cv2.destroyAllWindows()
        self.worker.join()

if __name__ == "__main__":
    OuroborosEngine().execute()