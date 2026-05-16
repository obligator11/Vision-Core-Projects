import cv2
import numpy as np
import mediapipe as mp
import math
import random
from collections import deque
import pygame

class AudioEngine:
    """ S-Tier Procedural Audio Generator """
    def __init__(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            self.enabled = True
            sample_rate = 44100
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            noise = np.random.normal(0, 1, len(t))
            envelope = np.exp(-t * 40)
            audio = noise * envelope * 32767
            self.hit_sound = pygame.sndarray.make_sound(audio.astype(np.int16))
        except:
            self.enabled = False

    def play_hit(self):
        if self.enabled:
            self.hit_sound.play()

class Ball:
    """ Z-Axis Physics Engine with Pitch Bounce """
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.pitch_y = int(self.h * 0.8) # The Pitch Line
        self.reset()

    def reset(self):
        # Bowler Release Point
        self.x = random.randint(int(self.w * 0.4), int(self.w * 0.6))
        self.y = int(self.h * 0.2) 
        self.z = 100.0  
        
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(3, 8)       # Thrown heavily downwards
        self.vz = random.uniform(-4.0, -7.0) # Moving towards camera
        self.gravity = 0.5                   # Gravity pulls it to the pitch
        
        self.base_radius = 35
        self.trail = deque(maxlen=10)
        self.is_hit = False
        self.has_bounced = False

    def update(self):
        self.x += self.vx
        self.vy += self.gravity
        self.y += self.vy
        self.z += self.vz
        
        # PITCH BOUNCE LOGIC
        if self.y >= self.pitch_y and not self.has_bounced and not self.is_hit:
            self.y = self.pitch_y
            self.vy = self.vy * -0.85 # Invert Y-velocity (Bounce up) with damping
            self.has_bounced = True
            self.vx += random.uniform(-2, 2) # Spin off the pitch

        if self.z > -5: 
            self.trail.append((int(self.x), int(self.y), self.get_radius()))
            
        if self.z < -20 or self.y > self.h + 200:
            return False
        return True

    def trigger_hit(self, is_six):
        self.is_hit = True
        self.vz = 15.0 if is_six else 8.0     
        self.vy = -30.0 if is_six else -15.0  
        self.vx = random.uniform(-10, 10)
        self.gravity = 1.5 

    def get_radius(self):
        scale = max(0.1, (120 - self.z) / 100.0)
        return int(self.base_radius * scale)

    def draw(self, frame):
        # Draw Pitch Line
        cv2.line(frame, (0, self.pitch_y), (self.w, self.pitch_y), (0, 100, 0), 2)
        
        r = self.get_radius()
        # Draw Trail
        for i in range(len(self.trail)):
            tx, ty, tr = self.trail[i]
            alpha = i / len(self.trail)
            color = (30, 30, int(200 * alpha))
            cv2.circle(frame, (tx, ty), max(2, int(tr * 0.7)), color, -1)

        # Draw Cricket Ball
        cv2.circle(frame, (int(self.x), int(self.y)), r, (40, 40, 200), -1) 
        cv2.circle(frame, (int(self.x), int(self.y)), r, (20, 20, 150), 2)  
        if not self.is_hit:
            cv2.line(frame, (int(self.x - r*0.7), int(self.y)), 
                            (int(self.x + r*0.7), int(self.y)), (255, 255, 255), max(1, int(r*0.1)))

class HoloStrikeEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.cap = cv2.VideoCapture(0)
        self.w = int(self.cap.get(3))
        self.h = int(self.cap.get(4))
        
        self.ball = Ball(self.w, self.h)
        self.audio = AudioEngine()
        
        self.score = 0
        self.hit_text = ""
        self.hit_frames = 0
        self.BAT_LENGTH = 400 # Massive Bat for better coverage

        # Kinematic Extrapolation Variables
        self.last_p1 = None
        self.last_p2 = None
        self.vel_p1 = (0, 0)
        self.vel_p2 = (0, 0)
        self.ghost_frames = 0
        self.prev_wrist_x = 0

        cv2.namedWindow('Sayyam AI Lab: Holo-Strike V3', cv2.WINDOW_NORMAL)

    def draw_bat(self, frame, p1, p2):
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 45) # Hitbox Glow
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (100, 150, 200), 20) # Core

    def point_to_line_dist(self, p1, p2, p3):
        """ Absolute Mathematical Vector Projection for flawless collision """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        px, py = x2 - x1, y2 - y1
        norm = px*px + py*py
        if norm == 0: return math.hypot(x3-x1, y3-y1)
        
        u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
        u = max(0, min(1, u)) # Clamp to line segment
        
        x = x1 + u * px
        y = y1 + u * py
        return math.hypot(x - x3, y - y3)

    def check_collision(self, bat_p1, bat_p2):
        if self.ball.is_hit or self.ball.z > 30 or self.ball.z < -15:
            return False 

        radius = self.ball.get_radius()
        dist = self.point_to_line_dist(bat_p1, bat_p2, (self.ball.x, self.ball.y))
        
        # Generous hitbox
        if dist < (radius + 60):
            return True
        return False

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            bat_p1, bat_p2 = None, None
            current_wrist_x = self.prev_wrist_x

            # 1. LIVE TRACKING
            if results.multi_hand_landmarks:
                self.ghost_frames = 15 # Give the AI a 15-frame safety net
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    index_mcp = hand_landmarks.landmark[5] # Changed to Index Knuckle (More stable than middle finger)
                    
                    wx, wy = int(wrist.x * self.w), int(wrist.y * self.h)
                    mx, my = int(index_mcp.x * self.w), int(index_mcp.y * self.h)
                    current_wrist_x = wx
                    
                    dx, dy = mx - wx, my - wy
                    length = math.hypot(dx, dy)
                    if length > 0:
                        ux, uy = dx / length, dy / length
                        bat_p1 = (mx, my)
                        bat_p2 = (mx + ux * self.BAT_LENGTH, my + uy * self.BAT_LENGTH)
                        
                        # Calculate Velocity for Extrapolation
                        if self.last_p1:
                            self.vel_p1 = (bat_p1[0] - self.last_p1[0], bat_p1[1] - self.last_p1[1])
                            self.vel_p2 = (bat_p2[0] - self.last_p2[0], bat_p2[1] - self.last_p2[1])
                        
                        self.last_p1, self.last_p2 = bat_p1, bat_p2

            # 2. KINEMATIC EXTRAPOLATION (Ghost Bat Fix)
            elif self.ghost_frames > 0 and self.last_p1:
                # If hand vanishes, KEEP SWINGING the bat using its last known velocity
                bat_p1 = (self.last_p1[0] + self.vel_p1[0], self.last_p1[1] + self.vel_p1[1])
                bat_p2 = (self.last_p2[0] + self.vel_p2[0], self.last_p2[1] + self.vel_p2[1])
                
                self.last_p1, self.last_p2 = bat_p1, bat_p2 # Update ghost position
                self.ghost_frames -= 1

            # 3. RENDER AND COLLIDE
            if bat_p1 and bat_p2:
                self.draw_bat(frame, bat_p1, bat_p2)
                
                if self.check_collision(bat_p1, bat_p2):
                    self.audio.play_hit()
                    swing_speed = abs(current_wrist_x - self.prev_wrist_x)
                    
                    # Add velocity from ghost bat if currently extrapolating
                    if self.ghost_frames > 0:
                        swing_speed += abs(self.vel_p1[0])
                        
                    is_six = swing_speed > 25
                    self.ball.trigger_hit(is_six)
                    self.score += 6 if is_six else 4
                    self.hit_text = "SIX!!" if is_six else "FOUR!"
                    self.hit_frames = 20
                    
            self.prev_wrist_x = current_wrist_x

            if not self.ball.update():
                self.ball.reset()

            self.ball.draw(frame)

            # UI Rendering
            cv2.putText(frame, f"RUNS: {self.score}", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            zone_color = (0, 255, 0) if (-15 < self.ball.z < 30) else (0, 0, 255)
            cv2.putText(frame, f"Z-DEPTH: {int(self.ball.z)}", (30, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, zone_color, 2)

            if self.hit_frames > 0:
                color = (0, 215, 255) if self.hit_text == "SIX!!" else (0, 255, 0)
                cv2.putText(frame, self.hit_text, (int(self.w*0.35), int(self.h*0.5)), 
                            cv2.FONT_HERSHEY_TRIPLEX, 5, color, 10)
                self.hit_frames -= 1

            cv2.imshow('Sayyam AI Lab: Holo-Strike V3', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = HoloStrikeEngine()
    game.run()