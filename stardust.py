import cv2
import mediapipe as mp
import numpy as np
import math
import time
from threading import Thread
import queue

# ==========================================
# MATHEMATICAL COLLISION ENGINE
# ==========================================
def calculate_intersection(p1, p2, p3, p4):
    """Calculates the intersection point of two line segments in 2D space."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = int(x1 + t * (x2 - x1))
        iy = int(y1 + t * (y2 - y1))
        return (ix, iy)
    return None

# ==========================================
# PARTICLE VFX SYSTEM
# ==========================================
class ParticleSystem:
    def __init__(self):
        self.particles = []

    def spawn_sparks(self, origin, count=15):
        for _ in range(count):
            vx = np.random.uniform(-15, 15)
            vy = np.random.uniform(-15, 5) # Upward/outward burst
            life = np.random.uniform(5, 15)
            self.particles.append([list(origin), [vx, vy], life])

    def update_and_draw(self, frame):
        for p in self.particles[:]:
            pos, vel, life = p
            # Apply gravity
            vel[1] += 1.5 
            pos[0] += vel[0]
            pos[1] += vel[1]
            p[2] -= 1

            if p[2] <= 0:
                self.particles.remove(p)
            else:
                cv2.circle(frame, (int(pos[0]), int(pos[1])), int(life // 3) + 1, (255, 255, 255), -1)
                cv2.circle(frame, (int(pos[0]), int(pos[1])), int(life // 2) + 2, (0, 215, 255), 1)

# ==========================================
# PLASMA BLADE CLASS
# ==========================================
class PlasmaBlade:
    def __init__(self, color):
        self.color = color
        self.is_ignited = False
        self.base = None
        self.tip = None
        self.trail = [] # For motion blur

    def update(self, hand_landmarks, frame_shape):
        h, w, _ = frame_shape
        # Extract coordinates
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Calculate pixel coordinates
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        mx, my = int(middle_mcp.x * w), int(middle_mcp.y * h)
        
        # Fist Detection (Distance from fingertips to palm base)
        fingertips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
        is_fist = True
        for tip_idx in fingertips:
            tip = hand_landmarks.landmark[tip_idx]
            tx, ty = int(tip.x * w), int(tip.y * h)
            dist = math.hypot(tx - wx, ty - wy)
            if dist > 100: # Threshold for open hand
                is_fist = False
                break

        if is_fist:
            self.is_ignited = True
            # Vector Projection
            length = 400
            angle = math.atan2(my - wy, mx - wx)
            tip_x = int(mx + length * math.cos(angle))
            tip_y = int(my + length * math.sin(angle))
            
            self.base = (mx, my)
            self.tip = (tip_x, tip_y)
            
            # Motion trail buffer
            self.trail.append((self.base, self.tip))
            if len(self.trail) > 4:
                self.trail.pop(0)
        else:
            self.is_ignited = False
            self.base = None
            self.tip = None
            self.trail.clear()

    def draw(self, frame, glow_layer):
        if not self.is_ignited or not self.base or not self.tip:
            return

        # Draw Motion Blur / Trails
        for i, (p_base, p_tip) in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail)
            thickness = int(15 * alpha)
            cv2.line(glow_layer, p_base, p_tip, self.color, thickness)

        # Draw Core Blade
        cv2.line(glow_layer, self.base, self.tip, self.color, 30) # Outer Aura
        cv2.line(glow_layer, self.base, self.tip, self.color, 15) # Inner Aura
        cv2.line(frame, self.base, self.tip, (255, 255, 255), 6)  # White Hot Core

# ==========================================
# ASYNCHRONOUS ENGINE
# ==========================================
class KyberEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7
        )
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = True
        
        self.particles = ParticleSystem()
        
        # Player 1 (Blue/Green), Player 2 (Sith Red)
        self.blades = {
            "Right": PlasmaBlade((255, 100, 0)),  # Jedi Blue (BGR)
            "Left": PlasmaBlade((0, 0, 255))      # Sith Red (BGR)
        }

    def inference_thread(self):
        """Runs MediaPipe on a separate thread to bypass I/O locking."""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if self.result_queue.full():
                    self.result_queue.get()
                self.result_queue.put(results)
            time.sleep(0.001)

    def run(self):
        cv2.namedWindow("Project Kyber - AR Combat", cv2.WINDOW_NORMAL)
        
        # Start Inference Thread
        thread = Thread(target=self.inference_thread, daemon=True)
        thread.start()

        prev_time = time.time()

        while self.cap.isOpened() and self.running:
            success, frame = self.cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1) # Mirror for AR interaction
            
            # Feed frame to inference queue
            if self.frame_queue.empty():
                self.frame_queue.put(frame.copy())

            # Rendering Layers
            glow_layer = np.zeros_like(frame)

            # Process Results
            if not self.result_queue.empty():
                results = self.result_queue.get()
                
                active_blades = []
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label # 'Left' or 'Right'
                        blade = self.blades.get(label)
                        
                        if blade:
                            blade.update(hand_landmarks, frame.shape)
                            blade.draw(frame, glow_layer)
                            if blade.is_ignited:
                                active_blades.append(blade)

                # Collision Engine (The Clash)
                if len(active_blades) == 2:
                    b1, b2 = active_blades[0], active_blades[1]
                    intersect = calculate_intersection(b1.base, b1.tip, b2.base, b2.tip)
                    
                    if intersect:
                        # Massive Flash
                        cv2.circle(frame, intersect, 40, (255, 255, 255), -1)
                        cv2.circle(glow_layer, intersect, 80, (255, 255, 255), -1)
                        self.particles.spawn_sparks(intersect, count=10)

            # Update particles
            self.particles.update_and_draw(frame)

            # Apply Additive Blending (Neon Aura)
            glow_layer = cv2.GaussianBlur(glow_layer, (31, 31), 0)
            final_frame = cv2.addWeighted(frame, 1.0, glow_layer, 0.8, 0)

            # FPS Counter
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(final_frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Project Kyber - AR Combat", final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = KyberEngine()
    engine.run()
    