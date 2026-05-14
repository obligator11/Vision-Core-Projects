import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import random
from collections import deque

# ==============================================================================
# --- STARK-TECH PALETTE (UPGRADED: MAX SATURATION / SHARPER COLORS) ---
# ==============================================================================
COLORS = [
    (0, 0, 255),      # Deep Crimson Red (Maximum Contrast)
    (255, 0, 0),      # Electric Sapphire Blue
    (200, 0, 255),    # Vivid Violet
    (0, 80, 255),     # Burnt Orange
    (0, 0, 0)         # Pitch Black (Looks incredible as AR ink)
]


# ==============================================================================
# --- BLOCK 1: THE CORE ENGINE (HARDWARE OPTIMIZATION & SOLID PAINT) ---
# ==============================================================================
class TensorEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Engine] Rendering initialized on: {self.device}")

    def apply_glow_and_blend(self, raw_frame, canvas_layer):
        glow_canvas = cv2.GaussianBlur(canvas_layer, (21, 21), 0)
        
        t_frame = torch.from_numpy(raw_frame).to(self.device, dtype=torch.float16)
        t_canvas = torch.from_numpy(canvas_layer).to(self.device, dtype=torch.float16)
        t_glow = torch.from_numpy(glow_canvas).to(self.device, dtype=torch.float16)

        # --- OPAQUE PAINT RENDER UPDATE ---
        # 1. The glow remains additive so it looks like projected light
        background = t_frame + (t_glow * 0.8)
        
        # 2. We create a Boolean Mask. Wherever you have drawn paint, we force 
        # the output to be 100% solid canvas color, overwriting the background completely.
        mask = t_canvas > 0
        t_output = torch.where(mask, t_canvas, background)
        
        t_output = torch.clamp(t_output, 0, 255).to(torch.uint8)
        return t_output.cpu().numpy()


# ==============================================================================
# --- BLOCK 2: THE SENSOR (3D MATH & AMPLITUDE SWIPE TRACKING) ---
# ==============================================================================
import math

class KinematicSensor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.85, 
            min_tracking_confidence=0.85
        )
        
        # Shorter memory (15 frames = ~0.25s) for incredibly snappy, fast swipes
        self.swipe_history = deque(maxlen=15)
        self.last_swipe_time = 0
        self.prev_cx = 0
        self.prev_cy = 0

    def get_finger_states(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        tips = [4, 8, 12, 16, 20] 
        joints = [2, 6, 10, 14, 18] 
        
        states = []
        for i in range(5):
            tip = hand_landmarks.landmark[tips[i]]
            joint = hand_landmarks.landmark[joints[i]]
            
            dist_tip = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
            dist_joint = math.sqrt((joint.x - wrist.x)**2 + (joint.y - wrist.y)**2 + (joint.z - wrist.z)**2)
            
            if dist_tip > dist_joint * 1.1:
                states.append(True)
            else:
                states.append(False)
        return states

    def process_kinematics(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        
        action = "NONE"
        coords = None
        z_depth = 0.0
        shift_triggered = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                states = self.get_finger_states(hand_landmarks)
                
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, c = frame.shape
                raw_cx, raw_cy = int(index_tip.x * w), int(index_tip.y * h)
                
                alpha = 0.6 
                if self.prev_cx == 0 and self.prev_cy == 0:
                    self.prev_cx, self.prev_cy = raw_cx, raw_cy
                    
                cx = int(alpha * raw_cx + (1 - alpha) * self.prev_cx)
                cy = int(alpha * raw_cy + (1 - alpha) * self.prev_cy)
                self.prev_cx, self.prev_cy = cx, cy
                
                coords = (cx, cy)
                z_depth = index_tip.z

                fingers_up = sum(states[1:]) 
                
                if states[1] and not states[2] and not states[3] and states[4]:
                    action = "CLEAR"
                    self.swipe_history.clear()
                
                elif fingers_up == 0:
                    action = "ERASE"
                    self.swipe_history.clear()
                    
                elif sum(states) >= 4:
                    action = "OPEN_PALM" 
                    self.swipe_history.append(cx)
                
                elif states[1] and states[2] and not states[3] and not states[4]:
                    action = "SPRAY" 
                    self.swipe_history.clear()
                    
                elif states[1] and not states[2] and not states[3] and not states[4]:
                    action = "DRAW" 
                    self.swipe_history.clear()
                
                else:
                    # REMOVED: self.swipe_history.clear()
                    # By doing nothing here, we forgive motion-blur micro-stutters.
                    pass

        # --- BULLETPROOF SWIPE DETECTION (Amplitude Tracker) ---
        if action == "OPEN_PALM" and len(self.swipe_history) >= 5:
            # Find the maximum and minimum X coordinates in recent memory
            max_x = max(self.swipe_history)
            min_x = min(self.swipe_history)
            
            # If the hand moved more than 120 pixels in any horizontal direction
            if (max_x - min_x) > 120 and (time.time() - self.last_swipe_time > 0.8):
                shift_triggered = True
                self.last_swipe_time = time.time()
                self.swipe_history.clear()

        return action, coords, z_depth, shift_triggered
    

# ==============================================================================
# --- BLOCK 3: THE HOLOGRAPHIC CANVAS (Z-AXIS PHYSICS & MASSIVE ERASER) ---
# ==============================================================================
class HolographicCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        self.color_idx = 0
        self.current_color = COLORS[self.color_idx]
        self.prev_x, self.prev_y = 0, 0

    def shift_color(self):
        self.color_idx = (self.color_idx + 1) % len(COLORS)
        self.current_color = COLORS[self.color_idx]

    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def update_stroke(self, action, coords, z_depth):
        if coords is None:
            self.prev_x, self.prev_y = 0, 0
            return

        x, y = coords
        base_thickness = 10
        z_multiplier = max(1, int(-z_depth * 150)) 
        dynamic_thickness = base_thickness + z_multiplier

        if action == "DRAW":
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = x, y
            
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                     self.current_color, dynamic_thickness, cv2.LINE_AA)
            cv2.circle(self.canvas, (x, y), dynamic_thickness // 2, self.current_color, cv2.FILLED)
            self.prev_x, self.prev_y = x, y

        elif action == "ERASE":
            if self.prev_x == 0 and self.prev_y == 0:
                self.prev_x, self.prev_y = x, y
            
            # --- MASSIVE AOE ERASER (150px thickness) ---
            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), 
                     (0, 0, 0), 150, cv2.LINE_AA) 
            cv2.circle(self.canvas, (x, y), 75, (0, 0, 0), cv2.FILLED)
            self.prev_x, self.prev_y = x, y
            
        elif action == "SPRAY":
            spray_radius = dynamic_thickness * 3
            for _ in range(15): 
                ox = random.randint(-spray_radius, spray_radius)
                oy = random.randint(-spray_radius, spray_radius)
                if (ox**2 + oy**2) <= spray_radius**2:
                    cv2.circle(self.canvas, (x + ox, y + oy), 2, self.current_color, cv2.FILLED)
            self.prev_x, self.prev_y = x, y

        else:
            self.prev_x, self.prev_y = 0, 0


# ==============================================================================
# --- BLOCK 4: THE HUD (STARK-TECH CYBERNETIC UI) ---
# ==============================================================================
class HUDOverlay:
    def __init__(self):
        self.palette_y = 60
        self.palette_x_start = 600
        self.spacing = 80
        
    def draw_tech_bracket(self, img, pt1, pt2, color, thickness=2, length=12):
        """Draws futuristic Sci-Fi corner brackets with node dots."""
        x1, y1 = pt1
        x2, y2 = pt2
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.circle(img, (x1, y1), 3, color, -1)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.circle(img, (x2, y1), 3, color, -1)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.circle(img, (x1, y2), 3, color, -1)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
        cv2.circle(img, (x2, y2), 3, color, -1)

    def draw_ui(self, frame, current_action, current_color_idx, coords):
        h, w, _ = frame.shape
        
        # 1. Translucent Cyber-Grid Navigation Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (5, 5, 10), -1) 
        
        # Draw Tech Grid
        for i in range(0, w, 40):
            cv2.line(overlay, (i, 0), (i, 110), (25, 30, 40), 1)
        for i in range(0, 110, 20):
            cv2.line(overlay, (0, i), (w, i), (25, 30, 40), 1)
            
        # Dual glowing bottom borders
        cv2.line(overlay, (0, 110), (w, 110), (0, 255, 255), 2)
        cv2.line(overlay, (0, 115), (w, 115), (0, 100, 100), 1)
        
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0) 
        
        # System Text
        cv2.putText(frame, "AERO-CANVAS // SYS.OS v3.0", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "TRACKING: OPTIMAL | CORE: SECURE", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # 2. Dynamic Tool Indicators (Glowing Active States)
        tools = [
            ("DRAW", (20, 65), (130, 95)),
            ("ERASE", (150, 65), (270, 95)),
            ("SPRAY", (290, 65), (400, 95))
        ]
        
        for tool_name, pt1, pt2 in tools:
            is_active = current_action == tool_name
            
            if is_active:
                self.draw_tech_bracket(frame, pt1, pt2, (0, 255, 255), thickness=2)
                # Glowing sub-panel
                cv2.rectangle(frame, (pt1[0]+2, pt1[1]+2), (pt2[0]-2, pt2[1]-2), (0, 60, 60), -1)
                cv2.putText(frame, tool_name, (pt1[0] + 25, pt1[1] + 22), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, pt1, pt2, (50, 50, 50), 1)
                cv2.putText(frame, tool_name, (pt1[0] + 25, pt1[1] + 22), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

        # 3. Dynamic Color Palette (Crosshair Locked)
        cv2.putText(frame, "PALETTE_CORE", (self.palette_x_start - 120, self.palette_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        for i, color in enumerate(COLORS):
            x = self.palette_x_start + (i * self.spacing)
            y = self.palette_y
            
            if i == current_color_idx:
                # Multi-ring Targeting Reticle for Active Color
                cv2.circle(frame, (x, y), 24, (255, 255, 255), 1, cv2.LINE_AA) 
                cv2.circle(frame, (x, y), 28, (0, 255, 255), 1, cv2.LINE_AA) 
                cv2.circle(frame, (x, y), 16, color, cv2.FILLED, cv2.LINE_AA)
                # Outer Crosshairs
                cv2.line(frame, (x-35, y), (x-20, y), (0, 255, 255), 2)
                cv2.line(frame, (x+20, y), (x+35, y), (0, 255, 255), 2)
                cv2.line(frame, (x, y-35), (x, y-20), (0, 255, 255), 2)
                cv2.line(frame, (x, y+20), (x, y+35), (0, 255, 255), 2)
            else:
                cv2.circle(frame, (x, y), 12, color, 2, cv2.LINE_AA)

        # 4. Holographic HUD Warnings
        if current_action == "CLEAR":
            cv2.putText(frame, "[ ! ] SYSTEM PURGE OVERRIDE [ ! ]", (w//2 - 260, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        elif current_action == "OPEN_PALM":
            cv2.putText(frame, ">>> COLOR SHIFT DETECTED <<<", (w//2 - 220, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # 5. NEW: MASSIVE REPULSOR ERASER RETICLE
        if current_action == "ERASE" and coords is not None:
            ex, ey = coords
            radius = 75 # Matches the new physical eraser size perfectly
            
            # Inner and Outer Rings
            cv2.circle(frame, (ex, ey), radius, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), radius - 12, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (ex, ey), 4, (0, 255, 255), -1) # Center dot
            
            # Sci-Fi Crosshairs
            cv2.line(frame, (ex - radius - 20, ey), (ex - radius + 10, ey), (0, 255, 255), 2)
            cv2.line(frame, (ex + radius - 10, ey), (ex + radius + 20, ey), (0, 255, 255), 2)
            cv2.line(frame, (ex, ey - radius - 20), (ex, ey - radius + 10), (0, 255, 255), 2)
            cv2.line(frame, (ex, ey + radius - 10), (ex, ey + radius + 20), (0, 255, 255), 2)

        return frame

# ==============================================================================
# --- BLOCK 5: MAIN EXECUTION LOOP ---
# ==============================================================================
def main():
    print("Initiating Project 'Aero-Canvas' (V2 Upgraded)...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ret, frame = cap.read()
    if not ret: return

    h, w, c = frame.shape
    
    sensor = KinematicSensor()
    canvas = HolographicCanvas(w, h)
    engine = TensorEngine()
    hud = HUDOverlay() 

    fps_history = deque(maxlen=30)
    
    # --- NEW: SYSTEM FLAG TO ALLOW WINDOW RESIZING ---
    cv2.namedWindow("Sayyam AI Lab: Aero-Canvas", cv2.WINDOW_NORMAL)

    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # 1. Sensor Phase
        action, coords, z_depth, shift_triggered = sensor.process_kinematics(frame)
        
        # 2. Logic Phase
        if shift_triggered:
            canvas.shift_color()
        if action == "CLEAR":
            canvas.clear()
            
        canvas.update_stroke(action, coords, z_depth)
        
        # 3. Engine Phase 
        output_frame = engine.apply_glow_and_blend(frame, canvas.canvas)
        
        # 4. HUD Phase 
        output_frame = hud.draw_ui(output_frame, action, canvas.color_idx, coords)
        
        # FPS Counter
        fps = 1.0 / (time.time() - start_time)
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        cv2.putText(output_frame, f"FPS: {int(avg_fps)}", (20, 700), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Sayyam AI Lab: Aero-Canvas", output_frame)
        
        if cv2.waitKey(1) & 0xFF == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()