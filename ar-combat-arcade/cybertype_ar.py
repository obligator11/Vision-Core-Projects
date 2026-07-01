import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import random
import os

# ==========================================
# ⚙️ SYSTEM SETTINGS & CONSTANTS
# ==========================================
CAM_W, CAM_H = 1280, 720
TERM_W, TERM_H = 800, 400

# The Goldilocks Thresholds
VELOCITY_DROP_THRESH = 0.6  # High enough to ignore hovering, low enough to catch taps
VELOCITY_STOP_THRESH = 0.15 
KEYBOARD_ZONE = (0.2, 0.6, 0.8, 1.0) # Shrink the box back down to the actual desk area

# ==========================================
# 🔊 AUDIO MANAGER (Zero Latency)
# ==========================================
class AudioManager:
    def __init__(self):
        # Pre-initialize mixer for zero-latency audio before pygame.init()
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        
        self.clack_sound = None
        self.hum_sound = None
        
        # Safely load sounds if they exist, otherwise bypass gracefully
        try:
            if os.path.exists("mechanical_clack.wav"):
                self.clack_sound = pygame.mixer.Sound("mechanical_clack.wav")
            if os.path.exists("mainframe_hum.wav"):
                self.hum_sound = pygame.mixer.Sound("mainframe_hum.wav")
                self.hum_sound.play(-1) # Loop infinitely
        except Exception as e:
            print(f"[AUDIO WARN] Could not initialize audio assets: {e}")

    def play_clack(self):
        if self.clack_sound:
            # Play on an available channel to prevent cutting off overlapping fast types
            pygame.mixer.find_channel().play(self.clack_sound)

# ==========================================
# 🖥️ HOLOGRAPHIC TERMINAL ENGINE
# ==========================================
class TerminalRenderer:
    def __init__(self, width, height):
        self.w, self.h = width, height
        # Hidden surface for drawing text
        self.surface = pygame.Surface((self.w, self.h))
        self.font = pygame.font.SysFont('consolas', 24, bold=True)
        self.lines = []
        self.max_lines = 14
        
        # Pre-fill with boot sequence
        self.lines.append("SYSTEM BOOT...")
        self.lines.append("INITIALIZING NEURAL LINK...")

    def trigger_keystroke(self):
        """Generates random hacker-style strings on keystroke"""
        code_types = [
            f"0x{random.randint(1000, 999999):06X} : MEMORY_ALLOC_OK",
            f"CONNECTING -> 192.168.{random.randint(0,255)}.{random.randint(1,254)}...",
            "DECRYPTING SECTOR 7G...",
            "".join(random.choices(['0', '1', 'A', 'F', 'X', '*'], k=30)),
            "ACCESS GRANTED."
        ]
        self.lines.append(random.choice(code_types))
        if len(self.lines) > self.max_lines:
            self.lines.pop(0)

    def render(self):
        """Draws the scrolling text to the hidden surface"""
        # Semi-transparent dark background for the floating screen
        self.surface.fill((10, 15, 10)) 
        
        # Render text lines upward
        y_offset = 20
        for line in self.lines:
            text_surf = self.font.render(line, True, (0, 255, 65)) # Matrix Green
            self.surface.blit(text_surf, (20, y_offset))
            y_offset += 26
            
        return self.surface

# ==========================================
# 🖐️ KINEMATIC KEYSTROKE ESTIMATOR
# ==========================================
class KeystrokeEstimator:
    def __init__(self):
        self.prev_tips = []
        self.prev_time = time.time()
        self.dropping_state = {} # Tracks fingers that are currently falling fast

    def estimate(self, current_tips):
        curr_time = time.time()
        dt = curr_time - self.prev_time
        if dt == 0: dt = 0.001
        
        strike_detected = False
        new_dropping_state = {}

        # If we have history, compare
        if self.prev_tips:
            for i, (cx, cy) in enumerate(current_tips):
                # Find nearest previous tip (simplistic finger tracking across frames)
                dists = [np.hypot(cx - px, cy - py) for (px, py) in self.prev_tips]
                if not dists: continue
                min_idx = np.argmin(dists)
                px, py = self.prev_tips[min_idx]
                
                # Calculate Y-Velocity (Positive = moving DOWN)
                vy = (cy - py) / dt
                
                # 1. First, check if finger is inside keyboard bounding box
                in_zone = (KEYBOARD_ZONE[0] < cx < KEYBOARD_ZONE[2] and 
                           KEYBOARD_ZONE[1] < cy < KEYBOARD_ZONE[3])

                # 2. NOW we can safely print the debug speed
                if in_zone and vy > 0.3: 
                    print(f"Finger Speed: {vy:.2f}") 

                # 3. Process the strike logic
                if in_zone:
                    # Logic: If it was dropping fast previously, and now stopped abruptly -> Strike!
                    if self.dropping_state.get(min_idx, False) and vy < VELOCITY_STOP_THRESH:
                        strike_detected = True
                    
                    # Update state: Is it currently dropping fast?
                    if vy > VELOCITY_DROP_THRESH:
                        new_dropping_state[i] = True
                    else:
                        new_dropping_state[i] = False

        self.dropping_state = new_dropping_state
        self.prev_tips = current_tips
        self.prev_time = curr_time
        
        return strike_detected

# ==========================================
# 🚀 MAIN APPLICATION ORCHESTRATOR
# ==========================================
class CyberTypeARApp:
    def __init__(self):
        self.audio = AudioManager()
        self.terminal = TerminalRenderer(TERM_W, TERM_H)
        self.keystroke_engine = KeystrokeEstimator()
        
        # Setup OpenCV
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        
        # Setup MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Setup Pygame Display
        self.screen = pygame.display.set_mode((CAM_W, CAM_H), pygame.RESIZABLE)
        pygame.display.set_caption("CyberType AR - Spatial Computing Console")
        self.clock = pygame.time.Clock()

        # Homography Points (The 3D Trapezoid projection in the camera)
        # We project the UI to float *above* the keyboard zone, tilted back 45 degrees
        self.src_pts = np.float32([[0, 0], [TERM_W, 0], [TERM_W, TERM_H], [0, TERM_H]])
        offset_y = 100
        self.dst_pts = np.float32([
            [CAM_W*0.25, CAM_H*0.1 + offset_y],  # Top Left (Pinched in for perspective)
            [CAM_W*0.75, CAM_H*0.1 + offset_y],  # Top Right
            [CAM_W*0.90, CAM_H*0.5 + offset_y],  # Bottom Right (Wider, closer to camera)
            [CAM_W*0.10, CAM_H*0.5 + offset_y]   # Bottom Left
        ])
        # Calculate Homography Matrix once
        self.H_matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

    def extract_fingertips(self, hand_landmarks):
        """Extracts the 5 fingertip coordinates from a hand"""
        tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        tips = []
        for id in tip_ids:
            lm = hand_landmarks.landmark[id]
            tips.append((lm.x, lm.y)) # Keep normalized for consistent velocity math
        return tips

    def run(self):
        running = True
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            # 2. Capture & Process OpenCV Frame
            success, frame = self.cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1) # Mirror for AR interaction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 3. MediaPipe Hand Tracking
            results = self.hands.process(rgb_frame)
            all_tips = []
            
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    all_tips.extend(self.extract_fingertips(hand_lms))
            
            # 4. Keystroke Estimation Math
            if self.keystroke_engine.estimate(all_tips):
                self.terminal.trigger_keystroke()
                self.audio.play_clack()
                
            # 5. Render Hidden Terminal & Warp (Homography)
            term_surf = self.terminal.render()
            
            # Convert Pygame Surface -> NumPy Array -> OpenCV BGR
            term_array = pygame.surfarray.array3d(term_surf)
            term_array = np.transpose(term_array, (1, 0, 2)) # Pygame (X,Y,C) -> OpenCV (Y,X,C)
            term_bgr = cv2.cvtColor(term_array, cv2.COLOR_RGB2BGR)
            
            # Apply Inverse Perspective Mapping
            warped_terminal = cv2.warpPerspective(term_bgr, self.H_matrix, (CAM_W, CAM_H))
            
            # Additive Blending (Create Hologram effect over live feed)
            mask = cv2.cvtColor(warped_terminal, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            
            # Composite holographic projection onto the real-world feed
            frame_bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame_bg, warped_terminal)

            # Draw Keyboard Reference Zone (Debug/Visual aid)
            x1, y1 = int(KEYBOARD_ZONE[0]*CAM_W), int(KEYBOARD_ZONE[1]*CAM_H)
            x2, y2 = int(KEYBOARD_ZONE[2]*CAM_W), int(KEYBOARD_ZONE[3]*CAM_H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(frame, "PHYSICAL KEYBOARD ZONE", (x1+10, y1+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

            # 6. Push to Pygame Window Dynamically
            rgb_final = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            final_surf = pygame.image.frombuffer(rgb_final.tobytes(), rgb_final.shape[1::-1], "RGB")
            
            # Scale to current window size for responsive UI
            current_w, current_h = self.screen.get_size()
            scaled_surf = pygame.transform.scale(final_surf, (current_w, current_h))
            
            self.screen.blit(scaled_surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)

        # Teardown
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    app = CyberTypeARApp()
    app.run()