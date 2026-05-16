import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import threading
import queue
import time
import math

# PyAutoGUI Safety Failsafe (Move mouse to corner to abort)
pyautogui.FAILSAFE = True
# Disable PyAutoGUI's default pause to allow zero-latency threaded execution
pyautogui.PAUSE = 0 

class OmniGraspEngine:
    def __init__(self):
        print("[SYS] Initializing Sayyam AI Lab: Omni-Grasp Engine...")
        
        # --- MediaPipe Setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.8, 
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # --- OS Command Queue (The Decoupler) ---
        self.cmd_queue = queue.Queue()
        self.executor_thread = threading.Thread(target=self._os_executor, daemon=True)
        self.executor_thread.start()
        
        # --- EMA Smoothing Variables ---
        self.screen_w, self.screen_h = pyautogui.size()
        self.ema_x, self.ema_y = self.screen_w / 2, self.screen_h / 2
        self.alpha = 0.35 # EMA smoothing factor (Lower = smoother but slight lag)
        
        # --- State Machine Trackers ---
        self.PINCH_THRESHOLD = 0.04 # Normalized distance
        self.is_dragging = False
        self.rc_lock = False
        self.macro_lock = False
        self.scroll_anchor = None
        
        # UI Colors (BGR)
        self.COLOR_PRIMARY = (0, 255, 0)   # Green for Left Click
        self.COLOR_SECONDARY = (0, 0, 255) # Red for Right Click
        self.COLOR_SCROLL = (255, 215, 0)  # Cyan for Scroll
        self.COLOR_OVERRIDE = (255, 0, 255)# Magenta for Macro

    def _os_executor(self):
        """Asynchronous worker that processes OS commands without blocking the vision loop."""
        while True:
            cmd = self.cmd_queue.get()
            if cmd[0] == 'MOVE':
                pyautogui.moveTo(cmd[1], cmd[2])
            elif cmd[0] == 'MOUSE_DOWN':
                pyautogui.mouseDown()
            elif cmd[0] == 'MOUSE_UP':
                pyautogui.mouseUp()
            elif cmd[0] == 'RIGHT_CLICK':
                pyautogui.click(button='right')
            elif cmd[0] == 'SCROLL':
                pyautogui.scroll(cmd[1])
            elif cmd[0] == 'MACRO':
                # OS-Level Task Switcher (Alt + Tab / Win + D)
                pyautogui.hotkey('win', 'd') 
            self.cmd_queue.task_done()

    def get_euclidean_distance(self, p1, p2):
        """Calculates the sub-millimeter Euclidean distance between two landmarks."""
        return math.hypot(p2.x - p1.x, p2.y - p1.y)

    def draw_haptic_shockwave(self, frame, center_x, center_y, color):
        """Draws an alpha-blended Gaussian blur shockwave over the active pinch."""
        overlay = frame.copy()
        cv2.circle(overlay, (center_x, center_y), 50, color, -1)
        # Apply Gaussian Blur to the glow core
        overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
        return cv2.addWeighted(overlay, 0.5, frame, 0.7, 0)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Phase 1: The Canvas (Adjustable Ingestion)
        cv2.namedWindow('Sayyam AI Lab: Omni-Grasp', cv2.WINDOW_NORMAL)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Mirror frame for natural movement
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe Inference
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw cybernetic mesh
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )

                    landmarks = hand_landmarks.landmark
                    
                    # Target Knuckles
                    thumb = landmarks[4]
                    index = landmarks[8]
                    middle = landmarks[12]
                    ring = landmarks[16]
                    pinky = landmarks[20]

                    # Phase 2: Navigation Engine (EMA Math)
                    raw_x = index.x * self.screen_w
                    raw_y = index.y * self.screen_h
                    
                    # Calculate Exponential Moving Average to destroy cursor jitter
                    self.ema_x = (raw_x * self.alpha) + (self.ema_x * (1 - self.alpha))
                    self.ema_y = (raw_y * self.alpha) + (self.ema_y * (1 - self.alpha))
                    
                    # Queue mouse movement
                    self.cmd_queue.put(('MOVE', int(self.ema_x), int(self.ema_y)))

                    # Phase 3: 4-State Kinematic Trigger Calculations
                    dist_index = self.get_euclidean_distance(thumb, index)
                    dist_middle = self.get_euclidean_distance(thumb, middle)
                    dist_ring = self.get_euclidean_distance(thumb, ring)
                    dist_pinky = self.get_euclidean_distance(thumb, pinky)

                    px_thumb, py_thumb = int(thumb.x * w), int(thumb.y * h)

                    # State 1: Primary (Left Click & Drag)
                    if dist_index < self.PINCH_THRESHOLD:
                        frame = self.draw_haptic_shockwave(frame, px_thumb, py_thumb, self.COLOR_PRIMARY)
                        if not self.is_dragging:
                            self.cmd_queue.put(('MOUSE_DOWN',))
                            self.is_dragging = True
                    elif self.is_dragging:
                        self.cmd_queue.put(('MOUSE_UP',))
                        self.is_dragging = False

                    # State 2: Secondary (Right Click)
                    if dist_middle < self.PINCH_THRESHOLD:
                        frame = self.draw_haptic_shockwave(frame, px_thumb, py_thumb, self.COLOR_SECONDARY)
                        if not self.rc_lock:
                            self.cmd_queue.put(('RIGHT_CLICK',))
                            self.rc_lock = True
                    else:
                        self.rc_lock = False

                    # State 3: The Scroller (Lock and drag Y-axis)
                    if dist_ring < self.PINCH_THRESHOLD:
                        frame = self.draw_haptic_shockwave(frame, px_thumb, py_thumb, self.COLOR_SCROLL)
                        if self.scroll_anchor is None:
                            self.scroll_anchor = thumb.y
                        else:
                            # Calculate Y-axis delta
                            delta = self.scroll_anchor - thumb.y
                            if abs(delta) > 0.01: # Small deadzone
                                scroll_mag = int(delta * 2000) # Scaling factor
                                self.cmd_queue.put(('SCROLL', scroll_mag))
                                self.scroll_anchor = thumb.y # Reset anchor
                    else:
                        self.scroll_anchor = None

                    # State 4: The Override (Macro)
                    if dist_pinky < self.PINCH_THRESHOLD:
                        frame = self.draw_haptic_shockwave(frame, px_thumb, py_thumb, self.COLOR_OVERRIDE)
                        if not self.macro_lock:
                            self.cmd_queue.put(('MACRO',))
                            self.macro_lock = True
                    else:
                        self.macro_lock = False

            # Render UI
            cv2.putText(frame, "Sayyam AI: Omni-Grasp Engine Active", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('Sayyam AI Lab: Omni-Grasp', frame)

            if cv2.waitKey(1) & 0xFF == 27: # Press ESC to quit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = OmniGraspEngine()
    engine.run()