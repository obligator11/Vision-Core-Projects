import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading
import queue
import pygame
import os
import random

class SpartanCoach:
    def __init__(self):
        # AI Model Initialization
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
        )
        
        # Next-Gen Cybernetic Theme
        self.STARK_CYAN = (255, 255, 0)
        self.NEON_GREEN = (50, 255, 50)
        self.ALERT_RED = (0, 0, 255)
        self.HUD_BG = (15, 15, 20) # Deeper, darker background
        self.HUD_ACCENT = (200, 200, 200)
        
        self.BODY_CONNECTIONS = frozenset([
            conn for conn in self.mp_pose.POSE_CONNECTIONS 
            if conn[0] > 10 and conn[1] > 10 # Filter out face nodes
        ])
        self.landmark_spec = self.mp_drawing.DrawingSpec(color=self.NEON_GREEN, thickness=3, circle_radius=5)
        self.connection_spec = self.mp_drawing.DrawingSpec(color=self.STARK_CYAN, thickness=3)
        
        # Architecture
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = True
        
        # State Machine Variables
        self.exercise_modes = ["BICEP_CURL", "SQUAT", "PUSH_UP"]
        self.current_mode = 0
        self.counter = 0
        self.stage = "UP"
        
        # Biomechanical Telemetry
        self.current_angle = 0
        self.lowest_angle = 180 
        self.energy_percent = 0
        self.fps = 0
        self.form_warning = ""
        self.warning_timer = 0
        self.posture_state = "UNKNOWN"
        self.hands_up_time = 0
        
        # Zero-Latency Audio Engine Setup (PyGame)
        pygame.mixer.init()
        self.sounds = {}
        self.last_spoken = {}
        self.load_audio_cache()
        
        self.play_sound("startup", cooldown=0)

    def load_audio_cache(self):
        """Pre-loads all MP3s into RAM for zero-latency playback"""
        expected_files = [
            "startup", "switch", "half_rep", "legs_straight", 
            "back_sag", "stand_up", "get_on_floor",
            "hype_1", "hype_2", "hype_3"
        ] + [str(i) for i in range(1, 31)] # Handles reps up to 30
        
        for name in expected_files:
            path = f"sounds/{name}.mp3"
            if os.path.exists(path):
                self.sounds[name] = pygame.mixer.Sound(path)
            else:
                self.sounds[name] = None # Failsafe if file is missing

    def play_sound(self, sound_key, text_warning="", cooldown=2.0):
        """Fires audio instantly and optionally updates the visual HUD warning"""
        current_time = time.time()
        
        # Rep numbers bypass the cooldown to keep up with fast workouts
        if sound_key.isdigit() or sound_key not in self.last_spoken or (current_time - self.last_spoken[sound_key]) > cooldown:
            if sound_key in self.sounds and self.sounds[sound_key] is not None:
                self.sounds[sound_key].play()
            
            self.last_spoken[sound_key] = current_time
            
            # Show the warning on the HUD
            if text_warning:
                self.form_warning = text_warning
                self.warning_timer = time.time()

    def trigger_rep_count(self):
        """Fires the rep count audio, and occasionally plays a hype sound"""
        rep_key = str(self.counter)
        if self.counter > 0 and self.counter % 5 == 0:
            hype = random.choice(["hype_1", "hype_2", "hype_3"])
            self.play_sound(rep_key, cooldown=0.5)
            # Add a slight delay for the hype sound so it doesn't overlap the number
            threading.Timer(0.8, lambda: self.play_sound(hype, cooldown=1.0)).start()
        else:
            self.play_sound(rep_key, cooldown=0.5)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / math.pi)
        return 360 - angle if angle > 180.0 else angle

    def check_mode_switch_gesture(self, landmarks):
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        
        # UI LOCK: Hips must be below shoulders (You must be standing upright)
        is_standing_upright = hip_y > shoulder_y + 0.1

        if is_standing_upright and left_wrist.y < nose.y and right_wrist.y < nose.y:
            if self.hands_up_time == 0:
                self.hands_up_time = time.time()
                self.form_warning = "[ SYSTEM OVERRIDE INITIATED ]"
                self.warning_timer = time.time()
            elif time.time() - self.hands_up_time > 1.5: 
                self.current_mode = (self.current_mode + 1) % len(self.exercise_modes)
                self.counter = 0
                self.stage = "UP"
                self.lowest_angle = 180
                
                self.play_sound("switch", text_warning=f"MODE: {self.exercise_modes[self.current_mode]}", cooldown=0)
                self.hands_up_time = 0
                time.sleep(1) 
        else:
            self.hands_up_time = 0

    def process_inference_loop(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.pose.process(image)
                self.result_queue.put((frame, results))
            else:
                time.sleep(0.005)

    def evaluate_kinematics(self, landmarks):
        mode = self.exercise_modes[self.current_mode]
        self.check_mode_switch_gesture(landmarks)

        # ========================================================
        # GLOBAL POSTURE LOCK
        # ========================================================
        shoulder_pos = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        ankle_pos = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        
        dx = abs(shoulder_pos.x - ankle_pos.x)
        dy = abs(shoulder_pos.y - ankle_pos.y)
        
        if dy > dx + 0.05: 
            self.posture_state = "STANDING"
        elif dx > dy + 0.05: 
            self.posture_state = "HORIZONTAL"
        else:
            self.posture_state = "TRANSITIONING"

        # Form Enforcement & Rejection
        if mode == "BICEP_CURL" and self.posture_state == "HORIZONTAL":
            self.play_sound("stand_up", text_warning="POSTURE ERROR: STAND UP", cooldown=4)
            self.energy_percent = 0
            return 
            
        if mode == "SQUAT" and self.posture_state == "HORIZONTAL":
            self.play_sound("stand_up", text_warning="POSTURE ERROR: STAND UP", cooldown=4)
            self.energy_percent = 0
            return 

        
        # ========================================================
        # ISOLATION KINEMATICS
        # ========================================================
        
        if mode == "BICEP_CURL":
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            leg_angle = self.calculate_angle(hip, knee, ankle)
            
            if leg_angle < 140:
                self.play_sound("legs_straight", text_warning="FORM WARNING: KNEES BENT", cooldown=3)
                self.energy_percent = 0
                return

            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            self.current_angle = self.calculate_angle(shoulder, elbow, wrist)
            self.energy_percent = np.interp(self.current_angle, (30, 160), (100, 0))
            
            if self.stage == "DOWN": 
                self.lowest_angle = min(self.lowest_angle, self.current_angle)
                if self.lowest_angle < 100 and self.current_angle > self.lowest_angle + 20:
                    self.play_sound("half_rep", text_warning="REP DENIED: HALF REP", cooldown=3)
                    self.lowest_angle = 180 

            if self.current_angle < 35 and self.stage == 'DOWN':
                self.stage = "UP"
                self.counter += 1
                self.trigger_rep_count()
            elif self.current_angle > 150:
                self.stage = "DOWN"
                self.lowest_angle = 180

        elif mode == "SQUAT":
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            self.current_angle = self.calculate_angle(hip, knee, ankle)
            self.energy_percent = np.interp(self.current_angle, (90, 170), (100, 0))
            
            if self.stage == "UP": 
                self.lowest_angle = min(self.lowest_angle, self.current_angle)
                if self.lowest_angle < 140 and self.current_angle > self.lowest_angle + 20:
                    self.play_sound("half_rep", text_warning="REP DENIED: SQUAT DEEPER", cooldown=3)
                    self.lowest_angle = 180 

            if self.current_angle < 90 and self.stage == 'UP':
                self.stage = "DOWN"
                self.counter += 1
                self.trigger_rep_count()
            elif self.current_angle > 160:
                self.stage = "UP"
                self.lowest_angle = 180

        elif mode == "PUSH_UP":
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # ========================================================
            # ANTI-CURL LOCK: Ratio-Based Torso Check
            # ========================================================
            torso_dx = abs(shoulder[0] - hip[0])
            torso_dy = abs(shoulder[1] - hip[1])
            
            if torso_dy > (torso_dx * 2.0):
                self.play_sound("get_on_floor", text_warning="POSTURE ERROR: GET ON FLOOR", cooldown=4)
                self.energy_percent = 0
                return

            # Only calculate the arm angle! Posture math has been purged.
            self.current_angle = self.calculate_angle(shoulder, elbow, wrist)
            self.energy_percent = np.interp(self.current_angle, (70, 160), (100, 0))

            # Half-rep check
            if self.stage == "UP": 
                self.lowest_angle = min(self.lowest_angle, self.current_angle)
                if self.lowest_angle < 130 and self.current_angle > self.lowest_angle + 20:
                    self.play_sound("half_rep", text_warning="REP DENIED: HALF REP", cooldown=3)
                    self.lowest_angle = 180

            # Rep Success Trigger (No posture requirement!)
            if self.current_angle < 85 and self.stage == 'UP':
                self.stage = "DOWN"
                self.counter += 1
                self.trigger_rep_count()
            elif self.current_angle > 160:
                self.stage = "UP"
                self.lowest_angle = 180

    def draw_corner_brackets(self, frame, x, y, w, h, color, length=20, thickness=3):
        """Draws sci-fi targeting brackets for the HUD"""
        cv2.line(frame, (x, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + length), color, thickness)
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)

    def render_hud(self, frame, results):
        h_cam, w_cam, _ = frame.shape
        
        # 1. Headless Skeleton Override
        if results.pose_landmarks:
            for i in range(11):
                results.pose_landmarks.landmark[i].visibility = 0.0

            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.BODY_CONNECTIONS,
                self.landmark_spec, self.connection_spec
            )
            self.evaluate_kinematics(results.pose_landmarks.landmark)

        # 2. Advanced Transparent Dashboard Base
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (350, 220), self.HUD_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        self.draw_corner_brackets(frame, 20, 20, 330, 200, self.STARK_CYAN)

        # 3. Typography & Telemetry Data
        cv2.putText(frame, "SPARTAN KINEMATICS V7", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.HUD_ACCENT, 1, cv2.LINE_AA)
        cv2.line(frame, (30, 55), (330, 55), self.STARK_CYAN, 1)
        
        mode_str = self.exercise_modes[self.current_mode].replace("_", " ")
        cv2.putText(frame, f"TARGET: {mode_str}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.STARK_CYAN, 2, cv2.LINE_AA)
        
        cv2.putText(frame, "REPS", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.HUD_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{self.counter:02d}", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 2.0, self.NEON_GREEN, 4, cv2.LINE_AA)
        
        cv2.putText(frame, f"STATE: {self.stage}", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.HUD_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(self.fps)}", (150, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.HUD_ACCENT, 1, cv2.LINE_AA)

        # 4. Pop-up Form Warnings (Center Screen, Glitch Effect)
        if time.time() - self.warning_timer < 2.0 and self.form_warning:
            # Draw dark background for text readability
            text_size = cv2.getTextSize(self.form_warning, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w_cam - text_size[0]) // 2
            
            cv2.rectangle(frame, (text_x - 10, 80), (text_x + text_size[0] + 10, 140), self.HUD_BG, -1)
            self.draw_corner_brackets(frame, text_x - 10, 80, text_size[0] + 20, 60, self.ALERT_RED)
            cv2.putText(frame, self.form_warning, (text_x, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.ALERT_RED, 3, cv2.LINE_AA)

        # 5. Segmented Sci-Fi Energy Bar
        bar_x, bar_y = w_cam - 80, 50
        bar_w, bar_h = 40, 300
        
        # Draw Battery Casing
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.HUD_BG, -1)
        self.draw_corner_brackets(frame, bar_x, bar_y, bar_w, bar_h, self.STARK_CYAN, length=10)
        
        # Calculate segments to fill
        segments = 10
        segment_h = (bar_h - (segments * 2)) // segments
        fill_segments = int((self.energy_percent / 100) * segments)

        for i in range(segments):
            y_pos = bar_y + bar_h - ((i + 1) * segment_h) - (i * 2)
            
            if i < fill_segments:
                color = self.NEON_GREEN if not self.form_warning else self.ALERT_RED
                cv2.rectangle(frame, (bar_x + 5, y_pos), (bar_x + bar_w - 5, y_pos + segment_h), color, -1)
            else:
                cv2.rectangle(frame, (bar_x + 5, y_pos), (bar_x + bar_w - 5, y_pos + segment_h), (50, 50, 50), -1)

        cv2.putText(frame, f"{int(self.energy_percent)}%", (bar_x - 10, bar_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.STARK_CYAN, 2)

        return frame

    def launch(self):
        cap = cv2.VideoCapture(0) # Change to 1 if webcam fails to open
        
        # We keep the internal camera capture high for AI accuracy
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        threading.Thread(target=self.process_inference_loop, daemon=True).start()
        prev_time = time.time()

        print("[SYSTEM] Spartan Core Online. Rendering GUI...")


        window_name = 'Sayyam AI Lab: Spartan'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        cv2.resizeWindow(window_name, 1280, 720) 

        while cap.isOpened() and self.running:
            success, raw_frame = cap.read()
            if not success: break
            
            raw_frame = cv2.flip(raw_frame, 1) # Mirror

            if not self.frame_queue.full():
                self.frame_queue.put(raw_frame)

            if not self.result_queue.empty():
                frame, results = self.result_queue.get()
                
                current_time = time.time()
                self.fps = 1 / (current_time - prev_time)
                prev_time = current_time

                output_frame = self.render_hud(frame, results)
                cv2.imshow(window_name, output_frame)

            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('f'): 
                
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    coach = SpartanCoach()
    coach.launch()