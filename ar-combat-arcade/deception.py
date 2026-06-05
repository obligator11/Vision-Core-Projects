import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import time

class GlitchDeceptionEngine:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.sound_correct = self._synthesize_tone(600, 0.15, type='sine')
        self.sound_wrong = self._synthesize_tone(150, 0.3, type='sawtooth')
        self.sound_glitch = self._synthesize_tone(800, 0.08, type='noise')

        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.state = "START"  # START, NEXT_PROMPT, WAITING_FOR_INPUT, FAILED
        self.score = 0
        self.high_score = 0
        self.deception_rate = 0.30  # Fixed predictable deception curve factor
        
        self.prompts = ["LEFT", "RIGHT", "DUCK", "JUMP"]
        self.current_prompt = ""
        self.is_deceptive = False
        self.expected_action = ""
        
        self.prompt_start_time = 0
        self.time_limit = 4.0  # Generous time limit for reliable execution human pacing

        self.baseline_y = None
        self.y_history = []
        self.glitch_active = False
        self.glitch_intensity = 0
        
        # Debounce/Intent Check State Machines
        self.action_cooldown = 0.0
        self.last_detected_action = None

    def _synthesize_tone(self, frequency, duration, type='sine'):
        sample_rate = 22050
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        if type == 'sine':
            data = np.sin(2 * np.pi * frequency * t)
        elif type == 'sawtooth':
            data = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        else:
            data = np.random.uniform(-1, 1, n_samples)
            data *= np.sign(np.sin(2 * np.pi * 50 * t))

        data = (data * 32767).astype(np.int16)
        stereo_data = np.column_stack((data, data))
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_data))

    def evaluate_gestures(self, frame_rgb, width, height):
        pose_res = self.pose.process(frame_rgb)
        hand_res = self.hands.process(frame_rgb)

        detected_actions = []
        thumbs_up_detected = False

        if hand_res.multi_hand_landmarks:
            for idx, hand_lms in enumerate(hand_res.multi_hand_landmarks):
                thumb_tip = hand_lms.landmark[4]
                thumb_ip = hand_lms.landmark[3]
                wrist = hand_lms.landmark[0]

                # Highly stable Thumbs Up parser check
                is_thumb_up = thumb_tip.y < thumb_ip.y
                is_fingers_curled = True
                for tip_id in [8, 12, 16, 20]:
                    if hand_lms.landmark[tip_id].y < hand_lms.landmark[tip_id - 2].y:
                        is_fingers_curled = False
                
                if is_thumb_up and is_fingers_curled:
                    thumbs_up_detected = True

                # Normalized tracking boundaries
                cx = int(wrist.x * width)
                if cx < width * 0.35:
                    detected_actions.append("LEFT")
                elif cx > width * 0.65:
                    detected_actions.append("RIGHT")

        if pose_res.pose_landmarks:
            l_shoulder = pose_res.pose_landmarks.landmark[11]
            r_shoulder = pose_res.pose_landmarks.landmark[12]
            mid_y = (l_shoulder.y + r_shoulder.y) / 2.0

            if self.baseline_y is None:
                self.baseline_y = mid_y

            self.y_history.append(mid_y)
            if len(self.y_history) > 15:
                self.y_history.pop(0)

            if len(self.y_history) >= 8:
                # Smooth continuous check instead of twitch metrics
                avg_y = sum(self.y_history[:-1]) / len(self.y_history[:-1])
                
                if mid_y > self.baseline_y + 0.08:
                    detected_actions.append("DUCK")
                
                delta_y = self.y_history[-1] - self.y_history[0]
                if delta_y < -0.04:
                    detected_actions.append("JUMP")

        return thumbs_up_detected, detected_actions

    def apply_cyber_glitch(self, frame, intensity):
        if intensity <= 0:
            return frame
        h, w, c = frame.shape
        glitched = frame.copy()
        
        for _ in range(int(intensity)):
            y1 = random.randint(0, h - 20)
            h_slice = random.randint(5, 20)
            shift = random.randint(-15, 15)
            glitched[y1:y1+h_slice, :, 0] = np.roll(glitched[y1:y1+h_slice, :, 0], shift, axis=0)
            glitched[y1:y1+h_slice, :, 2] = np.roll(glitched[y1:y1+h_slice, :, 2], -shift, axis=0)
        return glitched

    def execute(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow("DECEPTION MATRIX ENGINE", cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_time = time.time()

            thumbs_up, inputs = self.evaluate_gestures(frame_rgb, w, h)

            if self.glitch_active:
                frame = self.apply_cyber_glitch(frame, self.glitch_intensity)
                if current_time - self.prompt_start_time > 0.4:
                    self.glitch_active = False

            # Screen HUD guidance lines
            cv2.rectangle(frame, (0, 0), (int(w * 0.35), h), (255, 255, 255), 1)
            cv2.rectangle(frame, (int(w * 0.65), 0), (w, h), (255, 255, 255), 1)

            if self.state == "START":
                self.score = 0
                self.time_limit = 4.0
                self.baseline_y = None  # Force recalibration on every engine reset
                
                cv2.putText(frame, "SYSTEM INTEGRITY COMPROMISED", (w // 4, h // 3), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)
                cv2.putText(frame, "HOLD THUMBS UP TO INITIALIZE", (w // 4, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                if thumbs_up:
                    self.sound_correct.play()
                    self.state = "NEXT_PROMPT"
                    self.action_cooldown = current_time + 1.0  # Safe window transition lock

            elif self.state == "NEXT_PROMPT":
                self.current_prompt = random.choice(self.prompts)
                self.last_detected_action = None
                
                if random.random() < self.deception_rate:
                    self.is_deceptive = True
                    if self.current_prompt == "LEFT": self.expected_action = "RIGHT"
                    elif self.current_prompt == "RIGHT": self.expected_action = "LEFT"
                    elif self.current_prompt == "DUCK": self.expected_action = "JUMP"
                    elif self.current_prompt == "JUMP": self.expected_action = "DUCK"
                    
                    self.glitch_active = True
                    self.glitch_intensity = 15
                    self.sound_glitch.play()
                else:
                    self.is_deceptive = False
                    self.expected_action = self.current_prompt

                self.prompt_start_time = time.time()
                self.state = "WAITING_FOR_INPUT"

            elif self.state == "WAITING_FOR_INPUT":
                elapsed = current_time - self.prompt_start_time
                time_remaining = max(0.0, self.time_limit - elapsed)

                # Command Banner Text Positioning Overlays
                color = (0, 0, 255) if (self.is_deceptive and random.random() < 0.2) else (0, 255, 255)
                cv2.putText(frame, f"COMMAND: {self.current_prompt}", (w // 3, h // 2), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
                
                # Dynamic visual notification when a lie is injected
                if self.is_deceptive and elapsed < 1.5:
                    cv2.putText(frame, "(!)", (w // 2 - 30, h // 3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                bar_width = int((time_remaining / self.time_limit) * (w // 2))
                cv2.rectangle(frame, (w // 4, int(h * 0.6)), (w // 4 + bar_width, int(h * 0.62)), (0, 255, 0), -1)

                # Safe execution window bypass
                if current_time > self.action_cooldown and len(inputs) > 0:
                    # Look for intentional gesture locks
                    self.last_detected_action = inputs[0]

                # Evaluate performance loop criteria only *after* action frame window ends or shifts
                if time_remaining <= 0:
                    if self.last_detected_action == self.expected_action:
                        self.score += 1
                        self.high_score = max(self.score, self.high_score)
                        self.sound_correct.play()
                        self.action_cooldown = current_time + 0.8
                        self.state = "NEXT_PROMPT"
                    else:
                        self.sound_wrong.play()
                        self.state = "FAILED"

            elif self.state == "FAILED":
                cv2.putText(frame, "DECEPTION COMPLETE. YOU FAILED.", (w // 4, h // 3), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)
                cv2.putText(frame, f"FINAL SCORE: {self.score} | BEST: {self.high_score}", (w // 3, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "RAISE THUMBS UP TO REBOOT MATRIX", (w // 4, int(h * 0.65)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                if thumbs_up:
                    self.sound_correct.play()
                    self.state = "START"

            cv2.putText(frame, f"SCORE: {self.score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("DECEPTION MATRIX ENGINE", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    engine = GlitchDeceptionEngine()
    engine.execute()