import cv2
import mediapipe as mp
import numpy as np
import time
import random
import multiprocessing
from enum import Enum, auto

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class GameState(Enum):
    STANDBY = auto()
    PLAYING = auto()
    CLUTCH_SUCCESS = auto()
    GAME_OVER = auto()

class ActionType(Enum):
    BLOCK = "BLOCK (Raise Hand)"
    DUCK = "DUCK (Move Down)"
    CATCH = "CATCH (Close Fist)"

def generate_procedural_sound(frequency, duration, sound_type="sine"):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    if sound_type == "sine":
        data = np.sin(2 * np.pi * frequency * t)
    elif sound_type == "square":
        data = np.sign(np.sin(2 * np.pi * frequency * t))
    elif sound_type == "noise":
        data = np.random.uniform(-1, 1, n_samples)
        decay = np.exp(-4 * t)
        data = data * decay
        
    audio_buffer = np.int16(data * 32767)
    stereo_buffer = np.column_stack((audio_buffer, audio_buffer))
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_buffer))

def tracking_worker(frame_queue, coord_queue, stop_event):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    )
    
    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.001)
            continue
            
        frame = frame_queue.get()
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            landmarks_dict = {i: (int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)}
            
            while not coord_queue.empty():
                try:
                    coord_queue.get_nowait()
                except:
                    pass
            coord_queue.put(landmarks_dict)
        else:
            while not coord_queue.empty():
                try:
                    coord_queue.get_nowait()
                except:
                    pass
            coord_queue.put(None)
            
    hands.close()

class ReflexEngine:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        pygame.mixer.set_num_channels(8)
        
        self.snd_start = generate_procedural_sound(587.33, 0.25, "sine")     
        self.snd_success = generate_procedural_sound(880.00, 0.12, "sine")   
        self.snd_clutch = generate_procedural_sound(1200.00, 0.40, "square") 
        self.snd_fail = generate_procedural_sound(120.00, 0.60, "noise")     
        
        self.state = GameState.STANDBY
        self.score = 0
        
        # Human Calibration Tuning parameters
        self.reaction_window = 1.20   # Generous baseline starting time window
        self.min_window = 0.42       # Safe floor to maintain playability limits
        
        self.current_action = None
        self.event_start_time = 0.0
        self.event_active = False
        self.next_event_time = time.time() + 1.5
        
        self.clutch_freeze_until = 0.0
        self.clutch_frame = None
        
    def parse_kinematics(self, landmarks, frame_shape):
        if not landmarks:
            return "NO_HAND_DETECTED", False, False
            
        h, w = frame_shape[0], frame_shape[1]
        wrist = landmarks[0]
        
        extended_fingers = 0
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        for tip, pip in zip(tips, pips):
            if landmarks[tip][1] < landmarks[pip][1]:
                extended_fingers += 1
                
        is_open = extended_fingers >= 3
        is_fist = extended_fingers == 0
        
        is_high = wrist[1] < h * 0.45
        is_low = wrist[1] > h * 0.55
        
        if is_fist:
            return ActionType.CATCH, True, False
        if is_open and is_high:
            return ActionType.BLOCK, True, True
        if is_open and is_low:
            return ActionType.DUCK, True, True
        if is_open:
            return "OPEN_PALM", True, True
            
        return "UNCERTAIN_GESTURE", False, False

    def execute(self):
        print("[SYS] Initializing Adaptive Reflex Calibration Matrix...")
        cv2.namedWindow("Sayyam AI Lab: Overdrive-Reflex", cv2.WINDOW_NORMAL)
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_queue = multiprocessing.Queue()
        coord_queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        
        worker_process = multiprocessing.Process(
            target=tracking_worker, 
            args=(frame_queue, coord_queue, stop_event)
        )
        worker_process.start()
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                    
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                t_now = time.time()
                
                if frame_queue.empty():
                    frame_queue.put(frame.copy())
                    
                landmarks = None
                if not coord_queue.empty():
                    landmarks = coord_queue.get()
                
                detected_state, valid_gesture, absolute_open_palm = self.parse_kinematics(landmarks, (h, w))
                
                display_frame = frame.copy()
                overlay = display_frame.copy()
                
                if self.state == GameState.CLUTCH_SUCCESS:
                    if t_now < self.clutch_freeze_until:
                        display_frame = self.clutch_frame.copy()
                        cv2.imshow("Sayyam AI Lab: Overdrive-Reflex", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue
                    else:
                        self.state = GameState.PLAYING
                        self.next_event_time = t_now + random.uniform(1.2, 2.0)
                
                if self.state == GameState.STANDBY:
                    cv2.rectangle(overlay, (0, 0), (w, h), (25, 12, 12), -1)
                    cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)
                    
                    cv2.putText(display_frame, "ADAPTIVE CALIBRATION STANDBY", (w // 5, h // 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, cv2.LINE_AA)
                    cv2.putText(display_frame, "SHOW OPEN PALM TO CAM TO START PLAYABLE CALIBRATION LOOP", (w // 6, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    if valid_gesture and absolute_open_palm:
                        self.snd_start.play()
                        self.state = GameState.PLAYING
                        self.score = 0
                        self.reaction_window = 1.20   # Reset to playable timing constraints
                        self.event_active = False
                        self.next_event_time = t_now + 1.2
                        
                elif self.state == GameState.PLAYING:
                    if not self.event_active and t_now >= self.next_event_time:
                        self.current_action = random.choice(list(ActionType))
                        self.event_start_time = t_now
                        self.event_active = True
                        
                    if self.event_active:
                        elapsed = t_now - self.event_start_time
                        time_left = self.reaction_window - elapsed
                        
                        if time_left <= 0:
                            self.snd_fail.play()
                            self.state = GameState.GAME_OVER
                            self.event_active = False
                        else:
                            bar_width = int((time_left / self.reaction_window) * w)
                            pct = time_left / self.reaction_window
                            color = (int(0 + (1 - pct) * 255), int(255 * pct), 0)
                            cv2.rectangle(display_frame, (0, 0), (bar_width, 20), color, -1)
                            
                            cv2.putText(display_frame, f"ACTION THREAT: {self.current_action.value}", (40, 120),
                                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
                            
                            if valid_gesture and detected_state == self.current_action:
                                self.event_active = False
                                
                                # Clutch triggers relative to your dynamic window metrics (<15% time left)
                                if time_left < (self.reaction_window * 0.15):  
                                    self.snd_clutch.play()
                                    self.score += 25
                                    self.state = GameState.CLUTCH_SUCCESS
                                    self.clutch_freeze_until = t_now + 0.65
                                    
                                    rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 1.18)
                                    self.clutch_frame = cv2.warpAffine(display_frame, rot_mat, (w, h))
                                    
                                    flash = np.zeros_like(self.clutch_frame)
                                    flash[:] = (0, 255, 255) 
                                    cv2.addWeighted(self.clutch_frame, 0.65, flash, 0.35, 0, self.clutch_frame)
                                    cv2.putText(self.clutch_frame, "CLUTCH DEFIANCE SAVE! +25", (w // 4, h // 2),
                                                cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
                                else:
                                    self.snd_success.play()
                                    self.score += 10
                                    # Safe, non-linear progression steps to ensure smooth scaling
                                    self.reaction_window = max(self.min_window, self.reaction_window - 0.025)
                                    self.next_event_time = t_now + random.uniform(1.0, 1.8)
                                    
                    cv2.putText(display_frame, f"SCORE: {self.score}", (40, h - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, f"TIME WINDOW: {self.reaction_window:.3f}s", (w - 360, h - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2, cv2.LINE_AA)
                                
                elif self.state == GameState.GAME_OVER:
                    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 45), -1)
                    cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)
                    
                    cv2.putText(display_frame, "ROUND COMPLETED", (w // 3, h // 3),
                                cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(display_frame, f"FINAL REFLEX EVALUATION: {self.score}", (w // 4, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, f"SHOW OPEN PALM TO REBOOT THE MATRIX", (w // 4, int(h * 0.65)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    if valid_gesture and absolute_open_palm:
                        self.snd_start.play()
                        self.state = GameState.STANDBY
                        time.sleep(0.5)
                
                # Active telemetry diagnostic console layer
                hud_color = (0, 255, 0) if valid_gesture else (0, 165, 255)
                cv2.rectangle(display_frame, (15, 15), (450, 65), (20, 20, 20), -1)
                cv2.rectangle(display_frame, (15, 15), (450, 65), hud_color, 1)
                
                text_string = f"CONSOLE TELEMETRY: {str(detected_state)}"
                cv2.putText(display_frame, text_string, (30, 47),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 1, cv2.LINE_AA)
                
                if landmarks:
                    cv2.circle(display_frame, landmarks[0], 10, (255, 0, 255), -1)
                    for idx in [4, 8, 12, 16, 20]:
                        cv2.circle(display_frame, landmarks[idx], 6, (0, 255, 255), -1)
                        
                cv2.imshow("Sayyam AI Lab: Overdrive-Reflex", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            stop_event.set()
            worker_process.join()
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    engine = ReflexEngine()
    engine.execute()