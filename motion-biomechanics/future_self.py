import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import threading
from collections import deque

# -------------------------------------------------------------------------
# CONSTANTS & SETUP
# -------------------------------------------------------------------------
WINDOW_TITLE = "Your Future Self Reacts"
INITIAL_WIDTH, INITIAL_HEIGHT = 1280, 720
CAMERA_INDEX = 0

TRAIL_LENGTH = 15
PREDICTION_STEPS = 12  # Clean temporal prediction gap balance

# Face-stripped clear body configuration connections matrix
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Upper Torso / Arms
    (11, 23), (12, 24), (23, 24),                     # Center Spine Bounds
    (23, 25), (24, 26), (25, 27), (26, 28)            # Lower Extremities
]

# Focus the game engine comparisons on active limbs to simplify play
CORE_GAME_JOINTS = [13, 14, 15, 16]

# -------------------------------------------------------------------------
# ASYNCHRONOUS THREAD-ISOLATED AUDIO SYNTHESIS
# -------------------------------------------------------------------------
class ProceduralSoundManager:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sample_rate = 44100
        self.sound_queue = deque(maxlen=5)
        self._running = True
        
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        
        self.ambient_sound = self._synthesize_ambient()
        self.ambient_sound.play(loops=-1)

    def _generate_wave(self, freq, duration, type='sine', volume=0.2):
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        if type == 'sine':
            data = np.sin(2 * np.pi * freq * t)
        elif type == 'square':
            data = np.sign(np.sin(2 * np.pi * freq * t))
        else:
            data = np.sin(2 * np.pi * freq * t)

        envelope = np.exp(-4.0 * np.linspace(0, 1, num_samples))
        data = data * envelope * volume
        
        stereo_data = np.zeros((num_samples, 2), dtype=np.int16)
        signal_16bit = np.int16(data * 32767)
        stereo_data[:, 0] = signal_16bit
        stereo_data[:, 1] = signal_16bit
        return pygame.sndarray.make_sound(stereo_data)

    def _synthesize_ambient(self):
        duration = 4.0
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        wave = 0.2 * np.sin(2 * np.pi * 55 * t)
        envelope = np.sin(np.pi * np.linspace(0, 1, num_samples))
        wave = wave * envelope * 0.1
        
        stereo_data = np.zeros((num_samples, 2), dtype=np.int16)
        signal_16bit = np.int16(wave * 32767)
        stereo_data[:, 0] = signal_16bit
        stereo_data[:, 1] = signal_16bit
        return pygame.sndarray.make_sound(stereo_data)

    def _audio_worker(self):
        while self._running:
            if self.sound_queue:
                sound_func = self.sound_queue.popleft()
                try: sound_func()
                except: pass
            time.sleep(0.01)

    def trigger_prediction_change(self):
        def play():
            self._generate_wave(freq=440, duration=0.08, type='sine', volume=0.2).play()
        self.sound_queue.append(play)

    def trigger_success(self):
        def play():
            s1 = self._generate_wave(freq=523.25, duration=0.08, type='sine', volume=0.25)
            s2 = self._generate_wave(freq=659.25, duration=0.15, type='sine', volume=0.25)
            s1.play()
            time.sleep(0.05)
            s2.play()
        self.sound_queue.append(play)

    def trigger_fail(self):
        def play():
            self._generate_wave(freq=130, duration=0.25, type='square', volume=0.2).play()
        self.sound_queue.append(play)

    def stop(self):
        self._running = False
        self.ambient_sound.stop()
        pygame.mixer.quit()


# -------------------------------------------------------------------------
# THREAD-ISOLATED VIDEO CAPTURE ENGINE
# -------------------------------------------------------------------------
class ThreadedVideoStream:
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# -------------------------------------------------------------------------
# STABILIZED KINEMATIC MOTION PREDICTION LOGIC
# -------------------------------------------------------------------------
class KinematicMotionEngine:
    def __init__(self, history_len=TRAIL_LENGTH):
        self.history = {i: deque(maxlen=history_len) for i in range(33)}
        self.alpha = 0.3  # Clean moving average smoothing filter

    def update_history(self, landmarks):
        for i in range(33):
            lm = landmarks[i]
            current_pos = np.array([lm.x, lm.y, lm.z])
            if len(self.history[i]) > 0:
                prev_pos = self.history[i][-1]
                smoothed = self.alpha * current_pos + (1 - self.alpha) * prev_pos
                self.history[i].append(smoothed)
            else:
                self.history[i].append(current_pos)

    def predict_future_state(self, steps=PREDICTION_STEPS):
        predicted_joints = {}
        for i in range(33):
            pts = list(self.history[i])
            if len(pts) < 4:
                if len(pts) == 0: continue
                predicted_joints[i] = pts[-1]
                continue
            
            # Extract basic linear velocity profile
            v1 = pts[-1] - pts[-2]
            v2 = pts[-2] - pts[-3]
            avg_velocity = (v1 + v2) / 2.0
            
            future_pos = pts[-1] + (avg_velocity * steps)
            predicted_joints[i] = np.clip(future_pos, 0.0, 1.0)
            
        return predicted_joints


# -------------------------------------------------------------------------
# ADJUSTED GAMEPLAY CONFIGURATION ENGINE
# -------------------------------------------------------------------------
class GameStateEngine:
    def __init__(self):
        self.score = 0
        self.combo = 0
        self.state = "START"
        self.last_state_tick = time.time()
        self.prediction_lock_interval = 2.0  
        self.replay_buffer = deque(maxlen=60)
        self.replay_index = 0

    def evaluate_game_mechanics(self, current_live_joints, baseline_predicted_joints):
        if not baseline_predicted_joints or len(current_live_joints) == 0:
            return "MATCHED", 0.0
        
        accumulated_error = 0.0
        valid_counts = 0
        
        # Calculate coordinate distances specifically across your active hands and elbows
        for idx in CORE_GAME_JOINTS:
            if idx in current_live_joints and idx in baseline_predicted_joints:
                live = current_live_joints[idx][:2]
                pred = baseline_predicted_joints[idx][:2]
                dist = math.sqrt((live[0] - pred[0])**2 + (live[1] - pred[1])**2)
                accumulated_error += dist
                valid_counts += 1
                
        if valid_counts == 0:
            return "MATCHED", 0.0
            
        mean_deviation = accumulated_error / valid_counts
        
        # LOWERED DISPLACEMENT THRESHOLD: Adjusted down to 0.06 coordinate space points.
        # This means moving even a tiny bit clear of the pink wireframe awards you points!
        relaxed_success_threshold = 0.06
        
        # Diagnostic tracking message logged to terminal to aid live user feedback loop runs
        print(f"[ROUND RESULT] Your Shift Distance: {mean_deviation:.4f} vs Target Threshold: {relaxed_success_threshold:.4f}")
        
        if mean_deviation >= relaxed_success_threshold:
            return "TRICKED", mean_deviation
        else:
            return "MATCHED", mean_deviation


# -------------------------------------------------------------------------
# RENDERING LAYER HELPERS
# -------------------------------------------------------------------------
class PresentationRenderer:
    @staticmethod
    def draw_wireframe(img, joints, connections, color, thickness=3):
        h, w, _ = img.shape
        pixel_map = {}
        
        for idx, pos in joints.items():
            cx, cy = int(pos[0] * w), int(pos[1] * h)
            if 0 <= cx < w and 0 <= cy < h:
                pixel_map[idx] = (cx, cy)
                
        for conn in connections:
            if conn[0] in pixel_map and conn[1] in pixel_map:
                cv2.line(img, pixel_map[conn[0]], pixel_map[conn[1]], color, thickness, cv2.LINE_AA)
                
        for idx in CORE_GAME_JOINTS:
            if idx in pixel_map:
                cv2.circle(img, pixel_map[idx], thickness + 3, (255, 255, 255), -1)
                cv2.circle(img, pixel_map[idx], thickness, color, -1)


# -------------------------------------------------------------------------
# MAIN PROGRAM PIPELINE
# -------------------------------------------------------------------------
def main():
    sound_mgr = ProceduralSoundManager()
    stream = ThreadedVideoStream(CAMERA_INDEX)
    kinematics = KinematicMotionEngine()
    game = GameStateEngine()
    
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, INITIAL_WIDTH, INITIAL_HEIGHT)

    frozen_future_pose = None
    evaluation_text = "AI INITIALIZING TIMELINE"
    eval_color = (0, 255, 255)
    action_prompt = "PREPARING DIAGNOSTICS"

    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
            
        frame = cv2.flip(frame, 1)
        window_rect = cv2.getWindowImageRect(WINDOW_TITLE)
        canvas_w = window_rect[2] if window_rect[2] > 100 else INITIAL_WIDTH
        canvas_h = window_rect[3] if window_rect[3] > 100 else INITIAL_HEIGHT
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)
        
        current_live_joints = {}
        if results.pose_landmarks:
            kinematics.update_history(results.pose_landmarks.landmark)
            for i in range(33):
                lm = results.pose_landmarks.landmark[i]
                current_live_joints[i] = np.array([lm.x, lm.y, lm.z])

        now = time.time()
        time_spent = now - game.last_state_tick
        time_left = max(0.0, game.prediction_lock_interval - time_spent)

        if game.state == "START":
            display_img = cv2.resize(frame, (canvas_w, canvas_h))
            overlay = display_img.copy()
            cv2.rectangle(overlay, (50, canvas_h // 2 - 140), (canvas_w - 50, canvas_h // 2 + 100), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.85, display_img, 0.15, 0, display_img)
            
            cv2.putText(display_img, "YOUR FUTURE SELF REACTS", (canvas_w // 2 - 340, canvas_h // 2 - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (56, 189, 248), 4, cv2.LINE_AA)
            cv2.putText(display_img, "How to Play: Step back so the camera sees your full torso and arms.", (canvas_w // 2 - 440, canvas_h // 2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (226, 232, 240), 2, cv2.LINE_AA)
            cv2.putText(display_img, "When the Pink Ghost freezes, quickly throw your arms in a completely DIFFERENT direction!", (canvas_w // 2 - 440, canvas_h // 2 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 211, 238), 2, cv2.LINE_AA)
            cv2.putText(display_img, "Press [SPACEBAR] to initiate play loop.", (canvas_w // 2 - 200, canvas_h // 2 + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                game.state = "GAMEPLAY"
                game.last_state_tick = time.time()
                sound_mgr.trigger_prediction_change()
            elif key == ord('q'):
                break
            cv2.imshow(WINDOW_TITLE, display_img)
            continue

        elif game.state == "GAMEPLAY":
            display_img = cv2.resize(frame, (canvas_w, canvas_h))
            
            # Catch snapshot target vectors right at the boundary frame reset split-second
            if time_spent < 0.15:
                frozen_future_pose = kinematics.predict_future_state(steps=PREDICTION_STEPS)
                evaluation_text = "AI CLONE LOCKED PREDICTION!"
                eval_color = (244, 63, 94)
                action_prompt = "MOVE AWAY FROM PINK GHOST SKELETON!"

            # Render locked down prediction skeleton copy
            if frozen_future_pose:
                PresentationRenderer.draw_wireframe(
                    display_img, frozen_future_pose, POSE_CONNECTIONS, 
                    color=(236, 72, 153), thickness=4
                )
                
            # Process delta evaluations at the end of the step interval countdown budget
            if time_spent >= game.prediction_lock_interval:
                outcome, deviation = game.evaluate_game_mechanics(current_live_joints, frozen_future_pose)
                
                if outcome == "TRICKED":
                    game.score += 100
                    game.combo += 1
                    evaluation_text = "TIMELINE DISRUPTED! (+100)"
                    eval_color = (34, 197, 94)
                    sound_mgr.trigger_success()
                    game.state = "FAKEOUT_REPLAY"
                    game.replay_index = 0
                else:
                    game.combo = 0
                    evaluation_text = "CAUGHT BY CLONE PREDICTION!"
                    eval_color = (239, 68, 68)
                    sound_mgr.trigger_fail()
                
                game.last_state_tick = now
                sound_mgr.trigger_prediction_change()

            # Construct display glass-HUD panels
            cv2.rectangle(display_img, (0, 0), (canvas_w, 110), (15, 23, 42), -1)
            cv2.line(display_img, (0, 110), (canvas_w, 110), (56, 189, 248), 2)
            
            # Populate text labels blocks cleanly onto top alignment regions
            cv2.putText(display_img, f"SCORE: {game.score}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_img, f"COMBO: x{game.combo}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (147, 197, 253), 2, cv2.LINE_AA)
            cv2.putText(display_img, evaluation_text, (canvas_w // 2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, eval_color, 2, cv2.LINE_AA)
            cv2.putText(display_img, action_prompt, (canvas_w // 2 - 200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (226, 232, 240), 2, cv2.LINE_AA)
            
            # Sync timing progress visual status fields
            progress_bar_w = int((time_left / game.prediction_lock_interval) * 220)
            cv2.rectangle(display_img, (canvas_w - 250, 70), (canvas_w - 30, 83), (30, 41, 59), -1)
            if progress_bar_w > 0:
                cv2.rectangle(display_img, (canvas_w - 250, 70), (canvas_w - 250 + progress_bar_w, 83), (34, 211, 238), -1)
            cv2.putText(display_img, "NEXT AI CALIBRATION", (canvas_w - 250, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

            # Draw Mini-Cam Picture in Picture frame layout bounds containing tracking validation paths
            cam_overlay_w, cam_overlay_h = 160, 120
            small_cam = cv2.resize(frame, (cam_overlay_w, cam_overlay_h))
            if len(current_live_joints) > 0:
                PresentationRenderer.draw_wireframe(small_cam, current_live_joints, POSE_CONNECTIONS, (34, 197, 94), 2)
                
            ox, oy = canvas_w - cam_overlay_w - 20, 130
            display_img[oy:oy+cam_overlay_h, ox:ox+cam_overlay_w] = small_cam
            cv2.rectangle(display_img, (ox, oy), (ox+cam_overlay_w, oy+cam_overlay_h), (56, 189, 248), 2)

            game.replay_buffer.append(display_img.copy())

        elif game.state == "FAKEOUT_REPLAY":
            if len(game.replay_buffer) == 0:
                game.state = "GAMEPLAY"
                game.last_state_tick = time.time()
                continue
                
            display_img = game.replay_buffer[game.replay_index].copy()
            
            # Render clear bottom warning stripe to call out slow motion visual play sequences cleanly
            cv2.rectangle(display_img, (0, canvas_h - 70), (canvas_w, canvas_h), (124, 58, 237), -1)
            cv2.putText(display_img, "⚡ TIMELINE DISRUPTED: SLOW-MO REPLAY ⚡", 
                        (canvas_w // 2 - 280, canvas_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            
            game.replay_index += 1
            if game.replay_index >= len(game.replay_buffer):
                game.replay_buffer.clear()
                game.state = "GAMEPLAY"
                game.last_state_tick = time.time()

        cv2.imshow(WINDOW_TITLE, display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    sound_mgr.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()