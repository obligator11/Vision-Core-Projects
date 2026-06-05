"""
SAYYAM AI LAB — PROJECT KINESIS OVERDRIVE V5.4 (DEFENSIVE MASTER BUILD)
========================================================================
Requirements:
    pip install opencv-python "mediapipe==0.10.13" numpy sounddevice

Controls:
    ESC  —  Quit program cleanly at any time

Engine Patches:
    - Zero-Dimension Safe Guard: Forces hard limits if cap.get metadata returns 0.
    - Robust Face-Independent Palm Recalibration: Bypasses nose landmark dependencies.
    - Decoupled Concurrent Workers: Mitigates lock contention in async sensor thread.
"""

import os
import cv2
import mpmath
import mediapipe as mp
import numpy as np
import time
import math
import random
import threading
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ─────────────────────────────────────────────────────────────
# HIGH-FREQUENCY PROCEDURAL AUDIO ENGINE
# ─────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    SD_SR = 44100
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False
    print("[WARN] Audio hardware link unverified. FALLBACK internal matrix routing active.")

def _play_async(samples: np.ndarray):
    if not AUDIO_OK: return
    threading.Thread(target=lambda: sd.play(samples, SD_SR, blocking=False), daemon=True).start()

def _synth_tone(f0, f1, dur, wave_type='saw', vol=0.45) -> np.ndarray:
    n = int(SD_SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    freqs = np.linspace(f0, f1, n)
    phase = np.cumsum(freqs / SD_SR) * 2 * np.pi
    
    if wave_type == 'saw': sig = (phase % (2 * np.pi)) / np.pi - 1.0
    elif wave_type == 'sqr': sig = np.sign(np.sin(phase))
    else: sig = np.sin(phase)
        
    env = np.exp(-4.5 * t / dur)
    mono_wave = sig * env * vol
    return np.column_stack((mono_wave, mono_wave)).astype(np.float32)

_SND = {}
if AUDIO_OK:
    _SND['hit']      = _synth_tone(160, 60, 0.12, 'saw', 0.50)
    _SND['rep_beep'] = _synth_tone(523, 784, 0.08, 'sine', 0.35)
    _SND['warning']  = _synth_tone(180, 120, 0.25, 'sqr', 0.40)
    _SND['angry']    = _synth_tone(130, 85, 0.45, 'sqr', 0.60)
    _SND['success']  = _synth_tone(400, 650, 0.20, 'sine', 0.40)
    _SND['voice_cmd']= _synth_tone(240, 270, 0.18, 'saw', 0.45)

def play_sfx(key: str):
    if key in _SND: _play_async(_SND[key])

# ─────────────────────────────────────────────────────────────
# MATHEMATICAL TRIGONOMETRIC BIOMECHANICS FUNCTIONS
# ─────────────────────────────────────────────────────────────
def calculate_joint_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# ─────────────────────────────────────────────────────────────
# STRUCTURAL TARGET MITTS COMPONENT
# ─────────────────────────────────────────────────────────────
class TargetPadMitts:
    def __init__(self, punch_type, side='left'):
        self.punch_type = punch_type
        self.side = side
        self.x, self.y = 0, 0
        self.radius = 45
        self.phase = 0.0

    def update_position(self, W, H, landmarks):
        self.phase += 0.18
        ls, rs = landmarks[11], landmarks[12]
        
        ls_x, ls_y = int(ls.x * W), int(ls.y * H)
        rs_x, rs_y = int(rs.x * W), int(rs.y * H)
        mid_x = (ls_x + rs_x) // 2
        mid_y = (ls_y + rs_y) // 2
        
        shoulder_span = max(60, int(math.hypot(rs_x - ls_x, rs_y - ls_y)))
        
        if self.punch_type == "JAB":
            self.x = mid_x - int(shoulder_span * 0.9)
            self.y = mid_y - int(shoulder_span * 0.2)
        elif self.punch_type == "CROSS":
            self.x = mid_x + int(shoulder_span * 0.9)
            self.y = mid_y - int(shoulder_span * 0.2)
        elif self.punch_type == "HOOK":
            self.x = rs_x + int(shoulder_span * 0.75) if self.side == 'right' else ls_x - int(shoulder_span * 0.75)
            self.y = mid_y - int(shoulder_span * 0.4)
        elif self.punch_type == "UPPERCUT":
            self.x = mid_x + int(shoulder_span * (0.35 if self.side == 'right' else -0.35))
            self.y = mid_y - int(shoulder_span * 0.6)

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        glow_r = self.radius + int(math.sin(self.phase) * 6)
        
        cv2.circle(frame, (cx, cy), glow_r, (0, 238, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), self.radius, (0, 140, 255), -1)
        cv2.circle(frame, (cx, cy), self.radius - 6, (15, 15, 15), -1)
        cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)
        
        cv2.putText(frame, self.punch_type, (cx - 24, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────
# DECOUPLED ASYNC BACKGROUND SENSOR PIPELINE
# ─────────────────────────────────────────────────────────────
class VisionSensorThread:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_buffer = None
        self.pose_landmarks = None
        self.hand_landmarks = None
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)

    def start(self): self.thread.start()

    def update_frame(self, frame):
        with self.lock: self.frame_buffer = frame.copy()

    def get_landmarks(self):
        with self.lock: return self.pose_landmarks, self.hand_landmarks

    def _process_loop(self):
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        with mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55) as pose, \
             mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.55) as hands:
            while self.running:
                local_frame = None
                with self.lock:
                    if self.frame_buffer is not None:
                        local_frame = self.frame_buffer.copy()
                        self.frame_buffer = None
                if local_frame is None:
                    time.sleep(0.005)
                    continue
                rgb = cv2.cvtColor(local_frame, cv2.COLOR_BGR2RGB)
                pose_res = pose.process(rgb)
                hands_res = hands.process(rgb)
                with self.lock:
                    self.pose_landmarks = pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None
                    self.hand_landmarks = []
                    if hands_res.multi_hand_landmarks:
                        for hl, hd in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                            self.hand_landmarks.append({'side': hd.classification[0].label, 'landmarks': hl.landmark})

# ─────────────────────────────────────────────────────────────
# MAIN GAME MACHINE ENGINE CORE
# ─────────────────────────────────────────────────────────────
class SayyamBoxingCoachV5:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Defensive initialization pass to prevent frame-zero sizing crashes
        raw_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.W = raw_w if raw_w > 100 else 640
        self.H = raw_h if raw_h > 100 else 480
        
        self.sensor = VisionSensorThread()
        self.sensor.start()

        self.state = "TRAINING"  
        self.current_target_punch = "JAB"
        self.mitt_pad = TargetPadMitts("JAB", "left")
        
        self.score = 0
        self.pro_points = 0
        self.warnings_count = 0
        self.max_warnings = 2
        
        self.timer_checkpoint = time.time()
        self.last_audio_tick = 0.0
        self.punish_mode_type = "PUSHUP"
        self.required_reps = 3
        self.completed_reps = 0
        self.kinematic_vector_flag = "DOWN"
        
        self._BONES = [(11,12), (11,13), (13,15), (12,14), (14,16), (11,23), (12,24), (23,24), (23,25), (24,26), (25,27), (26,28)]

    def _draw_dashboard_hud(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, 85), (12, 10, 10), -1)
        cv2.rectangle(overlay, (0, self.H - 45), (self.W, self.H), (12, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        def blit_text(txt, x, y, size, color, thick=2):
            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0,0,0), thick+3, cv2.LINE_AA)
            cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

        blit_text(f"SCORE: {self.score}", 20, 34, 0.75, (255, 255, 255))
        blit_text(f"PRO XP: {self.pro_points}", 20, 64, 0.55, (0, 238, 255))
        
        warn_text = f"WARNINGS: {self.warnings_count}/{self.max_warnings}"
        warn_color = (0, 140, 255) if self.warnings_count == 1 else (0, 0, 255) if self.warnings_count == 2 else (255, 255, 255)
        blit_text(warn_text, 240, 34, 0.6, warn_color)

        state_color = (120, 255, 120) if self.state == "TRAINING" else (80, 80, 255) if self.state == "PUNISHMENT" else (255, 180, 0)
        blit_text(f"COACH: {self.state}", self.W - 300, 34, 0.7, state_color)

        if self.state == "TRAINING":
            blit_text(f"COMMAND: THROW {self.current_target_punch}!", self.W - 300, 64, 0.55, (0, 215, 255))
            cv2.rectangle(frame, (self.W//2 - 140, 120), (self.W//2 + 140, self.H - 80), (200,200,200), 1, cv2.LINE_AA)
        elif self.state == "PUNISHMENT":
            blit_text(f"TASK: DO {self.punish_mode_type} ({self.completed_reps}/{self.required_reps})", self.W - 320, 64, 0.52, (80, 80, 255))
        elif self.state == "PALM_CHECK":
            blit_text("HOLD WRISTS HIGHER THAN SHOULDERS TO RESUME", self.W - 390, 64, 0.48, (255, 215, 0))

    def _trigger_infraction(self, now):
        self.warnings_count += 1
        self.timer_checkpoint = now 
        if self.warnings_count > self.max_warnings:
            play_sfx('angry')
            self.state = "PUNISHMENT"
            self.punish_mode_type = random.choice(["PUSHUP", "SQUAT"])
            self.completed_reps = 0
            self.kinematic_vector_flag = "DOWN"
            self.warnings_count = 0
        else:
            play_sfx('warning')

    def _eval_training_logic(self, landmarks, now):
        self.mitt_pad.update_position(self.W, self.H, landmarks)
        
        if now - self.last_audio_tick > 3.5:
            play_sfx('voice_cmd')
            self.last_audio_tick = now

        if now - self.timer_checkpoint > 10.0:
            self._trigger_infraction(now)
            return

        for side_prefix, wrist_index in [('left', 15), ('right', 16)]:
            wrist = landmarks[wrist_index]
            if wrist.visibility < 0.45: continue
            wx, wy = int(wrist.x * self.W), int(wrist.y * self.H)
            
            if math.hypot(wx - self.mitt_pad.x, wy - self.mitt_pad.y) <= self.mitt_pad.radius + 28:
                sh_i, el_i, wr_i = (11, 13, 15) if side_prefix == 'left' else (12, 14, 16)
                angle = calculate_joint_angle(
                    [landmarks[sh_i].x, landmarks[sh_i].y],
                    [landmarks[el_i].x, landmarks[el_i].y],
                    [landmarks[wr_i].x, landmarks[wr_i].y]
                )
                
                if self.current_target_punch in ["JAB", "CROSS"] and angle < 115:
                    self._trigger_infraction(now)
                    return

                play_sfx('hit')
                self.score += 150
                self.pro_points += 25
                
                punches_pool = ["JAB", "CROSS", "HOOK", "UPPERCUT"]
                self.current_target_punch = random.choice(punches_pool)
                self.mitt_pad = TargetPadMitts(self.current_target_punch, random.choice(['left', 'right']))
                self.timer_checkpoint = now
                break

    def _eval_punishment_logic(self, landmarks):
        if self.punish_mode_type == "PUSHUP":
            sh_l, el_l, wr_l = landmarks[11], landmarks[13], landmarks[15]
            sh_r, el_r, wr_r = landmarks[12], landmarks[14], landmarks[16]
            
            if el_l.visibility > 0.35 or el_r.visibility > 0.35:
                ang_l = calculate_joint_angle([sh_l.x, sh_l.y], [el_l.x, el_l.y], [wr_l.x, wr_l.y])
                ang_r = calculate_joint_angle([sh_r.x, sh_r.y], [el_r.x, el_r.y], [wr_r.x, wr_r.y])
                avg_angle = (ang_l + ang_r) / 2.0
                
                if avg_angle < 105 and self.kinematic_vector_flag == "DOWN":
                    self.kinematic_vector_flag = "UP"
                elif avg_angle > 140 and self.kinematic_vector_flag == "UP":
                    self.completed_reps += 1
                    play_sfx('rep_beep')
                    self.kinematic_vector_flag = "DOWN"

        elif self.punish_mode_type == "SQUAT":
            hip, knee, ank = landmarks[24], landmarks[26], landmarks[28]
            if knee.visibility > 0.35:
                angle = calculate_joint_angle([hip.x, hip.y], [knee.x, knee.y], [ank.x, ank.y])
                if angle < 110 and self.kinematic_vector_flag == "DOWN":
                    self.kinematic_vector_flag = "UP"
                elif angle > 150 and self.kinematic_vector_flag == "UP":
                    self.completed_reps += 1
                    play_sfx('rep_beep')
                    self.kinematic_vector_flag = "DOWN"

        if self.completed_reps >= self.required_reps:
            play_sfx('success')
            self.state = "PALM_CHECK"

    def execute_loop(self):
        cv2.namedWindow("Sayyam AI Boxing Coach", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sayyam AI Boxing Coach", self.W, self.H)

        while self.cap.isOpened():
            now_time = time.time()
            ok, raw_frame = self.cap.read()
            if not ok: continue

            # Update dynamic scale factors safely if hardware modifications occur mid-stream
            fh, fw, _ = raw_frame.shape
            if fw != self.W or fh != self.H:
                self.W, self.H = fw, fh

            frame = cv2.flip(raw_frame, 1)
            self.sensor.update_frame(frame)
            pose_lms, _ = self.sensor.get_landmarks()

            if pose_lms:
                for bone in self._BONES:
                    p1, p2 = pose_lms[bone[0]], pose_lms[bone[1]]
                    if p1.visibility > 0.4 and p2.visibility > 0.4:
                        cv2.line(frame, (int(p1.x*self.W), int(p1.y*self.H)), (int(p2.x*self.W), int(p2.y*self.H)), (0,0,0), 4, cv2.LINE_AA)
                        cv2.line(frame, (int(p1.x*self.W), int(p1.y*self.H)), (int(p2.x*self.W), int(p2.y*self.H)), (0, 255, 130), 2, cv2.LINE_AA)
                
                if self.state == "TRAINING":
                    self._eval_training_logic(pose_lms, now_time)
                elif self.state == "PUNISHMENT":
                    self._eval_punishment_logic(pose_lms)

                # Robust multi-node recovery alignment to handle sudden perspective transitions gracefully
                if self.state == "PALM_CHECK":
                    lw_wrist, rw_wrist = pose_lms[15], pose_lms[16]
                    l_shoulder, r_shoulder = pose_lms[11], pose_lms[12]
                    
                    if (lw_wrist.visibility > 0.4 and rw_wrist.visibility > 0.4 and 
                        lw_wrist.y < l_shoulder.y and rw_wrist.y < r_shoulder.y):
                        self.state = "TRAINING"
                        self.timer_checkpoint = now_time
                        self.current_target_punch = "JAB"
                        self.mitt_pad = TargetPadMitts("JAB", "left")

            if self.state == "TRAINING" and pose_lms:
                self.mitt_pad.draw(frame)

            self._draw_dashboard_hud(frame)
            cv2.imshow("Sayyam AI Boxing Coach", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        self.sensor.running = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    coach_engine = SayyamBoxingCoachV5()
    coach_engine.execute_loop()