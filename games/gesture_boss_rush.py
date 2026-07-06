

import cv2
import numpy as np
import mediapipe as mp
import pygame
import threading
import queue
import time
import random
import math
from enum import Enum

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720
POSE_MODEL_COMPLEXITY = 1
POSE_MIN_DET_CONF = 0.6
POSE_MIN_TRACK_CONF = 0.6

PLAYER_MAX_HP = 5
BOSS_MAX_HP = 12
IFRAME_SECONDS = 0.8

# Pose-state detection thresholds, expressed relative to the calibrated
# standing baseline so it self-adjusts to the player's body / distance.
ARMS_UP_MARGIN = 0.06     # wrists must be this much ABOVE shoulder line (norm y)
CROUCH_MARGIN = 0.05      # hips must drop this much BELOW baseline hip y
JUMP_VELOCITY_THRESHOLD = -0.55   # normalized-y/sec, negative = moving up fast

TELEGRAPH_SECONDS = 1.4     # how long the boss "winds up" before resolving
RESOLVE_WINDOW_SECONDS = 0.35  # tolerance window around the resolve instant
POST_ATTACK_PAUSE = 1.0

COUNTER_METER_MAX = 3
SPECIAL_DAMAGE = 3
NORMAL_COUNTER_DAMAGE = 1

SAMPLE_RATE = 44100

# Neon/danger color palette (BGR for OpenCV)
COL_BG = (18, 12, 8)
COL_SKELETON = (255, 220, 60)
COL_TEXT = (240, 240, 240)
COL_PLAYER_HP = (90, 255, 120)
COL_PLAYER_HP_LOST = (60, 60, 200)
COL_BOSS_HP = (255, 90, 90)
COL_BOSS_HP_LOST = (70, 70, 70)

TELEGRAPH_COLORS = {
    "SMASH": (60, 60, 255),   # red  -> needs BLOCK
    "SWEEP": (60, 220, 255),  # yellow -> needs DODGE
    "BEAM": (255, 160, 60),   # blue -> needs JUMP
}
REQUIRED_STATE = {
    "SMASH": "BLOCK",
    "SWEEP": "DODGE",
    "BEAM": "JUMP",
}


# ------------------------------------------------------------------------
# THREADED CAMERA GRABBER (identical pattern to Dodge Swarm — keeps the
# capture thread decoupled from the game/render loop)
# ------------------------------------------------------------------------
class FrameGrabber:
    def __init__(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.cap.release()


# ------------------------------------------------------------------------
# PROCEDURAL AUDIO
# ------------------------------------------------------------------------
def _tone(freq, duration, volume=0.5, wave="sine", decay=True):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    if wave == "sine":
        wav = np.sin(2 * np.pi * freq * t)
    elif wave == "square":
        wav = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        wav = np.random.uniform(-1, 1, t.shape)
    if decay:
        wav *= np.linspace(1, 0, t.shape[0])
    wav = (wav * volume * 32767).astype(np.int16)
    stereo = np.repeat(wav.reshape(-1, 1), 2, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


class Sfx:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        self.success = _tone(720, 0.18, 0.5, "sine")
        self.fail = _tone(110, 0.3, 0.7, "square")
        self.telegraph = _tone(300, 0.12, 0.25, "sine")
        self.special = _tone(950, 0.4, 0.6, "sine")
        self.win = _tone(880, 0.6, 0.6, "sine")
        self.lose = _tone(80, 1.0, 0.7, "square")

    def play(self, sound):
        sound.play()


# ------------------------------------------------------------------------
# POSE STATE MACHINE
# ------------------------------------------------------------------------
class PoseState(Enum):
    IDLE = "IDLE"
    BLOCK = "BLOCK"
    DODGE = "DODGE"
    JUMP = "JUMP"


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


class PoseClassifier:
    """
    Converts raw MediaPipe landmarks into a PoseState, using a calibrated
    baseline captured at game start (player standing neutrally).
    """

    def __init__(self):
        self.baseline_shoulder_y = None
        self.baseline_hip_y = None
        self.prev_hip_y = None
        self.prev_time = time.time()

    def calibrate(self, landmarks):
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        self.baseline_shoulder_y = (ls.y + rs.y) / 2.0
        self.baseline_hip_y = (lh.y + rh.y) / 2.0
        self.prev_hip_y = self.baseline_hip_y
        self.prev_time = time.time()

    def classify(self, landmarks):
        if self.baseline_shoulder_y is None:
            return PoseState.IDLE

        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        shoulder_y = (ls.y + rs.y) / 2.0
        hip_y = (lh.y + rh.y) / 2.0
        wrist_y = (lw.y + rw.y) / 2.0

        now = time.time()
        dt = max(1e-3, now - self.prev_time)
        hip_velocity = (hip_y - self.prev_hip_y) / dt  # negative = moving up
        self.prev_hip_y = hip_y
        self.prev_time = now

        # NOTE: image-space y grows DOWNWARD, so "above" means smaller y.
        arms_up = wrist_y < (shoulder_y - ARMS_UP_MARGIN)
        crouching = hip_y > (self.baseline_hip_y + CROUCH_MARGIN)
        jumping = hip_velocity < JUMP_VELOCITY_THRESHOLD

        # Priority: jump is a transient spike so check it first, then
        # arms/crouch which are held poses.
        if jumping:
            return PoseState.JUMP
        if arms_up:
            return PoseState.BLOCK
        if crouching:
            return PoseState.DODGE
        return PoseState.IDLE


# ------------------------------------------------------------------------
# BOSS
# ------------------------------------------------------------------------
class Boss:
    def __init__(self):
        self.hp = BOSS_MAX_HP
        self.attack_types = list(TELEGRAPH_COLORS.keys())
        self.current_attack = None
        self.phase_start = time.time()
        self.phase = "COOLDOWN"   # COOLDOWN -> TELEGRAPH -> RESOLVE -> COOLDOWN
        self.resolved_this_attack = False

    def start_telegraph(self):
        self.current_attack = random.choice(self.attack_types)
        self.phase = "TELEGRAPH"
        self.phase_start = time.time()
        self.resolved_this_attack = False

    def update(self, elapsed_in_phase):
        if self.phase == "COOLDOWN" and elapsed_in_phase > 0.9:
            self.start_telegraph()
        elif self.phase == "TELEGRAPH" and elapsed_in_phase > TELEGRAPH_SECONDS:
            self.phase = "RESOLVE"
            self.phase_start = time.time()
        elif self.phase == "RESOLVE" and elapsed_in_phase > RESOLVE_WINDOW_SECONDS:
            self.phase = "COOLDOWN"
            self.phase_start = time.time()
            self.current_attack = None


# ------------------------------------------------------------------------
# MAIN GAME
# ------------------------------------------------------------------------
class GestureBossRush:
    def __init__(self):
        self.grabber = FrameGrabber(CAM_INDEX)
        self.pose = mp_pose.Pose(
            model_complexity=POSE_MODEL_COMPLEXITY,
            min_detection_confidence=POSE_MIN_DET_CONF,
            min_tracking_confidence=POSE_MIN_TRACK_CONF,
        )
        self.classifier = PoseClassifier()
        self.sfx = Sfx()
        self.reset()

    def reset(self):
        self.player_hp = PLAYER_MAX_HP
        self.boss = Boss()
        self.counter_meter = 0
        self.last_hit_time = -999.0
        self.game_over = False
        self.won = False
        self.special_window_until = 0.0
        self.armed_for_special = False  # True after ARMS_UP seen, waiting for JUMP
        self.feedback_text = ""
        self.feedback_until = 0.0

    def flash_feedback(self, text, duration=0.9):
        self.feedback_text = text
        self.feedback_until = time.time() + duration

    def try_special(self, state):
        """Special combo: BLOCK held, then JUMP within 1s, only when meter is full."""
        if self.counter_meter < COUNTER_METER_MAX:
            return False
        if state == PoseState.BLOCK:
            self.armed_for_special = True
            self.special_window_until = time.time() + 1.0
            return False
        if state == PoseState.JUMP and self.armed_for_special and time.time() < self.special_window_until:
            self.boss.hp = max(0, self.boss.hp - SPECIAL_DAMAGE)
            self.counter_meter = 0
            self.armed_for_special = False
            self.sfx.play(self.sfx.special)
            self.flash_feedback("SPECIAL HIT!")
            return True
        if time.time() > self.special_window_until:
            self.armed_for_special = False
        return False

    def resolve_attack(self, state):
        attack = self.boss.current_attack
        required = REQUIRED_STATE[attack]
        now = time.time()
        invincible = (now - self.last_hit_time) < IFRAME_SECONDS

        if state.name == required:
            self.counter_meter = min(COUNTER_METER_MAX, self.counter_meter + 1)
            self.boss.hp = max(0, self.boss.hp - NORMAL_COUNTER_DAMAGE)
            self.sfx.play(self.sfx.success)
            self.flash_feedback(f"{required} — SUCCESS")
        else:
            if not invincible:
                self.player_hp -= 1
                self.last_hit_time = now
                self.sfx.play(self.sfx.fail)
                self.flash_feedback(f"HIT! needed {required}")
            if self.player_hp <= 0:
                self.game_over = True
                self.won = False
                self.sfx.play(self.sfx.lose)

        if self.boss.hp <= 0 and not self.game_over:
            self.game_over = True
            self.won = True
            self.sfx.play(self.sfx.win)

    def update(self, state):
        if self.game_over:
            return

        now = time.time()
        elapsed = now - self.boss.phase_start
        prev_phase = self.boss.phase
        self.boss.update(elapsed)

        if prev_phase == "TELEGRAPH" and self.boss.phase == "RESOLVE":
            self.sfx.play(self.sfx.telegraph)

        if self.boss.phase == "RESOLVE" and not self.boss.resolved_this_attack:
            self.boss.resolved_this_attack = True
            self.resolve_attack(state)

        self.try_special(state)

    # -------------------- rendering --------------------
    def draw_hp_bars(self, frame):
        for i in range(PLAYER_MAX_HP):
            color = COL_PLAYER_HP if i < self.player_hp else COL_PLAYER_HP_LOST
            cv2.circle(frame, (30 + i * 34, 30), 12, color, -1)
        cv2.putText(frame, "YOU", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 2)

        bar_w = 300
        boss_ratio = self.boss.hp / BOSS_MAX_HP
        x0 = FRAME_W - bar_w - 20
        cv2.rectangle(frame, (x0, 20), (x0 + bar_w, 45), COL_BOSS_HP_LOST, -1)
        cv2.rectangle(frame, (x0, 20), (x0 + int(bar_w * boss_ratio), 45), COL_BOSS_HP, -1)
        cv2.putText(frame, "BOSS", (x0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 2)

        meter_w = 150
        mx0 = FRAME_W // 2 - meter_w // 2
        ratio = self.counter_meter / COUNTER_METER_MAX
        cv2.rectangle(frame, (mx0, 20), (mx0 + meter_w, 40), (80, 80, 80), -1)
        cv2.rectangle(frame, (mx0, 20), (mx0 + int(meter_w * ratio), 40), (255, 220, 60), -1)
        label = "SPECIAL READY (BLOCK then JUMP)" if self.counter_meter >= COUNTER_METER_MAX else "COUNTER METER"
        cv2.putText(frame, label, (mx0 - 40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1)

    def draw_telegraph(self, frame):
        attack = self.boss.current_attack
        if not attack:
            return
        color = TELEGRAPH_COLORS[attack]
        required = REQUIRED_STATE[attack]

        if self.boss.phase == "TELEGRAPH":
            pulse = 0.5 + 0.5 * math.sin(time.time() * 10)
            thickness = int(6 + pulse * 6)
            cv2.rectangle(frame, (10, 10), (FRAME_W - 10, FRAME_H - 10), color, thickness)
            cv2.putText(frame, f"{attack} INCOMING", (FRAME_W // 2 - 160, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            cv2.putText(frame, f"PREPARE: {required}", (FRAME_W // 2 - 140, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_TEXT, 2)
        elif self.boss.phase == "RESOLVE":
            cv2.rectangle(frame, (0, 0), (FRAME_W, FRAME_H), color, -1)
            # note: this full-frame flash is drawn on an overlay and blended
            # in run(), not directly on the camera feed, to stay readable.

    def draw_state_label(self, frame, state):
        cv2.putText(frame, f"STATE: {state.value}", (20, FRAME_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_TEXT, 2)
        if self.feedback_text and time.time() < self.feedback_until:
            cv2.putText(frame, self.feedback_text, (FRAME_W // 2 - 180, FRAME_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    def draw_end_screen(self, frame):
        text = "VICTORY" if self.won else "DEFEATED"
        color = (90, 255, 120) if self.won else (60, 60, 255)
        cv2.putText(frame, text, (FRAME_W // 2 - 160, FRAME_H // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)
        cv2.putText(frame, "Press R to restart", (FRAME_W // 2 - 150, FRAME_H // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_TEXT, 2)

    def run(self):
        calibrating = True
        calib_start = time.time()

        while True:
            frame_bgr = self.grabber.read()
            frame_bgr = cv2.resize(frame_bgr, (FRAME_W, FRAME_H))
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)
            landmarks = result.pose_landmarks

            canvas = frame_bgr.copy()
            state = PoseState.IDLE

            if calibrating:
                remaining = 2.0 - (time.time() - calib_start)
                cv2.putText(canvas, "STAND NEUTRALLY", (FRAME_W // 2 - 200, FRAME_H // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, COL_TEXT, 3)
                if remaining <= 0:
                    if landmarks:
                        self.classifier.calibrate(landmarks.landmark)
                        calibrating = False
                        self.reset()
                    else:
                        calib_start = time.time()  # retry, no pose seen yet
            else:
                if landmarks:
                    state = self.classifier.classify(landmarks.landmark)
                self.update(state)

                if landmarks:
                    mp_draw.draw_landmarks(
                        canvas, landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=COL_SKELETON, thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=COL_SKELETON, thickness=2),
                    )

                if self.boss.phase == "RESOLVE" and self.boss.current_attack:
                    flash = np.full_like(canvas, TELEGRAPH_COLORS[self.boss.current_attack])
                    canvas = cv2.addWeighted(canvas, 0.7, flash, 0.3, 0)

                self.draw_telegraph(canvas)
                self.draw_state_label(canvas, state)

            self.draw_hp_bars(canvas)

            if self.game_over:
                self.draw_end_screen(canvas)

            cv2.imshow("Gesture Boss Rush", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.game_over:
                calibrating = True
                calib_start = time.time()
            elif key == ord('c'):
                calibrating = True
                calib_start = time.time()

        self.grabber.stop()
        self.pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureBossRush().run()