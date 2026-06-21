import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import sys
import threading
from collections import deque

# ─── INIT AND AUDIO CONFIGURATION ───────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

BASE_W, BASE_H = 1280, 720
screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
pygame.display.set_caption("🧠 Emotion Lies Detector")
clock = pygame.time.Clock()

SAMPLE_RATE = 44100

def generate_procedural_sound(freq=440, duration=0.2, wave_type='sine', volume=0.4):
    """Generates a procedural audio chunk directly inside RAM to avoid external file dependencies."""
    frames = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, frames, False)

    if wave_type == 'sine':
        data = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        data = np.sign(np.sin(2 * np.pi * freq * t))
    elif wave_type == 'sawtooth':
        data = 2 * (t * freq - np.floor(t * freq + 0.5))
    else:
        data = np.random.normal(0, 1, frames)  # White noise fallback

    decay = np.exp(-3 * np.linspace(0, 1, frames))
    data = data * decay * 32767 * volume
    audio_buffer = data.astype(np.int16)
    stereo_buffer = np.vstack((audio_buffer, audio_buffer)).T.copy()
    return pygame.sndarray.make_sound(stereo_buffer)

SOUND_ALERT = generate_procedural_sound(freq=660, duration=0.15, wave_type='square', volume=0.3)
SOUND_SUCCESS = generate_procedural_sound(freq=880, duration=0.25, wave_type='sine', volume=0.4)
SOUND_FAIL = generate_procedural_sound(freq=180, duration=0.4, wave_type='sawtooth', volume=0.5)
SOUND_TICK = generate_procedural_sound(freq=440, duration=0.04, wave_type='sine', volume=0.2)
SOUND_CALIBRATE = generate_procedural_sound(freq=520, duration=0.12, wave_type='sine', volume=0.25)
SOUND_COMBO = generate_procedural_sound(freq=1040, duration=0.18, wave_type='sine', volume=0.35)
SOUND_VERDICT_TRUTH = generate_procedural_sound(freq=720, duration=0.2, wave_type='sine', volume=0.35)
SOUND_VERDICT_LIE = generate_procedural_sound(freq=260, duration=0.3, wave_type='square', volume=0.35)

# ─── THREAD-SAFE VIDEO MANAGER ──────────────────────────────────────────────
class ThreadedVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.stream.read()
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while self.running:
            grabbed, frame = self.stream.read()
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.stream.release()

# ─── TUNABLE THRESHOLDS (relative to calibrated neutral baseline) ──────────
LIFT_THRESH = 0.014        # how much corners must rise above neutral to register HAPPY
DROOP_THRESH = 0.012       # how much corners must fall below neutral to register SAD/ANGRY
WIDTH_THRESH = 0.035       # mouth-width stretch (relative to interocular distance) to count as a grin
OPEN_THRESH = 0.05         # mouth-open delta (relative to interocular distance) without a smile curve
POSTURE_UP_THRESH = 0.05   # shoulder rise relative to neutral baseline -> ENERGETIC/OPEN
POSTURE_DOWN_THRESH = 0.05 # shoulder drop relative to neutral baseline -> DEFENSIVE/SLOUCHED
EMA_ALPHA = 0.4            # smoothing factor for jitter reduction
DEBOUNCE_FRAMES = 3        # consecutive frames required before a classification flips
CALIBRATION_SECONDS = 3.0
VERDICT_GAP_SECONDS = 2.0  # pause between questions where the result is revealed

# ─── CORE DETECTOR AND STATE CALCULATORS ────────────────────────────────────
class HumanMetricEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.pose = self.mp_pose.Pose(model_complexity=1)

        self.smooth = {
            "mouth_width_n": None,
            "corner_lift": None,
            "mouth_open_n": None,
            "posture_y": None,
            "shoulder_w": None,
        }

        self.baseline = {
            "mouth_width_n": 0.0,
            "corner_lift": 0.0,
            "mouth_open_n": 0.0,
            "posture_y": 0.5,
            "shoulder_w": 0.2,
        }
        self.calibrated = False

        self._pending_emotion = "NEUTRAL"
        self._pending_emotion_count = 0
        self._stable_emotion = "NO FACE DETECTED"

        self._pending_posture = "NEUTRAL"
        self._pending_posture_count = 0
        self._stable_posture = "NO POSTURE DETECTED"

        self.jitter_window = deque(maxlen=30)

    def _ema(self, key, value):
        prev = self.smooth[key]
        if prev is None:
            self.smooth[key] = value
        else:
            self.smooth[key] = prev + EMA_ALPHA * (value - prev)
        return self.smooth[key]

    def _raw_metrics(self, bgr_frame):
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        face_found = False
        pose_found = False
        raw = {"mouth_width_n": 0.0, "corner_lift": 0.0, "mouth_open_n": 0.0,
               "posture_y": 0.5, "shoulder_w": 0.2}

        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark

            left_eye_outer = np.array([lm[33].x, lm[33].y])
            right_eye_outer = np.array([lm[263].x, lm[263].y])
            norm = max(1e-4, np.linalg.norm(left_eye_outer - right_eye_outer))

            top_lip = np.array([lm[13].x, lm[13].y])
            bottom_lip = np.array([lm[14].x, lm[14].y])
            left_mouth = np.array([lm[61].x, lm[61].y])
            right_mouth = np.array([lm[291].x, lm[291].y])

            mouth_width = np.linalg.norm(left_mouth - right_mouth)
            mouth_open = np.linalg.norm(top_lip - bottom_lip)
            lip_center_y = (top_lip[1] + bottom_lip[1]) / 2.0
            avg_corner_y = (left_mouth[1] + right_mouth[1]) / 2.0

            raw["mouth_width_n"] = mouth_width / norm
            raw["mouth_open_n"] = mouth_open / norm
            raw["corner_lift"] = (lip_center_y - avg_corner_y) / norm
            face_found = True

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].visibility])
            right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].visibility])

            if left_shoulder[2] > 0.5 and right_shoulder[2] > 0.5:
                raw["posture_y"] = (left_shoulder[1] + right_shoulder[1]) / 2.0
                raw["shoulder_w"] = abs(left_shoulder[0] - right_shoulder[0])
                pose_found = True

        return raw, face_found, pose_found

    def set_baseline(self, face_samples, pose_samples):
        if face_samples:
            self.baseline["mouth_width_n"] = float(np.mean([s["mouth_width_n"] for s in face_samples]))
            self.baseline["corner_lift"] = float(np.mean([s["corner_lift"] for s in face_samples]))
            self.baseline["mouth_open_n"] = float(np.mean([s["mouth_open_n"] for s in face_samples]))
        if pose_samples:
            self.baseline["posture_y"] = float(np.mean([s["posture_y"] for s in pose_samples]))
            self.baseline["shoulder_w"] = float(np.mean([s["shoulder_w"] for s in pose_samples]))
        self.calibrated = True
        # Reset smoothing/debounce so stale pre-calibration values don't linger
        for k in self.smooth:
            self.smooth[k] = None
        self._pending_emotion_count = 0
        self._pending_posture_count = 0
        self.jitter_window.clear()

    def process_frame(self, bgr_frame):
        raw, face_found, pose_found = self._raw_metrics(bgr_frame)

        d_lift = d_width = d_open = 0.0

        if face_found:
            mw = self._ema("mouth_width_n", raw["mouth_width_n"])
            cl = self._ema("corner_lift", raw["corner_lift"])
            mo = self._ema("mouth_open_n", raw["mouth_open_n"])

            d_lift = cl - self.baseline["corner_lift"]
            d_width = mw - self.baseline["mouth_width_n"]
            d_open = mo - self.baseline["mouth_open_n"]

            self.jitter_window.append(d_lift)

            if d_lift > LIFT_THRESH or (d_width > WIDTH_THRESH and d_lift > -DROOP_THRESH * 0.5):
                frame_emotion_candidate = "HAPPY"
            elif d_lift < -DROOP_THRESH or (d_open > OPEN_THRESH and d_lift <= 0):
                frame_emotion_candidate = "SAD/ANGRY"
            else:
                frame_emotion_candidate = "NEUTRAL"
        else:
            frame_emotion_candidate = "NO FACE DETECTED"

        if frame_emotion_candidate == self._pending_emotion:
            self._pending_emotion_count += 1
        else:
            self._pending_emotion = frame_emotion_candidate
            self._pending_emotion_count = 1

        if self._pending_emotion_count >= DEBOUNCE_FRAMES or frame_emotion_candidate == "NO FACE DETECTED":
            self._stable_emotion = frame_emotion_candidate

        if pose_found:
            py = self._ema("posture_y", raw["posture_y"])
            d_post = py - self.baseline["posture_y"]

            if d_post < -POSTURE_UP_THRESH:
                frame_posture_candidate = "ENERGETIC/OPEN"
            elif d_post > POSTURE_DOWN_THRESH:
                frame_posture_candidate = "DEFENSIVE/SLOUCHED"
            else:
                frame_posture_candidate = "NEUTRAL"
        else:
            frame_posture_candidate = "NO POSTURE DETECTED"

        if frame_posture_candidate == self._pending_posture:
            self._pending_posture_count += 1
        else:
            self._pending_posture = frame_posture_candidate
            self._pending_posture_count = 1

        if self._pending_posture_count >= DEBOUNCE_FRAMES or frame_posture_candidate == "NO POSTURE DETECTED":
            self._stable_posture = frame_posture_candidate

        stress_level = 0.0
        if len(self.jitter_window) > 4:
            stress_level = float(np.std(self.jitter_window)) / max(1e-4, LIFT_THRESH)
            stress_level = min(1.0, stress_level)

        metrics = {
            "d_lift": d_lift,
            "d_width": d_width,
            "d_open": d_open,
            "stress": stress_level,
        }

        return self._stable_emotion, self._stable_posture, metrics, face_found, pose_found

# ─── PIPELINE GAME ARCHITECTURE ─────────────────────────────────────────────
class EmotionLiesGame:
    def __init__(self):
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.game_state = "PROLOGUE"  # PROLOGUE, CALIBRATING, CHALLENGE, VERDICT

        self.emotions_pool = ["HAPPY", "SAD/ANGRY", "NEUTRAL"]
        self.postures_pool = ["ENERGETIC/OPEN", "DEFENSIVE/SLOUCHED", "NEUTRAL"]

        self.target_emotion = ""
        self.target_posture = ""
        self.is_lie_challenge = False

        self.base_challenge_duration = 6.0
        self.challenge_duration = self.base_challenge_duration
        self.challenge_timer = 0.0

        self.verdict_until = 0.0
        self.verdict_text = ""
        self.verdict_color = (255, 255, 255)
        self.verdict_is_truth = True
        self.verdict_detail = ""

        self.calibration_start = 0.0
        self.face_samples = []
        self.pose_samples = []

    def begin_calibration(self):
        self.game_state = "CALIBRATING"
        self.calibration_start = time.time()
        self.face_samples = []
        self.pose_samples = []
        SOUND_CALIBRATE.play()

    def calibration_progress(self):
        elapsed = time.time() - self.calibration_start
        return min(1.0, elapsed / CALIBRATION_SECONDS)

    def calibration_done(self):
        return (time.time() - self.calibration_start) >= CALIBRATION_SECONDS

    def difficulty_scaled_duration(self):
        reduction = min(3.0, self.score / 1000.0)
        return max(3.0, self.base_challenge_duration - reduction)

    def spawn_challenge(self):
        self.target_emotion = random.choice(self.emotions_pool)
        self.target_posture = random.choice(self.postures_pool)
        self.is_lie_challenge = random.choice([True, False])
        self.challenge_duration = self.difficulty_scaled_duration()
        self.challenge_timer = time.time() + self.challenge_duration
        self.game_state = "CHALLENGE"
        SOUND_ALERT.play()

    def evaluate_round(self, live_emotion, live_posture, face_found, pose_found):
        """Called once when the CHALLENGE timer expires. Emotion match is the
        hard requirement (posture is unreliable to track); posture match is a
        bonus. Always lands in VERDICT with a 2s gap before the next question."""
        if not face_found:
            self.score = max(0, self.score - 30)
            self.streak = 0
            self.verdict_text = "❌ NO READING — FACE LOST"
            self.verdict_color = (255, 140, 0)
            self.verdict_is_truth = False
            self.verdict_detail = "Stay in frame so the detector can read you."
            SOUND_FAIL.play()
        else:
            emotion_matched = (live_emotion == self.target_emotion)
            posture_matched = pose_found and (live_posture == self.target_posture)

            if emotion_matched:
                self.streak += 1
                self.best_streak = max(self.best_streak, self.streak)
                multiplier = 1 + min(4, self.streak // 3)
                gained = (150 + (50 if posture_matched else 0)) * multiplier
                self.score += gained
                self.verdict_is_truth = True
                self.verdict_text = "✅ TRUTH CONFIRMED"
                self.verdict_color = (0, 255, 120)
                bonus_note = " + posture bonus" if posture_matched else ""
                self.verdict_detail = f"+{gained} pts{bonus_note}  (x{multiplier} streak)"
                if multiplier > 1 or posture_matched:
                    SOUND_COMBO.play()
                else:
                    SOUND_SUCCESS.play()
                SOUND_VERDICT_TRUTH.play()
            else:
                self.streak = 0
                self.score = max(0, self.score - 75)
                self.verdict_is_truth = False
                self.verdict_text = "❌ LIE DETECTED"
                self.verdict_color = (255, 50, 50)
                self.verdict_detail = f"Wanted FACE: {self.target_emotion} — read as: {live_emotion}"
                SOUND_FAIL.play()
                SOUND_VERDICT_LIE.play()

        self.verdict_until = time.time() + VERDICT_GAP_SECONDS
        self.game_state = "VERDICT"

# ─── RUNTIME PIPELINE EXECUTION ─────────────────────────────────────────────
def main():
    video_capture = ThreadedVideoStream(src=0)
    analyzer = HumanMetricEngine()
    game = EmotionLiesGame()

    font_large = pygame.font.SysFont("arial", 42, bold=True)
    font_medium = pygame.font.SysFont("arial", 26, bold=True)
    font_small = pygame.font.SysFont("arial", 18, bold=False)
    font_huge = pygame.font.SysFont("arial", 56, bold=True)

    running = True
    while running:
        window_w, window_h = screen.get_size()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if game.game_state == "PROLOGUE":
                        game.score = 0
                        game.streak = 0
                        game.begin_calibration()
                elif event.key == pygame.K_r:
                    if game.game_state in ("CHALLENGE", "VERDICT"):
                        game.begin_calibration()

        screen.fill((15, 23, 42))

        grabbed, bgr_frame = video_capture.read()
        live_emotion, live_posture = "NO FACE DETECTED", "NO POSTURE DETECTED"
        metrics = {"d_lift": 0.0, "d_width": 0.0, "d_open": 0.0, "stress": 0.0}
        face_found = pose_found = False

        camera_surface = None
        if grabbed and bgr_frame is not None:
            if game.game_state == "CALIBRATING":
                raw_for_calibration, face_found, pose_found = analyzer._raw_metrics(bgr_frame)
                if face_found:
                    game.face_samples.append(raw_for_calibration)
                if pose_found:
                    game.pose_samples.append(raw_for_calibration)
            else:
                live_emotion, live_posture, metrics, face_found, pose_found = analyzer.process_frame(bgr_frame)

            bgr_frame = cv2.flip(bgr_frame, 1)
            rgb_render = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            raw_surface = pygame.surfarray.make_surface(np.rot90(rgb_render))
            camera_surface = pygame.transform.smoothscale(raw_surface, (320, 240))

        if game.game_state == "CALIBRATING" and game.calibration_done():
            analyzer.set_baseline(game.face_samples, game.pose_samples)
            game.spawn_challenge()

        elif game.game_state == "CHALLENGE" and time.time() >= game.challenge_timer:
            game.evaluate_round(live_emotion, live_posture, face_found, pose_found)

        elif game.game_state == "VERDICT" and time.time() >= game.verdict_until:
            game.spawn_challenge()

        # 1. Render Video Overlay Panel
        if camera_surface:
            overlay_x = window_w - 340
            overlay_y = 20
            pygame.draw.rect(screen, (30, 41, 59), (overlay_x - 4, overlay_y - 4, 328, 248), border_radius=6)
            screen.blit(camera_surface, (overlay_x, overlay_y))

            color_emo = (239, 68, 68) if live_emotion == "NO FACE DETECTED" else (255, 255, 255)
            color_pos = (239, 68, 68) if live_posture == "NO POSTURE DETECTED" else (255, 255, 255)

            lbl_live_emo = font_small.render(f"FACE: {live_emotion}", True, color_emo)
            lbl_live_pos = font_small.render(f"BODY: {live_posture}", True, color_pos)
            screen.blit(lbl_live_emo, (overlay_x + 10, overlay_y + 190))
            screen.blit(lbl_live_pos, (overlay_x + 10, overlay_y + 215))

        # 2. Main HUD Interface
        if game.game_state == "PROLOGUE":
            txt_title = font_large.render("EMOTION LIES DETECTOR", True, (56, 189, 248))
            txt_sub = font_medium.render("Test your body's deceptive alignment integrity.", True, (148, 163, 184))
            txt_prompt = font_medium.render("[ PRESS SPACE TO CALIBRATE & BEGIN ]", True, (34, 211, 238))

            screen.blit(txt_title, (50, window_h // 2 - 100))
            screen.blit(txt_sub, (50, window_h // 2 - 40))
            screen.blit(txt_prompt, (50, window_h // 2 + 40))

        elif game.game_state == "CALIBRATING":
            progress = game.calibration_progress()
            txt_title = font_large.render("CALIBRATING NEUTRAL BASELINE...", True, (250, 204, 21))
            txt_sub = font_medium.render("Hold a relaxed, neutral face and posture.", True, (226, 232, 240))
            screen.blit(txt_title, (50, window_h // 2 - 100))
            screen.blit(txt_sub, (50, window_h // 2 - 50))

            bar_w = 500
            pygame.draw.rect(screen, (30, 41, 59), (50, window_h // 2, bar_w, 20), border_radius=6)
            pygame.draw.rect(screen, (250, 204, 21), (50, window_h // 2, int(bar_w * progress), 20), border_radius=6)

            if not face_found:
                txt_warn = font_small.render("⚠ No face detected — move into frame.", True, (248, 113, 113))
                screen.blit(txt_warn, (50, window_h // 2 + 40))

        elif game.game_state in ("CHALLENGE", "VERDICT"):
            lbl_score = font_medium.render(f"CREDIBILITY SCORE: {game.score}", True, (255, 255, 255))
            screen.blit(lbl_score, (50, 30))

            lbl_streak = font_small.render(
                f"Streak: {game.streak}  (Best: {game.best_streak})   [R = recalibrate]",
                True, (148, 163, 184))
            screen.blit(lbl_streak, (50, 70))

            challenge_y = 120
            pygame.draw.rect(screen, (30, 41, 59), (50, challenge_y, 650, 220), border_radius=12)
            pygame.draw.rect(screen, (71, 85, 105), (50, challenge_y, 650, 220), width=2, border_radius=12)

            if game.is_lie_challenge:
                header_text = "⚠️ DETECTOR DRIFT CRITICAL: SIMULATE A LIE!"
                header_color = (239, 68, 68)
            else:
                header_text = "✅ TRUTH MATRIX CALIBRATION: EMULATE HARMONY!"
                header_color = (34, 197, 94)

            lbl_task_header = font_medium.render(header_text, True, header_color)
            screen.blit(lbl_task_header, (70, challenge_y + 20))

            lbl_req_emo = font_large.render(f"REQUIRED FACE: {game.target_emotion}", True, (241, 245, 249))
            lbl_req_pos = font_large.render(f"REQUIRED BODY: {game.target_posture}", True, (241, 245, 249))
            screen.blit(lbl_req_emo, (70, challenge_y + 75))
            screen.blit(lbl_req_pos, (70, challenge_y + 135))

            bar_w = 650
            pygame.draw.rect(screen, (15, 23, 42), (50, challenge_y + 235, bar_w, 14), border_radius=4)

            if game.game_state == "CHALLENGE":
                time_left = max(0.0, game.challenge_timer - time.time())
                time_ratio = time_left / game.challenge_duration if game.challenge_duration > 0 else 0
                if time_ratio > 0:
                    bar_color = (56, 189, 248) if time_ratio > 0.4 else (248, 113, 113)
                    pygame.draw.rect(screen, bar_color, (50, challenge_y + 235, int(bar_w * time_ratio), 14), border_radius=4)
            else:
                # VERDICT gap — show countdown to next question instead
                gap_left = max(0.0, game.verdict_until - time.time())
                gap_ratio = gap_left / VERDICT_GAP_SECONDS
                pygame.draw.rect(screen, (148, 163, 184), (50, challenge_y + 235, int(bar_w * gap_ratio), 14), border_radius=4)

            if game.game_state == "VERDICT":
                # Big verdict banner over the challenge box
                overlay_rect = pygame.Rect(50, challenge_y, 650, 220)
                veil = pygame.Surface((overlay_rect.w, overlay_rect.h), pygame.SRCALPHA)
                veil.fill((15, 23, 42, 215))
                screen.blit(veil, overlay_rect.topleft)

                lbl_verdict = font_huge.render(game.verdict_text, True, game.verdict_color)
                vw = lbl_verdict.get_width()
                screen.blit(lbl_verdict, (50 + (650 - vw) // 2, challenge_y + 50))

                lbl_detail = font_small.render(game.verdict_detail, True, (203, 213, 225))
                dw = lbl_detail.get_width()
                screen.blit(lbl_detail, (50 + (650 - dw) // 2, challenge_y + 120))

                next_in = max(0.0, game.verdict_until - time.time())
                lbl_next = font_small.render(f"Next question in {next_in:.1f}s...", True, (100, 116, 139))
                nw = lbl_next.get_width()
                screen.blit(lbl_next, (50 + (650 - nw) // 2, challenge_y + 160))

            metrics_y = window_h - 200
            pygame.draw.rect(screen, (30, 41, 59), (50, metrics_y, 480, 160), border_radius=8)
            lbl_metrics_title = font_small.render("🔬 RAW TELEMETRY DATA STREAM", True, (148, 163, 184))
            screen.blit(lbl_metrics_title, (65, metrics_y + 10))

            lbl_m1 = font_small.render(f"Corner Lift Δ (smile curve): {metrics['d_lift']:+.4f}", True, (203, 213, 225))
            lbl_m2 = font_small.render(f"Mouth Width Δ (grin stretch): {metrics['d_width']:+.4f}", True, (203, 213, 225))
            lbl_m3 = font_small.render(f"Mouth Open Δ:                {metrics['d_open']:+.4f}", True, (203, 213, 225))
            screen.blit(lbl_m1, (65, metrics_y + 40))
            screen.blit(lbl_m2, (65, metrics_y + 70))
            screen.blit(lbl_m3, (65, metrics_y + 100))

            stress = metrics.get("stress", 0.0)
            lbl_stress = font_small.render("MICRO-EXPRESSION INSTABILITY ('THE TELL')", True, (148, 163, 184))
            screen.blit(lbl_stress, (65, metrics_y + 130))
            stress_bar_w = 300
            pygame.draw.rect(screen, (15, 23, 42), (65, metrics_y + 150, stress_bar_w, 12), border_radius=4)
            stress_color = (34, 197, 94) if stress < 0.4 else ((250, 204, 21) if stress < 0.75 else (239, 68, 68))
            pygame.draw.rect(screen, stress_color, (65, metrics_y + 150, int(stress_bar_w * stress), 12), border_radius=4)

        lbl_exit_prompt = font_small.render("Press 'Q' to safely terminate tracking pipelines.", True, (100, 116, 139))
        screen.blit(lbl_exit_prompt, (50, window_h - 35))

        pygame.display.flip()
        clock.tick(60)

    video_capture.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()