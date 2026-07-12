import sys
import time

import cv2
import numpy as np
import pygame
import mediapipe as mp

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 480, 360     # starts small since this is meant to sit in a corner
FPS = 15                     # low FPS is fine for a background watchdog, saves CPU
CAM_INDEX = 0
SAMPLE_RATE = 22050

SLOUCH_TOLERANCE_DEG = 12.0     # degrees of neck-angle deviation allowed from baseline
SLOUCH_GRACE_PERIOD = 8.0       # seconds of slouching allowed before nudging
REMINDER_REPEAT_INTERVAL = 25.0

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Posture Watchdog")
clock = pygame.time.Clock()
font_med = pygame.font.SysFont("arial", 22, bold=True)
font_small = pygame.font.SysFont("arial", 15)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# SOUND
# ----------------------------------------------------------------------------
def to_sound(samples, volume=0.4):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_gentle_chime(ms=500):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone1 = np.sin(2 * np.pi * 523 * t) * np.exp(-t * 3)
    tone2 = np.sin(2 * np.pi * 659 * t) * np.exp(-t * 3) * (t > ms / 1000 * 0.2)
    return to_sound(tone1 * 0.6 + tone2 * 0.4)


snd_reminder = make_gentle_chime()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def neck_angle_deg(landmarks):
    """
    Approximate forward-head/neck-bend angle using the shoulder midpoint and
    ear position: a bigger horizontal offset of the ear ahead of the
    shoulder (relative to torso height) = more forward head tilt/slouch.
    """
    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    ear = l_ear if l_ear.visibility >= r_ear.visibility else r_ear
    shoulder_mid_x = (l_sh.x + r_sh.x) / 2
    shoulder_mid_y = (l_sh.y + r_sh.y) / 2

    dx = ear.x - shoulder_mid_x
    dy = shoulder_mid_y - ear.y   # positive: ear above shoulder (normal)
    if dy <= 1e-4:
        return 90.0
    angle = np.degrees(np.arctan2(abs(dx), dy))
    return angle if dx >= 0 else -angle


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.calibrated = False
        self.baseline_angle = 0.0
        self.slouch_since = None
        self.last_reminder = 0
        self.muted = False

        self.session_start = time.time()
        self.good_time = 0.0
        self.bad_time = 0.0

    def reset_session(self):
        self.__init__()


state = GameState()


def update(angle, dt, now):
    if not state.calibrated:
        return "uncalibrated"

    deviation = abs(angle - state.baseline_angle)
    is_good = deviation <= SLOUCH_TOLERANCE_DEG

    if is_good:
        state.good_time += dt
        state.slouch_since = None
        return "good"
    else:
        state.bad_time += dt
        if state.slouch_since is None:
            state.slouch_since = now
        slouch_duration = now - state.slouch_since
        if slouch_duration > SLOUCH_GRACE_PERIOD:
            if not state.muted and now - state.last_reminder > REMINDER_REPEAT_INTERVAL:
                snd_reminder.play()
                state.last_reminder = now
            return "slouching"
        return "warning"


def draw_hud(status, person_seen):
    if not person_seen:
        msg = font_small.render("No person detected", True, (255, 120, 120))
        screen.blit(msg, (10, 10))
    elif not state.calibrated:
        msg = font_med.render("Sit up straight, then press C", True, (255, 210, 60))
        screen.blit(msg, (10, HEIGHT / 2 - 15))
    else:
        color = {"good": (80, 220, 120), "warning": (255, 210, 60),
                 "slouching": (255, 90, 90)}.get(status, (200, 200, 200))
        text = {"good": "Good posture", "warning": "Adjusting...",
                "slouching": "SLOUCHING - sit up!"}.get(status, "")
        label = font_med.render(text, True, color)
        screen.blit(label, (10, 10))

        total = max(1e-3, state.good_time + state.bad_time)
        pct = state.good_time / total * 100
        pct_label = font_small.render(f"Good posture: {pct:0.0f}% of session", True, (255, 255, 255))
        screen.blit(pct_label, (10, HEIGHT - 50))

        elapsed = time.time() - state.session_start
        time_label = font_small.render(f"Session: {int(elapsed // 60)}m {int(elapsed % 60)}s",
                                        True, (255, 255, 255))
        screen.blit(time_label, (10, HEIGHT - 28))

    mute_label = font_small.render(f"[M] Sound: {'off' if state.muted else 'on'}   [C] Recalibrate",
                                    True, (200, 200, 200))
    screen.blit(mute_label, (10, HEIGHT - 72))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    running = True
    last_time = time.time()

    while running:
        now = time.time()
        dt = now - last_time
        last_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = max(240, event.w), max(180, event.h)
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_m:
                    state.muted = not state.muted

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        person_seen = False
        status = "uncalibrated"
        angle = 0.0
        if results.pose_landmarks:
            person_seen = True
            angle = neck_angle_deg(results.pose_landmarks.landmark)

            keys_now = pygame.key.get_pressed()
            if keys_now[pygame.K_c]:
                state.baseline_angle = angle
                state.calibrated = True

            status = update(angle, dt, now)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        screen.blit(dim, (0, 0))

        draw_hud(status, person_seen)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()