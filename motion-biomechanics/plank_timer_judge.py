import json
import os
import sys
import time

import cv2
import numpy as np
import pygame
import mediapipe as mp

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1000, 700
FPS = 30
CAM_INDEX = 0
SAMPLE_RATE = 22050

SAG_TOLERANCE = 0.05        # normalized vertical deviation allowed below the line
PIKE_TOLERANCE = 0.06       # normalized vertical deviation allowed above the line
GRACE_PERIOD = 1.2          # seconds of "bad meter" allowed before the timer pauses
RECOVERY_RATE = 1.5         # how much faster good form drains the bad meter vs. bad form fills it
ATTEMPT_RESET_AFTER = 4.0   # seconds paused before we start a brand new attempt (keeps best)
GOOD_FORM_CHIME_INTERVAL = 15.0
DIFF_SMOOTHING_ALPHA = 0.35 # EMA smoothing for the sag/pike measurement (higher = snappier, lower = smoother)
MIN_SIDE_VISIBILITY = 0.4   # average landmark visibility required to trust a side (left/right)

BEST_TIME_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plank_best.json")

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Plank Timer Judge")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 64, bold=True)
font_med = pygame.font.SysFont("arial", 30, bold=True)
font_small = pygame.font.SysFont("arial", 20)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
try:
    # Reduce internal buffering so we see the freshest frame (not all backends support this).
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# SOUND
# ----------------------------------------------------------------------------
def to_sound(samples, volume=0.7):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_tone(freq, ms, wave="sine"):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    if wave == "sine":
        tone = np.sin(2 * np.pi * freq * t)
    else:
        tone = np.sign(np.sin(2 * np.pi * freq * t))
    fade = np.linspace(1, 0, n) ** 0.5
    return to_sound(tone * fade)


snd_warning = make_tone(220, 220, "square")
snd_good = make_tone(880, 150)
snd_milestone = make_tone(1200, 300)


# ----------------------------------------------------------------------------
# PERSISTED BEST TIME
# ----------------------------------------------------------------------------
def load_best_time():
    try:
        with open(BEST_TIME_FILE, "r") as f:
            return float(json.load(f).get("best_time", 0.0))
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return 0.0


def save_best_time(value):
    try:
        with open(BEST_TIME_FILE, "w") as f:
            json.dump({"best_time": value}, f)
    except OSError:
        pass  # non-fatal: persistence is a nice-to-have, not required to run


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def pick_visible_side(landmarks):
    """Pick whichever side (left/right) is more visible/confident.

    Returns None if neither side is visible enough to trust, instead of
    silently guessing based on near-zero-confidence landmarks.
    """
    l_pts = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    r_pts = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    l_vis = sum(p.visibility for p in l_pts) / 3
    r_vis = sum(p.visibility for p in r_pts) / 3

    if max(l_vis, r_vis) < MIN_SIDE_VISIBILITY:
        return None
    return l_pts if l_vis >= r_vis else r_pts


def compute_diff(shoulder, hip, ankle):
    """Signed vertical deviation of the hip from the shoulder-ankle line.
    Positive = hip below the line (sagging), negative = above (piking)."""
    if abs(ankle.x - shoulder.x) < 1e-4:
        expected_y = (shoulder.y + ankle.y) / 2
    else:
        t = (hip.x - shoulder.x) / (ankle.x - shoulder.x)
        t = max(0.0, min(1.0, t))
        expected_y = shoulder.y + t * (ankle.y - shoulder.y)
    return hip.y - expected_y


def classify_diff(diff):
    if diff > SAG_TOLERANCE:
        return "sag"
    if diff < -PIKE_TOLERANCE:
        return "pike"
    return "good"


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.held_time = 0.0
        self.best_time = load_best_time()
        self.bad_meter = 0.0
        self.timer_running = False
        self.paused_since = None
        self.last_milestone = 0.0
        self.last_warning_sound = 0.0
        self.smoothed_diff = None
        self.attempt_reset_flash_until = 0.0

    def reset_best(self):
        self.best_time = 0.0
        save_best_time(0.0)

    def reset_attempt(self, now, flash=True):
        self.held_time = 0.0
        self.last_milestone = 0.0
        self.bad_meter = 0.0
        self.paused_since = now
        if flash:
            self.attempt_reset_flash_until = now + 2.0


state = GameState()


def update(status, dt, now):
    if status == "good":
        state.bad_meter = max(0.0, state.bad_meter - dt * RECOVERY_RATE)
    else:
        state.bad_meter += dt

    if state.bad_meter >= GRACE_PERIOD:
        state.timer_running = False
        if now - state.last_warning_sound > 1.0:
            snd_warning.play()
            state.last_warning_sound = now
    else:
        state.timer_running = True
        state.held_time += dt
        if state.held_time > state.best_time:
            state.best_time = state.held_time
            save_best_time(state.best_time)
        if state.held_time - state.last_milestone >= GOOD_FORM_CHIME_INTERVAL:
            state.last_milestone = state.held_time
            snd_milestone.play()

    if state.timer_running:
        state.paused_since = None
    else:
        if state.paused_since is None:
            state.paused_since = now
        elif now - state.paused_since >= ATTEMPT_RESET_AFTER and state.held_time > 0:
            state.reset_attempt(now)


def draw_skeleton_line(shoulder, hip, ankle, status):
    color = {"good": (60, 220, 90), "sag": (255, 90, 90), "pike": (255, 200, 60)}[status]
    pts = [(shoulder.x * WIDTH, shoulder.y * HEIGHT),
           (hip.x * WIDTH, hip.y * HEIGHT),
           (ankle.x * WIDTH, ankle.y * HEIGHT)]
    pygame.draw.lines(screen, color, False, pts, 6)
    for p in pts:
        pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), 8)
        pygame.draw.circle(screen, color, (int(p[0]), int(p[1])), 6)

    ideal_line_start = (shoulder.x * WIDTH, shoulder.y * HEIGHT)
    ideal_line_end = (ankle.x * WIDTH, ankle.y * HEIGHT)
    pygame.draw.line(screen, (255, 255, 255), ideal_line_start, ideal_line_end, 1)


def fmt_time(t):
    m = int(t // 60)
    s = t % 60
    return f"{m:02d}:{s:05.2f}"


def draw_hud(status, person_seen, now):
    timer_color = (60, 220, 90) if state.timer_running else (255, 210, 60)
    timer_label = font_big.render(fmt_time(state.held_time), True, timer_color)
    screen.blit(timer_label, (WIDTH / 2 - timer_label.get_width() / 2, 20))

    best_label = font_small.render(f"Best this session: {fmt_time(state.best_time)}",
                                    True, (255, 255, 255))
    screen.blit(best_label, (WIDTH / 2 - best_label.get_width() / 2, 90))

    # Grace-period meter: how close we are to the timer pausing.
    if state.bad_meter > 0:
        bar_w = 160
        frac = min(1.0, state.bad_meter / GRACE_PERIOD)
        bar_color = (255, 90, 90) if frac >= 1.0 else (255, 210, 60)
        pygame.draw.rect(screen, (60, 60, 60), (WIDTH / 2 - bar_w / 2, 118, bar_w, 8))
        pygame.draw.rect(screen, bar_color, (WIDTH / 2 - bar_w / 2, 118, bar_w * frac, 8))

    if not person_seen:
        warn = font_med.render("Get into frame (side-on plank view)", True, (255, 210, 60))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT / 2))
    elif status == "sag":
        warn = font_med.render("HIPS SAGGING - lift your hips!", True, (255, 90, 90))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT - 90))
    elif status == "pike":
        warn = font_med.render("HIPS TOO HIGH - lower down!", True, (255, 200, 60))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT - 90))
    elif not state.timer_running:
        warn = font_med.render("Get into position to start the timer", True, (255, 255, 255))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT - 90))

    if now < state.attempt_reset_flash_until:
        flash = font_med.render("New attempt - here we go!", True, (120, 200, 255))
        screen.blit(flash, (WIDTH / 2 - flash.get_width() / 2, 150))

    title = font_small.render(
        "PLANK TIMER JUDGE - hold a straight line from shoulder to ankle  |  R: reset best  SPACE: reset attempt",
        True, (255, 255, 255))
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 26))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    running = True
    last_time = time.time()

    try:
        while running:
            now = time.time()
            dt = now - last_time
            last_time = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    WIDTH, HEIGHT = max(400, event.w), max(300, event.h)
                    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        state.reset_best()
                    elif event.key == pygame.K_SPACE:
                        state.reset_attempt(now, flash=False)

            ok, frame = cap.read()
            if not ok:
                clock.tick(FPS)
                continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            picked = pick_visible_side(results.pose_landmarks.landmark) if results.pose_landmarks else None

            person_seen = False
            status = "absent"
            shoulder = hip = ankle = None

            if picked:
                person_seen = True
                shoulder, hip, ankle = picked
                diff = compute_diff(shoulder, hip, ankle)
                if state.smoothed_diff is None:
                    state.smoothed_diff = diff
                else:
                    state.smoothed_diff = (DIFF_SMOOTHING_ALPHA * diff
                                            + (1 - DIFF_SMOOTHING_ALPHA) * state.smoothed_diff)
                status = classify_diff(state.smoothed_diff)
            else:
                state.smoothed_diff = None

            update(status, dt, now)

            surf = frame_to_surface(frame)
            screen.blit(surf, (0, 0))
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 70))
            screen.blit(dim, (0, 0))

            if person_seen:
                draw_skeleton_line(shoulder, hip, ankle, status)

            draw_hud(status, person_seen, now)

            pygame.display.flip()
            clock.tick(FPS)
    finally:
        save_best_time(state.best_time)
        cap.release()
        pose.close()
        pygame.quit()


if __name__ == "__main__":
    main()