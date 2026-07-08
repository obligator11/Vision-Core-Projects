

import math
import random
import sys
import time

import cv2
import numpy as np
import pygame
import mediapipe as mp
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1000, 700
FPS = 30
CAM_INDEX = 0

WATCH_TIME_RANGE = (1.5, 3.5)     # guard watching (freeze!)
AWAY_TIME_RANGE = (1.8, 3.8)      # guard looking away (safe to move)
TURN_WARNING_TIME = 0.5           # guard "starting to turn" telegraph before fully watching

PROGRESS_GOAL = 100.0
STARTING_LIVES = 3

# Movement is measured as a VELOCITY (normalized units / second), not a raw
# per-frame delta, so it stays consistent even if the frame rate jitters.
# Lower this if small movements still aren't being caught; raise it if the
# game feels too twitchy / triggers on camera noise.
MOVEMENT_THRESHOLD = 0.35
SHOW_MOVEMENT_DEBUG = True        # shows the live movement value on screen for tuning

SUSPICION_MAX = 100.0
SUSPICION_RISE_RATE = 220.0       # per second while moving & watched
SUSPICION_FALL_RATE = 60.0        # per second otherwise

KEY_LANDMARKS = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                  "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "NOSE"]

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stealth Freeze")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 56, bold=True)
font_med = pygame.font.SysFont("arial", 30, bold=True)
font_small = pygame.font.SysFont("arial", 20)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

print("Loading YOLOv8n (first run downloads weights)...")
yolo_model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def get_person_bbox(frame_bgr):
    results = yolo_model.predict(frame_bgr, classes=[0], verbose=False, conf=0.4)
    best, best_area = None, 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area, best = area, (x1, y1, x2, y2)
    return best


def get_key_points(landmarks):
    return {name: (landmarks[mp_pose.PoseLandmark[name].value].x,
                    landmarks[mp_pose.PoseLandmark[name].value].y)
            for name in KEY_LANDMARKS}


def movement_amount(prev_pts, cur_pts):
    """
    Returns the movement of the single MOST-moved tracked landmark
    (not the average across all of them).

    Why: averaging across 9 landmarks (wrists, shoulders, hips, knees, nose)
    silently kills sensitivity - if you only move one arm, the other 8 mostly
    static points drag the average way down and real movement never crosses
    the threshold. Using the max means any one limb moving is enough to
    register, which matches how a person actually perceives "did they move?".
    """
    if prev_pts is None or cur_pts is None:
        return 0.0
    deltas = []
    for name in KEY_LANDMARKS:
        px, py = prev_pts[name]
        cx, cy = cur_pts[name]
        deltas.append(math.hypot(px - cx, py - cy))
    return max(deltas)


# ----------------------------------------------------------------------------
# GAME STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.guard_state = "AWAY"       # AWAY (safe) -> WARNING (telegraph) -> WATCHING (freeze)
        self.state_start = time.time()
        self.state_duration = random.uniform(*AWAY_TIME_RANGE)
        self.progress = 0.0
        self.lives = STARTING_LIVES
        self.suspicion = 0.0
        self.caught_flash_until = 0
        self.won = False
        self.lost = False

    def reset(self):
        self.__init__()


state = GameState()


def advance_guard_state():
    now = time.time()
    if state.guard_state == "AWAY":
        state.guard_state = "WARNING"
        state.state_duration = TURN_WARNING_TIME
    elif state.guard_state == "WARNING":
        state.guard_state = "WATCHING"
        state.state_duration = random.uniform(*WATCH_TIME_RANGE)
    else:  # WATCHING
        state.guard_state = "AWAY"
        state.state_duration = random.uniform(*AWAY_TIME_RANGE)
    state.state_start = now


def update_game(move_amt, person_seen, dt):
    if state.won or state.lost:
        return
    now = time.time()
    if now - state.state_start >= state.state_duration:
        advance_guard_state()

    if not person_seen:
        return

    # Convert the raw per-frame delta into a velocity (units/sec) so
    # detection doesn't depend on the current frame rate.
    velocity = move_amt / max(dt, 1e-3)

    is_watching = state.guard_state == "WATCHING"
    is_moving = velocity > MOVEMENT_THRESHOLD

    if is_watching and is_moving:
        state.suspicion += SUSPICION_RISE_RATE * dt
    else:
        state.suspicion -= SUSPICION_FALL_RATE * dt
    state.suspicion = max(0.0, min(SUSPICION_MAX, state.suspicion))

    if state.suspicion >= SUSPICION_MAX:
        state.lives -= 1
        state.suspicion = 0.0
        state.caught_flash_until = now + 0.6
        state.progress = max(0.0, state.progress - 15)
        if state.lives <= 0:
            state.lost = True

    if not is_watching and is_moving:
        state.progress += min(velocity, 2.5) * dt * 12
        state.progress = min(PROGRESS_GOAL, state.progress)
        if state.progress >= PROGRESS_GOAL:
            state.won = True

    return velocity


def draw_guard():
    cx, cy = WIDTH - 120, 140
    is_watching = state.guard_state == "WATCHING"
    is_warning = state.guard_state == "WARNING"

    body_color = (90, 90, 100)
    pygame.draw.rect(screen, body_color, (cx - 30, cy, 60, 90), border_radius=10)
    head_color = (230, 60, 60) if is_watching else ((240, 200, 60) if is_warning else (60, 200, 90))
    pygame.draw.circle(screen, head_color, (cx, cy - 15), 32)

    # eyes indicate facing direction: watching = eyes toward camera (player)
    if is_watching:
        pygame.draw.circle(screen, (0, 0, 0), (cx - 10, cy - 20), 5)
        pygame.draw.circle(screen, (0, 0, 0), (cx + 10, cy - 20), 5)
    elif is_warning:
        pygame.draw.circle(screen, (0, 0, 0), (cx - 5, cy - 20), 4)
        pygame.draw.circle(screen, (0, 0, 0), (cx + 15, cy - 20), 4)
    else:
        # looking away - draw back of head (no eyes visible)
        pygame.draw.arc(screen, (0, 0, 0), (cx - 20, cy - 35, 40, 30), 0.3, 2.8, 3)

    label_text = {"WATCHING": "WATCHING! FREEZE!", "WARNING": "turning around...",
                  "AWAY": "not looking - go!"}[state.guard_state]
    label_color = {"WATCHING": (255, 90, 90), "WARNING": (255, 210, 60),
                   "AWAY": (90, 230, 120)}[state.guard_state]
    label = font_small.render(label_text, True, label_color)
    screen.blit(label, (cx - label.get_width() / 2, cy + 100))


def draw_hud(person_seen, cached_bbox, frame_shape, velocity=0.0):
    now = time.time()
    draw_guard()

    if SHOW_MOVEMENT_DEBUG:
        debug_color = (255, 90, 90) if velocity > MOVEMENT_THRESHOLD else (140, 220, 140)
        debug_label = font_small.render(
            f"movement velocity: {velocity:0.3f}  (threshold: {MOVEMENT_THRESHOLD})",
            True, debug_color)
        screen.blit(debug_label, (WIDTH / 2 - debug_label.get_width() / 2, HEIGHT - 120))

    # Suspicion meter
    bar_x, bar_y, bar_w, bar_h = 20, 60, 260, 24
    pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
    fill_w = int(bar_w * (state.suspicion / SUSPICION_MAX))
    fill_color = (250, 80, 80) if state.suspicion > 70 else (250, 210, 60)
    pygame.draw.rect(screen, fill_color, (bar_x, bar_y, fill_w, bar_h), border_radius=6)
    pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=6)
    susp_label = font_small.render("SUSPICION", True, (255, 255, 255))
    screen.blit(susp_label, (bar_x, bar_y - 22))

    # Progress bar
    bar_y2 = HEIGHT - 60
    pygame.draw.rect(screen, (40, 40, 40), (20, bar_y2, WIDTH - 40, 30), border_radius=8)
    fill_w2 = int((WIDTH - 40) * (state.progress / PROGRESS_GOAL))
    pygame.draw.rect(screen, (60, 200, 255), (20, bar_y2, fill_w2, 30), border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), (20, bar_y2, WIDTH - 40, 30), 2, border_radius=8)
    prog_label = font_small.render("DISTANCE TO VAULT", True, (255, 255, 255))
    screen.blit(prog_label, (20, bar_y2 - 24))

    # Lives
    for i in range(STARTING_LIVES):
        color = (230, 60, 60) if i < state.lives else (70, 70, 70)
        pygame.draw.circle(screen, color, (30 + i * 34, 20), 12)

    if not person_seen:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT / 2 - 200))

    if cached_bbox:
        x1, y1, x2, y2 = cached_bbox
        sx = WIDTH / frame_shape[1]
        sy = HEIGHT / frame_shape[0]
        pygame.draw.rect(screen, (0, 200, 255),
                          (x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy), 1)

    if now < state.caught_flash_until:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 0, 0, 90))
        screen.blit(overlay, (0, 0))
        caught_label = font_big.render("SPOTTED!", True, (255, 255, 255))
        screen.blit(caught_label, (WIDTH / 2 - caught_label.get_width() / 2, HEIGHT / 2 - 30))

    if state.won or state.lost:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        if state.won:
            msg = font_big.render("HEIST COMPLETE!", True, (80, 255, 120))
        else:
            msg = font_big.render("CAUGHT! GAME OVER", True, (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 80))
        hint = font_small.render("Press R to play again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    running = True
    last_time = time.time()
    frame_count = 0
    cached_bbox = None
    prev_pts = None

    while running:
        now = time.time()
        dt = now - last_time
        last_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r and (state.won or state.lost):
                    state.reset()
                    prev_pts = None

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % 5 == 0:
            cached_bbox = get_person_bbox(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        person_seen = False
        move_amt = 0.0
        if results.pose_landmarks:
            person_seen = True
            cur_pts = get_key_points(results.pose_landmarks.landmark)
            move_amt = movement_amount(prev_pts, cur_pts)
            prev_pts = cur_pts
        else:
            prev_pts = None

        velocity = update_game(move_amt, person_seen, dt) or 0.0

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        screen.blit(dim, (0, 0))

        draw_hud(person_seen, cached_bbox, frame.shape, velocity)

        title = font_small.render(
            "STEALTH FREEZE - move when the guard's away, freeze when they turn!",
            True, (255, 255, 255))
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 90))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()