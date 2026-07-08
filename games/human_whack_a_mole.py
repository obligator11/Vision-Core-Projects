

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

GRID_ROWS, GRID_COLS = 3, 4
HOLE_RADIUS = 55
GAME_DURATION = 60.0          # seconds
MOLE_UP_TIME_RANGE = (0.7, 1.4)
MOLE_SPAWN_GAP_RANGE = (0.3, 1.0)
MAX_ACTIVE_MOLES = 3

# Body landmarks that count as "mallets" (hands + feet)
MALLET_LANDMARKS = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE"]

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Human Whack-a-Mole")
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


def get_mallet_points(landmarks):
    """Returns list of (x_px, y_px) for hands/feet in screen space."""
    pts = []
    for name in MALLET_LANDMARKS:
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        if lm.visibility > 0.4:
            pts.append((lm.x * WIDTH, lm.y * HEIGHT))
    return pts


def build_hole_grid():
    margin_x, margin_y = 130, 150
    usable_w = WIDTH - 2 * margin_x
    usable_h = HEIGHT - 2 * margin_y - 40
    holes = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = margin_x + usable_w * (c + 0.5) / GRID_COLS
            y = margin_y + usable_h * (r + 0.5) / GRID_ROWS
            holes.append({"pos": (x, y), "state": "down", "up_since": 0, "up_duration": 0})
    return holes


# ----------------------------------------------------------------------------
# GAME STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.holes = build_hole_grid()
        self.start_time = time.time()
        self.next_spawn_time = time.time() + random.uniform(*MOLE_SPAWN_GAP_RANGE)
        self.score = 0
        self.hits = 0
        self.misses = 0
        self.game_over = False
        self.flash_hole_idx = None
        self.flash_until = 0

    def reset(self):
        self.__init__()


state = GameState()


def update_moles():
    now = time.time()
    elapsed = now - state.start_time
    if elapsed >= GAME_DURATION:
        state.game_over = True
        return

    active = sum(1 for h in state.holes if h["state"] == "up")

    # pop new moles
    if now >= state.next_spawn_time and active < MAX_ACTIVE_MOLES:
        down_holes = [i for i, h in enumerate(state.holes) if h["state"] == "down"]
        if down_holes:
            idx = random.choice(down_holes)
            state.holes[idx]["state"] = "up"
            state.holes[idx]["up_since"] = now
            state.holes[idx]["up_duration"] = random.uniform(*MOLE_UP_TIME_RANGE)
        state.next_spawn_time = now + random.uniform(*MOLE_SPAWN_GAP_RANGE)

    # duck moles whose time is up (missed)
    for h in state.holes:
        if h["state"] == "up" and now - h["up_since"] >= h["up_duration"]:
            h["state"] = "down"
            state.misses += 1


def check_hits(mallet_points):
    for i, h in enumerate(state.holes):
        if h["state"] != "up":
            continue
        hx, hy = h["pos"]
        for (px, py) in mallet_points:
            dist = ((px - hx) ** 2 + (py - hy) ** 2) ** 0.5
            if dist <= HOLE_RADIUS:
                h["state"] = "down"
                state.hits += 1
                state.score += 100
                state.flash_hole_idx = i
                state.flash_until = time.time() + 0.25
                break


def draw_holes():
    now = time.time()
    for i, h in enumerate(state.holes):
        x, y = h["pos"]
        pygame.draw.ellipse(screen, (40, 25, 10), (x - HOLE_RADIUS, y - HOLE_RADIUS * 0.5,
                                                     HOLE_RADIUS * 2, HOLE_RADIUS))
        if h["state"] == "up":
            remaining = 1 - (now - h["up_since"]) / h["up_duration"]
            color = (90, 200, 90) if remaining > 0.3 else (230, 150, 40)
            pygame.draw.circle(screen, color, (int(x), int(y - 15)), HOLE_RADIUS - 10)
            pygame.draw.circle(screen, (20, 20, 20), (int(x), int(y - 15)), HOLE_RADIUS - 10, 3)
            eyes_y = y - 25
            pygame.draw.circle(screen, (0, 0, 0), (int(x - 12), int(eyes_y)), 4)
            pygame.draw.circle(screen, (0, 0, 0), (int(x + 12), int(eyes_y)), 4)
        else:
            pygame.draw.ellipse(screen, (20, 10, 5), (x - HOLE_RADIUS + 8, y - 10,
                                                        (HOLE_RADIUS - 8) * 2, 20))

        if state.flash_hole_idx == i and now < state.flash_until:
            pygame.draw.circle(screen, (255, 255, 0), (int(x), int(y - 15)), HOLE_RADIUS + 10, 4)


def draw_hud(mallet_points, person_seen, cached_bbox, frame_shape):
    for (px, py) in mallet_points:
        pygame.draw.circle(screen, (0, 220, 255), (int(px), int(py)), 12, 3)

    if not person_seen:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, 40))

    if cached_bbox:
        x1, y1, x2, y2 = cached_bbox
        sx = WIDTH / frame_shape[1]
        sy = HEIGHT / frame_shape[0]
        pygame.draw.rect(screen, (0, 200, 255),
                          (x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy), 1)

    elapsed = time.time() - state.start_time
    time_left = max(0, GAME_DURATION - elapsed)
    timer_label = font_med.render(f"Time: {int(time_left)}s", True, (255, 255, 255))
    screen.blit(timer_label, (20, 20))

    score_label = font_med.render(f"Score: {state.score}", True, (255, 255, 255))
    screen.blit(score_label, (WIDTH - score_label.get_width() - 20, 20))

    stats_label = font_small.render(f"Hits: {state.hits}   Misses: {state.misses}",
                                     True, (255, 255, 255))
    screen.blit(stats_label, (WIDTH / 2 - stats_label.get_width() / 2, 20))

    if state.game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        msg = font_big.render("TIME'S UP!", True, (255, 220, 60))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 100))
        acc = state.hits / max(1, state.hits + state.misses) * 100
        sc = font_med.render(f"Score: {state.score}   Accuracy: {acc:.0f}%", True, (255, 255, 255))
        screen.blit(sc, (WIDTH / 2 - sc.get_width() / 2, HEIGHT / 2 - 20))
        hint = font_small.render("Press R to play again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 40))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    running = True
    frame_count = 0
    cached_bbox = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r and state.game_over:
                    state.reset()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        frame_count += 1
        if frame_count % 5 == 0:
            cached_bbox = get_person_bbox(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        mallet_points = []
        person_seen = False
        if results.pose_landmarks:
            mallet_points = get_mallet_points(results.pose_landmarks.landmark)
            person_seen = True

        if not state.game_over:
            update_moles()
            if mallet_points:
                check_hits(mallet_points)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 90))
        screen.blit(dim, (0, 0))

        draw_holes()
        draw_hud(mallet_points, person_seen, cached_bbox, frame.shape)

        title = font_small.render("HUMAN WHACK-A-MOLE - smack the moles with hands or feet!",
                                   True, (255, 255, 255))
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()