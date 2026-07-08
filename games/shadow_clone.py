"""
SHADOW CLONE RACE
==================
A ghostly "shadow" stick figure strikes a pose. Race to copy it with your
own body before time runs out! Match enough poses in a row to win.

Stack:
- OpenCV        -> webcam capture
- MediaPipe     -> body landmark tracking + pose-similarity scoring
- YOLOv8 (Ultralytics) -> person-presence detection / bounding box overlay
- Pygame        -> game window, shadow stick-figure rendering, HUD

Install:
    pip install opencv-python mediapipe pygame ultralytics numpy

Run:
    python 5_shadow_clone_race.py

Controls:
    Q or ESC -> quit
    R        -> restart after game over
"""

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

TOTAL_ROUNDS = 8
TIME_PER_POSE = 6.0
MATCH_THRESHOLD = 0.11          # avg normalized landmark distance to count as "matched"
HOLD_TIME_TO_CONFIRM = 0.5      # must hold the match for this long

# Joints used for both drawing the shadow skeleton and scoring similarity.
# Values are normalized (0..1) offsets relative to a bounding box, describing
# a handful of iconic poses. (nose, l_sh, r_sh, l_elbow, r_elbow, l_wrist,
# r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle)
JOINT_NAMES = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
               "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
               "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]

POSE_TEMPLATES = [
    # T-pose (arms out)
    {"NOSE": (0.5, 0.05), "LEFT_SHOULDER": (0.35, 0.18), "RIGHT_SHOULDER": (0.65, 0.18),
     "LEFT_ELBOW": (0.15, 0.18), "RIGHT_ELBOW": (0.85, 0.18),
     "LEFT_WRIST": (0.0, 0.18), "RIGHT_WRIST": (1.0, 0.18),
     "LEFT_HIP": (0.4, 0.5), "RIGHT_HIP": (0.6, 0.5),
     "LEFT_KNEE": (0.4, 0.75), "RIGHT_KNEE": (0.6, 0.75),
     "LEFT_ANKLE": (0.4, 1.0), "RIGHT_ANKLE": (0.6, 1.0)},
    # Arms up (touchdown)
    {"NOSE": (0.5, 0.05), "LEFT_SHOULDER": (0.35, 0.18), "RIGHT_SHOULDER": (0.65, 0.18),
     "LEFT_ELBOW": (0.25, 0.0), "RIGHT_ELBOW": (0.75, 0.0),
     "LEFT_WRIST": (0.2, -0.2), "RIGHT_WRIST": (0.8, -0.2),
     "LEFT_HIP": (0.4, 0.5), "RIGHT_HIP": (0.6, 0.5),
     "LEFT_KNEE": (0.4, 0.75), "RIGHT_KNEE": (0.6, 0.75),
     "LEFT_ANKLE": (0.4, 1.0), "RIGHT_ANKLE": (0.6, 1.0)},
    # One leg up (flamingo)
    {"NOSE": (0.5, 0.05), "LEFT_SHOULDER": (0.35, 0.18), "RIGHT_SHOULDER": (0.65, 0.18),
     "LEFT_ELBOW": (0.2, 0.35), "RIGHT_ELBOW": (0.8, 0.35),
     "LEFT_WRIST": (0.1, 0.5), "RIGHT_WRIST": (0.9, 0.5),
     "LEFT_HIP": (0.4, 0.5), "RIGHT_HIP": (0.6, 0.5),
     "LEFT_KNEE": (0.55, 0.6), "RIGHT_KNEE": (0.6, 0.75),
     "LEFT_ANKLE": (0.5, 0.5), "RIGHT_ANKLE": (0.6, 1.0)},
    # Arms crossed low / lunge left
    {"NOSE": (0.42, 0.05), "LEFT_SHOULDER": (0.3, 0.2), "RIGHT_SHOULDER": (0.58, 0.18),
     "LEFT_ELBOW": (0.5, 0.3), "RIGHT_ELBOW": (0.4, 0.32),
     "LEFT_WRIST": (0.35, 0.4), "RIGHT_WRIST": (0.55, 0.4),
     "LEFT_HIP": (0.35, 0.5), "RIGHT_HIP": (0.6, 0.5),
     "LEFT_KNEE": (0.2, 0.75), "RIGHT_KNEE": (0.65, 0.7),
     "LEFT_ANKLE": (0.1, 1.0), "RIGHT_ANKLE": (0.65, 1.0)},
    # Star jump
    {"NOSE": (0.5, 0.02), "LEFT_SHOULDER": (0.35, 0.15), "RIGHT_SHOULDER": (0.65, 0.15),
     "LEFT_ELBOW": (0.15, -0.05), "RIGHT_ELBOW": (0.85, -0.05),
     "LEFT_WRIST": (0.05, -0.25), "RIGHT_WRIST": (0.95, -0.25),
     "LEFT_HIP": (0.4, 0.45), "RIGHT_HIP": (0.6, 0.45),
     "LEFT_KNEE": (0.2, 0.7), "RIGHT_KNEE": (0.8, 0.7),
     "LEFT_ANKLE": (0.05, 0.95), "RIGHT_ANKLE": (0.95, 0.95)},
]

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shadow Clone Race")
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


def normalize_landmarks(landmarks):
    """
    Normalize player landmarks into a 0..1 bounding-box space (like the
    templates), anchored on shoulders/hips so it's translation & scale
    invariant.
    """
    pts = {name: (landmarks[mp_pose.PoseLandmark[name].value].x,
                   landmarks[mp_pose.PoseLandmark[name].value].y)
           for name in JOINT_NAMES}

    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    norm = {}
    for name, (x, y) in pts.items():
        norm[name] = ((x - min_x) / span_x, (y - min_y) / span_y)
    return norm


def pose_distance(player_norm, template):
    total = 0.0
    for name in JOINT_NAMES:
        px, py = player_norm[name]
        tx, ty = template[name]
        total += math.hypot(px - tx, py - ty)
    return total / len(JOINT_NAMES)


def draw_shadow_skeleton(template, alpha=140):
    """Draws the target 'shadow' pose scaled into a region of the screen."""
    region_x, region_y, region_w, region_h = WIDTH - 300, 40, 260, 420
    shadow_surf = pygame.Surface((region_w, region_h), pygame.SRCALPHA)

    def to_px(name):
        nx, ny = template[name]
        return (nx * region_w, ny * region_h + 40)

    bones = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_SHOULDER", "LEFT_ELBOW"),
        ("LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        ("RIGHT_ELBOW", "RIGHT_WRIST"), ("LEFT_SHOULDER", "LEFT_HIP"),
        ("RIGHT_SHOULDER", "RIGHT_HIP"), ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
        ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ]
    for a, b in bones:
        pygame.draw.line(shadow_surf, (120, 120, 255, alpha), to_px(a), to_px(b), 8)
    nose_px = to_px("NOSE")
    pygame.draw.circle(shadow_surf, (120, 120, 255, alpha), (int(nose_px[0]), int(nose_px[1])), 22)

    pygame.draw.rect(screen, (30, 30, 60), (region_x - 10, region_y - 10, region_w + 20, region_h + 20),
                      border_radius=12)
    screen.blit(shadow_surf, (region_x, region_y))
    label = font_small.render("MATCH THIS POSE", True, (255, 255, 255))
    screen.blit(label, (region_x + region_w / 2 - label.get_width() / 2, region_y + region_h + 8))


# ----------------------------------------------------------------------------
# GAME STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.round_num = 1
        self.template = random.choice(POSE_TEMPLATES)
        self.round_start = time.time()
        self.hold_start = None
        self.score = 0
        self.game_over = False
        self.won = False
        self.last_distance = 1.0

    def next_round(self):
        self.round_num += 1
        if self.round_num > TOTAL_ROUNDS:
            self.game_over = True
            self.won = True
            return
        self.template = random.choice(POSE_TEMPLATES)
        self.round_start = time.time()
        self.hold_start = None

    def reset(self):
        self.__init__()


state = GameState()


def update_game(player_norm, person_seen):
    if state.game_over:
        return
    now = time.time()
    elapsed = now - state.round_start
    remaining = TIME_PER_POSE - elapsed

    if remaining <= 0:
        state.game_over = True
        state.won = False
        return

    if person_seen:
        dist = pose_distance(player_norm, state.template)
        state.last_distance = dist
        if dist <= MATCH_THRESHOLD:
            if state.hold_start is None:
                state.hold_start = now
            elif now - state.hold_start >= HOLD_TIME_TO_CONFIRM:
                state.score += int(200 * max(0.3, remaining / TIME_PER_POSE))
                state.next_round()
        else:
            state.hold_start = None


def draw_hud(player_norm, person_seen, cached_bbox, frame_shape):
    draw_shadow_skeleton(state.template)

    now = time.time()
    elapsed = now - state.round_start
    remaining = max(0, TIME_PER_POSE - elapsed)

    round_label = font_med.render(f"Round {state.round_num}/{TOTAL_ROUNDS}", True, (255, 255, 255))
    screen.blit(round_label, (20, 20))

    time_label = font_med.render(f"{remaining:0.1f}s", True,
                                  (255, 255, 255) if remaining > 2 else (255, 90, 90))
    screen.blit(time_label, (20, 60))

    score_label = font_med.render(f"Score: {state.score}", True, (255, 255, 255))
    screen.blit(score_label, (20, 100))

    if not person_seen:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2 - 130, 40))
    else:
        match_pct = max(0, 1 - state.last_distance / (MATCH_THRESHOLD * 2.5)) * 100
        match_pct = min(100, match_pct)
        bar_x, bar_y, bar_w, bar_h = 20, 140, 260, 22
        pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        fill_color = (60, 220, 90) if match_pct > 75 else (250, 210, 60)
        pygame.draw.rect(screen, fill_color, (bar_x, bar_y, int(bar_w * match_pct / 100), bar_h),
                          border_radius=6)
        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=6)
        match_label = font_small.render(f"Match: {match_pct:0.0f}%", True, (255, 255, 255))
        screen.blit(match_label, (bar_x, bar_y + bar_h + 4))

        if state.hold_start is not None:
            hold_pct = min(1.0, (now - state.hold_start) / HOLD_TIME_TO_CONFIRM)
            pygame.draw.circle(screen, (60, 220, 90), (WIDTH // 2, HEIGHT - 60), 30, 4)
            pygame.draw.arc(screen, (60, 220, 90),
                             (WIDTH // 2 - 30, HEIGHT - 90, 60, 60),
                             -math.pi / 2, -math.pi / 2 + hold_pct * 2 * math.pi, 6)

    if cached_bbox:
        x1, y1, x2, y2 = cached_bbox
        sx = WIDTH / frame_shape[1]
        sy = HEIGHT / frame_shape[0]
        pygame.draw.rect(screen, (0, 200, 255),
                          (x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy), 1)

    if state.game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        if state.won:
            msg = font_big.render("ALL POSES MATCHED!", True, (80, 255, 120))
        else:
            msg = font_big.render("TIME'S UP!", True, (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 80))
        sc = font_med.render(f"Score: {state.score}", True, (255, 255, 255))
        screen.blit(sc, (WIDTH / 2 - sc.get_width() / 2, HEIGHT / 2))
        hint = font_small.render("Press R to play again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 50))


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

        person_seen = False
        player_norm = None
        if results.pose_landmarks:
            person_seen = True
            player_norm = normalize_landmarks(results.pose_landmarks.landmark)

        if not state.game_over and person_seen:
            update_game(player_norm, person_seen)
        elif not state.game_over:
            update_game(None, False)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        screen.blit(dim, (0, 0))

        draw_hud(player_norm, person_seen, cached_bbox, frame.shape)

        title = font_small.render("SHADOW CLONE RACE - copy the shadow's pose and hold it!",
                                   True, (255, 255, 255))
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2 - 130, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()