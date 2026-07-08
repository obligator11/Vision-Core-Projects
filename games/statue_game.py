"""
STATUE GAME (RED LIGHT, GREEN LIGHT)
=====================================
Move during GREEN LIGHT to make progress. Freeze completely during
RED LIGHT - any movement caught by the camera eliminates a life!

Stack:
- OpenCV        -> webcam capture + frame-difference motion detection
- MediaPipe     -> body landmark tracking (for precise movement measurement)
- YOLOv8 (Ultralytics) -> person-presence detection / bounding box overlay
- Pygame        -> game window, traffic-light state machine, HUD

Install:
    pip install opencv-python mediapipe pygame ultralytics numpy

Run:
    python 4_statue_game_red_light_green_light.py

Controls:
    Q or ESC -> quit
    R        -> restart after game over
"""

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

GREEN_TIME_RANGE = (2.0, 4.5)
RED_TIME_RANGE = (2.5, 5.0)
STARTING_LIVES = 3
PROGRESS_GOAL = 100.0
MOVEMENT_THRESHOLD = 0.015     # normalized landmark-delta threshold to count as "moving"
GRACE_PERIOD = 0.5             # seconds after switching to RED before penalties start

KEY_LANDMARKS = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_SHOULDER", "RIGHT_SHOULDER",
                  "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "NOSE"]

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Statue Game - Red Light, Green Light")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 64, bold=True)
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
    if prev_pts is None or cur_pts is None:
        return 0.0
    total = 0.0
    for name in KEY_LANDMARKS:
        px, py = prev_pts[name]
        cx, cy = cur_pts[name]
        total += ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
    return total / len(KEY_LANDMARKS)


# ----------------------------------------------------------------------------
# GAME STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.phase = "GREEN"
        self.phase_start = time.time()
        self.phase_duration = random.uniform(*GREEN_TIME_RANGE)
        self.progress = 0.0
        self.lives = STARTING_LIVES
        self.caught_flash_until = 0
        self.won = False
        self.lost = False

    def reset(self):
        self.__init__()


state = GameState()


def switch_phase():
    now = time.time()
    if state.phase == "GREEN":
        state.phase = "RED"
        state.phase_duration = random.uniform(*RED_TIME_RANGE)
    else:
        state.phase = "GREEN"
        state.phase_duration = random.uniform(*GREEN_TIME_RANGE)
    state.phase_start = now


def update_game(move_amt, person_seen):
    if state.won or state.lost:
        return
    now = time.time()
    if now - state.phase_start >= state.phase_duration:
        switch_phase()

    if not person_seen:
        return

    time_in_phase = now - state.phase_start

    if state.phase == "GREEN":
        # progress driven by intentional movement
        state.progress += min(move_amt, 0.08) * 40
        state.progress = min(PROGRESS_GOAL, state.progress)
        if state.progress >= PROGRESS_GOAL:
            state.won = True
    else:  # RED
        if time_in_phase > GRACE_PERIOD and move_amt > MOVEMENT_THRESHOLD:
            state.lives -= 1
            state.caught_flash_until = now + 0.6
            # reset phase timer to give a fresh red-light window (harsher)
            state.phase_start = now
            if state.lives <= 0:
                state.lost = True


def draw_hud(person_seen, cached_bbox, frame_shape, move_amt):
    now = time.time()

    # Traffic light
    light_color = (60, 220, 90) if state.phase == "GREEN" else (230, 60, 60)
    pygame.draw.circle(screen, (30, 30, 30), (WIDTH - 70, 70), 46)
    pygame.draw.circle(screen, light_color, (WIDTH - 70, 70), 38)
    phase_label = font_small.render(state.phase + " LIGHT", True, (255, 255, 255))
    screen.blit(phase_label, (WIDTH - 70 - phase_label.get_width() / 2, 120))

    time_left = max(0, state.phase_duration - (now - state.phase_start))
    tl_label = font_small.render(f"{time_left:0.1f}s", True, (255, 255, 255))
    screen.blit(tl_label, (WIDTH - 70 - tl_label.get_width() / 2, 145))

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 20, HEIGHT - 60, WIDTH - 40, 30
    pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h), border_radius=8)
    fill_w = int(bar_w * (state.progress / PROGRESS_GOAL))
    pygame.draw.rect(screen, (60, 220, 90), (bar_x, bar_y, fill_w, bar_h), border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=8)
    prog_label = font_small.render("PROGRESS TO FINISH LINE", True, (255, 255, 255))
    screen.blit(prog_label, (bar_x, bar_y - 24))

    # Lives
    for i in range(STARTING_LIVES):
        color = (230, 60, 60) if i < state.lives else (70, 70, 70)
        pygame.draw.circle(screen, color, (30 + i * 34, 30), 12)

    if not person_seen:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, 40))

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
        caught_label = font_big.render("CAUGHT MOVING!", True, (255, 255, 255))
        screen.blit(caught_label, (WIDTH / 2 - caught_label.get_width() / 2, HEIGHT / 2 - 30))

    if state.won or state.lost:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        if state.won:
            msg = font_big.render("YOU WIN!", True, (80, 255, 120))
        else:
            msg = font_big.render("ELIMINATED!", True, (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 80))
        hint = font_small.render("Press R to play again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 10))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    running = True
    frame_count = 0
    cached_bbox = None
    prev_pts = None

    while running:
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

        update_game(move_amt, person_seen)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        screen.blit(dim, (0, 0))

        draw_hud(person_seen, cached_bbox, frame.shape, move_amt)

        title = font_small.render(
            "STATUE GAME - move on GREEN, freeze completely on RED!", True, (255, 255, 255))
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 90))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()