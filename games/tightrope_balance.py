
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

BEAM_Y = HEIGHT - 160
BEAM_START_WIDTH = 320
BEAM_MIN_WIDTH = 90
BEAM_SHRINK_PER_SEC = 1.6         # beam narrows over time

WOBBLE_MAX = 100.0
WOBBLE_DRAIN_IN_ZONE = 35.0       # per second, meter refills while balanced
WOBBLE_FILL_OUT_ZONE = 55.0       # per second, meter fills while off-balance

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tightrope Balance")
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
    """Convert an OpenCV BGR frame into a pygame Surface sized to WIDTH x HEIGHT."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def get_person_bbox(frame_bgr):
    """Run YOLO to find the largest 'person' box in the frame (or None)."""
    results = yolo_model.predict(frame_bgr, classes=[0], verbose=False, conf=0.4)
    best = None
    best_area = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
    return best


def get_body_lean_x(landmarks, frame_w):
    """
    Returns the normalized horizontal position (0..1) of the body's
    center of mass, using shoulders + hips.
    """
    ids = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
           mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    xs = [landmarks[i.value].x for i in ids]
    return sum(xs) / len(xs)


# ----------------------------------------------------------------------------
# GAME STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.start_time = time.time()
        self.beam_width = BEAM_START_WIDTH
        self.wind_phase = random.uniform(0, math.tau)
        self.wobble = WOBBLE_MAX * 0.4
        self.score = 0.0
        self.game_over = False
        self.person_present = False

    def reset(self):
        self.__init__()


state = GameState()


def update_game(dt, body_x_norm, person_seen):
    if state.game_over:
        return

    state.person_present = person_seen
    elapsed = time.time() - state.start_time

    # Difficulty ramps: beam narrows, wind gusts more over time
    state.beam_width = max(BEAM_MIN_WIDTH, BEAM_START_WIDTH - BEAM_SHRINK_PER_SEC * elapsed)
    wind_speed = 0.6 + elapsed * 0.02
    gust_strength = min(90, 40 + elapsed * 1.5)
    target_offset = math.sin(elapsed * wind_speed + state.wind_phase) * gust_strength

    beam_center_x = WIDTH / 2 + target_offset

    if person_seen:
        player_x = body_x_norm * WIDTH
        distance = abs(player_x - beam_center_x)
        if distance <= state.beam_width / 2:
            state.wobble -= WOBBLE_DRAIN_IN_ZONE * dt
            state.score += dt * (10 + elapsed * 0.3)
        else:
            over_by = distance - state.beam_width / 2
            state.wobble += (WOBBLE_FILL_OUT_ZONE + over_by * 0.3) * dt
    else:
        # no person detected -> treat as unbalanced
        state.wobble += WOBBLE_FILL_OUT_ZONE * dt

    state.wobble = max(0.0, min(WOBBLE_MAX, state.wobble))

    if state.wobble >= WOBBLE_MAX:
        state.game_over = True

    return beam_center_x


def draw_hud(beam_center_x, body_x_norm, person_seen):
    # Beam
    beam_rect = pygame.Rect(0, 0, state.beam_width, 14)
    beam_rect.center = (beam_center_x, BEAM_Y)
    pygame.draw.rect(screen, (255, 210, 60), beam_rect, border_radius=7)
    pygame.draw.rect(screen, (120, 80, 0), beam_rect, width=3, border_radius=7)

    # Danger edges
    pygame.draw.line(screen, (255, 60, 60), (0, BEAM_Y + 60), (WIDTH, BEAM_Y + 60), 2)

    # Player marker
    if person_seen:
        px = body_x_norm * WIDTH
        color = (60, 220, 90) if abs(px - beam_center_x) <= state.beam_width / 2 else (230, 60, 60)
        pygame.draw.circle(screen, color, (int(px), BEAM_Y - 20), 16)
        pygame.draw.circle(screen, (255, 255, 255), (int(px), BEAM_Y - 20), 16, 2)
    else:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, 40))

    # Wobble meter
    meter_w, meter_h = 260, 26
    meter_x, meter_y = 20, 20
    pygame.draw.rect(screen, (40, 40, 40), (meter_x, meter_y, meter_w, meter_h), border_radius=6)
    fill_w = int(meter_w * (state.wobble / WOBBLE_MAX))
    fill_color = (250, 80, 80) if state.wobble > WOBBLE_MAX * 0.7 else (250, 210, 60)
    pygame.draw.rect(screen, fill_color, (meter_x, meter_y, fill_w, meter_h), border_radius=6)
    pygame.draw.rect(screen, (255, 255, 255), (meter_x, meter_y, meter_w, meter_h), 2, border_radius=6)
    label = font_small.render("BALANCE (fills = falling)", True, (255, 255, 255))
    screen.blit(label, (meter_x, meter_y + meter_h + 4))

    # Score
    score_label = font_med.render(f"Score: {int(state.score)}", True, (255, 255, 255))
    screen.blit(score_label, (WIDTH - score_label.get_width() - 20, 20))

    if state.game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        msg = font_big.render("YOU FELL!", True, (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 80))
        sc = font_med.render(f"Final Score: {int(state.score)}", True, (255, 255, 255))
        screen.blit(sc, (WIDTH / 2 - sc.get_width() / 2, HEIGHT / 2))
        hint = font_small.render("Press R to try again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 60))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    running = True
    last_time = time.time()
    frame_count = 0
    cached_bbox = None

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
                elif event.key == pygame.K_r and state.game_over:
                    state.reset()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        # Run YOLO every 3rd frame for performance, reuse bbox otherwise
        frame_count += 1
        if frame_count % 3 == 0:
            cached_bbox = get_person_bbox(frame)
        person_seen_yolo = cached_bbox is not None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        body_x_norm = 0.5
        person_seen = False
        if results.pose_landmarks:
            body_x_norm = get_body_lean_x(results.pose_landmarks.landmark, frame.shape[1])
            person_seen = True
        else:
            person_seen = person_seen_yolo  # fall back to YOLO presence only

        beam_center_x = update_game(dt, body_x_norm, person_seen)
        if beam_center_x is None:
            beam_center_x = WIDTH / 2

        # Draw camera feed as background
        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))

        # dim overlay so HUD pops
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 60))
        screen.blit(dim, (0, 0))

        if cached_bbox:
            x1, y1, x2, y2 = cached_bbox
            sx = WIDTH / frame.shape[1]
            sy = HEIGHT / frame.shape[0]
            pygame.draw.rect(screen, (0, 200, 255),
                              (x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy), 2)

        draw_hud(beam_center_x, body_x_norm, person_seen)

        title = font_small.render("TIGHTROPE BALANCE - lean to stay on the beam!", True, (255, 255, 255))
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()