
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

TOTAL_TARGETS = 30
SPAWN_LEAD_TIME = 1.8         # seconds from spawn to hit-time (ring closing time)
HIT_WINDOW_PERFECT = 0.10
HIT_WINDOW_GOOD = 0.25
TARGET_RADIUS = 55

LEFT_X = WIDTH * 0.28
RIGHT_X = WIDTH * 0.72
TARGET_Y = HEIGHT * 0.5

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rhythm Punch")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 56, bold=True)
font_med = pygame.font.SysFont("arial", 30, bold=True)
font_small = pygame.font.SysFont("arial", 20)


def make_click_sound(freq=880, ms=60):
    sr = 22050
    n = int(sr * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(freq * t * 2 * np.pi)
    fade = np.linspace(1, 0, n)
    tone = (tone * fade * 32767).astype(np.int16)
    stereo = np.column_stack([tone, tone])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


snd_perfect = make_click_sound(1200, 80)
snd_good = make_click_sound(700, 80)
snd_miss = make_click_sound(220, 120)

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


def get_wrist_positions(landmarks):
    """Returns dict with 'left' and 'right' wrist (x,y) in normalized coords."""
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    return {"left": (lw.x, lw.y), "right": (rw.x, rw.y)}


# ----------------------------------------------------------------------------
# TARGET / GAME STATE
# ----------------------------------------------------------------------------
class Target:
    def __init__(self, side, spawn_time):
        self.side = side  # "left" or "right"
        self.spawn_time = spawn_time
        self.hit_time = spawn_time + SPAWN_LEAD_TIME
        self.x = LEFT_X if side == "left" else RIGHT_X
        self.y = TARGET_Y + random.uniform(-80, 80)
        self.resolved = False
        self.result = None  # "perfect" / "good" / "miss"


class GameState:
    def __init__(self):
        self.start_time = time.time()
        self.next_spawn_index = 0
        self.targets = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        self.result_banner = ""
        self.result_banner_until = 0
        self.game_over = False
        self.spawn_schedule = [i * 1.3 + 1.0 for i in range(TOTAL_TARGETS)]

    def reset(self):
        self.__init__()


state = GameState()


def spawn_targets():
    elapsed = time.time() - state.start_time
    while (state.next_spawn_index < len(state.spawn_schedule)
           and elapsed >= state.spawn_schedule[state.next_spawn_index]):
        side = random.choice(["left", "right"])
        state.targets.append(Target(side, time.time()))
        state.next_spawn_index += 1


def resolve_targets(wrist_positions, wrist_speed):
    now = time.time()
    for t in state.targets:
        if t.resolved:
            continue
        dt_hit = now - t.hit_time
        if abs(dt_hit) <= HIT_WINDOW_GOOD:
            wx, wy = wrist_positions[t.side]
            px, py = wx * WIDTH, wy * HEIGHT
            dist = math.hypot(px - t.x, py - t.y)
            fast_enough = wrist_speed[t.side] > 0.35  # normalized units/sec
            if dist <= TARGET_RADIUS and fast_enough:
                if abs(dt_hit) <= HIT_WINDOW_PERFECT:
                    t.result = "perfect"
                    state.score += 100
                    snd_perfect.play()
                else:
                    t.result = "good"
                    state.score += 50
                    snd_good.play()
                t.resolved = True
                state.combo += 1
                state.max_combo = max(state.max_combo, state.combo)
                state.result_banner = t.result.upper()
                state.result_banner_until = now + 0.5
        if not t.resolved and dt_hit > HIT_WINDOW_GOOD:
            t.result = "miss"
            t.resolved = True
            state.combo = 0
            snd_miss.play()
            state.result_banner = "MISS"
            state.result_banner_until = now + 0.5

    state.targets = [t for t in state.targets if not (t.resolved and now - t.hit_time > 1.0)]

    if (state.next_spawn_index >= len(state.spawn_schedule)
            and all(t.resolved for t in state.targets)):
        state.game_over = True


def draw_targets():
    now = time.time()
    for t in state.targets:
        progress = (now - t.spawn_time) / SPAWN_LEAD_TIME
        progress = max(0.0, min(1.2, progress))
        ring_radius = int(TARGET_RADIUS * 3 * (1 - min(progress, 1.0)) + TARGET_RADIUS)
        color = (0, 200, 255) if t.side == "left" else (255, 140, 0)

        if not t.resolved:
            pygame.draw.circle(screen, color, (int(t.x), int(t.y)), TARGET_RADIUS, 4)
            pygame.draw.circle(screen, (255, 255, 255), (int(t.x), int(t.y)), ring_radius, 2)
        else:
            fade_color = {"perfect": (80, 255, 120), "good": (255, 220, 60),
                          "miss": (255, 70, 70)}[t.result]
            pygame.draw.circle(screen, fade_color, (int(t.x), int(t.y)), TARGET_RADIUS, 6)

        hand_label = font_small.render("L" if t.side == "left" else "R", True, (255, 255, 255))
        screen.blit(hand_label, (t.x - hand_label.get_width() / 2, t.y - hand_label.get_height() / 2))


def draw_hud(wrist_positions, person_seen, cached_bbox, frame_shape):
    if person_seen and wrist_positions:
        for side, (nx, ny) in wrist_positions.items():
            px, py = nx * WIDTH, ny * HEIGHT
            color = (0, 200, 255) if side == "left" else (255, 140, 0)
            pygame.draw.circle(screen, color, (int(px), int(py)), 14)
            pygame.draw.circle(screen, (255, 255, 255), (int(px), int(py)), 14, 2)
    else:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, 40))

    if cached_bbox:
        x1, y1, x2, y2 = cached_bbox
        sx = WIDTH / frame_shape[1]
        sy = HEIGHT / frame_shape[0]
        pygame.draw.rect(screen, (0, 200, 255),
                          (x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy), 1)

    score_label = font_med.render(f"Score: {state.score}", True, (255, 255, 255))
    screen.blit(score_label, (WIDTH - score_label.get_width() - 20, 20))
    combo_label = font_small.render(f"Combo: {state.combo}  (Best: {state.max_combo})",
                                     True, (255, 255, 255))
    screen.blit(combo_label, (WIDTH - combo_label.get_width() - 20, 60))

    remaining = max(0, len(state.spawn_schedule) - state.next_spawn_index) + \
        sum(1 for t in state.targets if not t.resolved)
    prog_label = font_small.render(f"Targets left: {remaining}", True, (255, 255, 255))
    screen.blit(prog_label, (20, 20))

    if time.time() < state.result_banner_until and state.result_banner:
        colors = {"PERFECT": (80, 255, 120), "GOOD": (255, 220, 60), "MISS": (255, 70, 70)}
        banner = font_big.render(state.result_banner, True,
                                  colors.get(state.result_banner, (255, 255, 255)))
        screen.blit(banner, (WIDTH / 2 - banner.get_width() / 2, HEIGHT * 0.2))

    if state.game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        msg = font_big.render("SESSION COMPLETE", True, (255, 255, 255))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 100))
        sc = font_med.render(f"Final Score: {state.score}   Best Combo: {state.max_combo}",
                              True, (255, 255, 255))
        screen.blit(sc, (WIDTH / 2 - sc.get_width() / 2, HEIGHT / 2 - 20))
        hint = font_small.render("Press R to play again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 40))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    running = True
    last_time = time.time()
    frame_count = 0
    cached_bbox = None
    prev_wrist = None
    prev_t = time.time()

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

        frame_count += 1
        if frame_count % 5 == 0:
            cached_bbox = get_person_bbox(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        wrist_positions = None
        wrist_speed = {"left": 0.0, "right": 0.0}
        person_seen = False
        if results.pose_landmarks:
            wrist_positions = get_wrist_positions(results.pose_landmarks.landmark)
            person_seen = True
            time_delta = max(1e-3, now - prev_t)
            if prev_wrist:
                for side in ("left", "right"):
                    dx = wrist_positions[side][0] - prev_wrist[side][0]
                    dy = wrist_positions[side][1] - prev_wrist[side][1]
                    wrist_speed[side] = math.hypot(dx, dy) / time_delta
            prev_wrist = wrist_positions
            prev_t = now

        if not state.game_over:
            spawn_targets()
            if person_seen:
                resolve_targets(wrist_positions, wrist_speed)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        screen.blit(dim, (0, 0))

        draw_targets()
        draw_hud(wrist_positions, person_seen, cached_bbox, frame.shape)

        title = font_small.render("RHYTHM PUNCH - punch the target as the ring closes!",
                                   True, (255, 255, 255))
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()