

import random
import sys
import time

import cv2
import numpy as np
import pygame
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1000, 700
FPS = 30
CAM_INDEX = 0

TOTAL_ROUNDS = 10
STARTING_LIVES = 3
ROUND_TIME = 9.0
CONFIRM_HOLD_TIME = 0.6        # seconds the correct object must stay in the zone
DETECTION_CONF = 0.45
YOLO_EVERY_N_FRAMES = 2        # throttle YOLO for performance

ZONE_W, ZONE_H = 340, 340      # capture zone size (px, in game coords)

# Curated to common desk/household items YOLOv8 (COCO classes) recognizes well.
TARGET_CLASS_POOL = [
    "cup", "bottle", "cell phone", "book", "remote", "scissors",
    "apple", "banana", "clock", "keyboard", "mouse", "fork", "spoon",
]

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Object Heist")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 56, bold=True)
font_med = pygame.font.SysFont("arial", 30, bold=True)
font_small = pygame.font.SysFont("arial", 20)

print("Loading YOLOv8n (first run downloads weights)...")
yolo_model = YOLO("yolov8n.pt")

# Only keep target classes the loaded model actually knows about
available_names = set(yolo_model.names.values())
TARGET_CLASSES = [c for c in TARGET_CLASS_POOL if c in available_names]
if not TARGET_CLASSES:
    print("WARNING: none of the curated target classes exist in this YOLO model's class list.")
    TARGET_CLASSES = list(available_names)[:10]

TARGET_CLASS_IDS = [cid for cid, name in yolo_model.names.items() if name in TARGET_CLASSES]

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


def detect_objects(frame_bgr):
    """Runs YOLO restricted to our curated class list, returns list of dicts."""
    results = yolo_model.predict(frame_bgr, classes=TARGET_CLASS_IDS, verbose=False,
                                  conf=DETECTION_CONF)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"name": name, "conf": conf, "bbox": (x1, y1, x2, y2)})
    return detections


def bbox_center_in_zone(bbox, frame_shape, zone_rect):
    x1, y1, x2, y2 = bbox
    sx = WIDTH / frame_shape[1]
    sy = HEIGHT / frame_shape[0]
    cx = (x1 + x2) / 2 * sx
    cy = (y1 + y2) / 2 * sy
    return zone_rect.collidepoint(cx, cy)


def get_zone_rect():
    return pygame.Rect(WIDTH // 2 - ZONE_W // 2, HEIGHT // 2 - ZONE_H // 2 + 20, ZONE_W, ZONE_H)


# ----------------------------------------------------------------------------
# GAME STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.round_num = 1
        self.target = random.choice(TARGET_CLASSES)
        self.round_start = time.time()
        self.hold_start = None
        self.score = 0
        self.lives = STARTING_LIVES
        self.game_over = False
        self.won = False
        self.flash_until = 0
        self.flash_text = ""
        self.flash_color = (255, 255, 255)

    def next_round(self, success):
        if success:
            self.round_num += 1
            if self.round_num > TOTAL_ROUNDS:
                self.game_over = True
                self.won = True
                return
        else:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                self.won = False
                return
        self.target = random.choice(TARGET_CLASSES)
        self.round_start = time.time()
        self.hold_start = None

    def reset(self):
        self.__init__()


state = GameState()


def update_game(detections, frame_shape):
    if state.game_over:
        return
    now = time.time()
    remaining = ROUND_TIME - (now - state.round_start)

    zone_rect = get_zone_rect()
    correct_in_zone = False
    for d in detections:
        if d["name"] == state.target and bbox_center_in_zone(d["bbox"], frame_shape, zone_rect):
            correct_in_zone = True
            break

    if correct_in_zone:
        if state.hold_start is None:
            state.hold_start = now
        elif now - state.hold_start >= CONFIRM_HOLD_TIME:
            gained = int(100 + max(0, remaining) * 10)
            state.score += gained
            state.flash_text = f"GOT IT! +{gained}"
            state.flash_color = (80, 255, 120)
            state.flash_until = now + 0.8
            state.next_round(success=True)
    else:
        state.hold_start = None

    if not state.game_over and remaining <= 0:
        state.flash_text = "TOO SLOW!"
        state.flash_color = (255, 90, 90)
        state.flash_until = now + 0.8
        state.next_round(success=False)


def draw_hud(detections, frame_shape):
    now = time.time()
    zone_rect = get_zone_rect()

    # capture zone
    hold_progress = 0.0
    if state.hold_start:
        hold_progress = min(1.0, (now - state.hold_start) / CONFIRM_HOLD_TIME)
    zone_color = (80, 255, 120) if hold_progress > 0 else (0, 200, 255)
    pygame.draw.rect(screen, zone_color, zone_rect, width=4, border_radius=12)
    if hold_progress > 0:
        fill_rect = zone_rect.inflate(-10, -10)
        fill_h = int(fill_rect.height * hold_progress)
        fill_surf = pygame.Surface((fill_rect.width, fill_h), pygame.SRCALPHA)
        fill_surf.fill((80, 255, 120, 90))
        screen.blit(fill_surf, (fill_rect.x, fill_rect.bottom - fill_h))

    zone_label = font_small.render("CAPTURE ZONE", True, (255, 255, 255))
    screen.blit(zone_label, (zone_rect.centerx - zone_label.get_width() / 2, zone_rect.top - 26))

    # detection boxes
    sx = WIDTH / frame_shape[1]
    sy = HEIGHT / frame_shape[0]
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        rect = pygame.Rect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
        is_target = d["name"] == state.target
        color = (80, 255, 120) if is_target else (150, 150, 150)
        pygame.draw.rect(screen, color, rect, 2)
        label = font_small.render(f"{d['name']} {d['conf']:.2f}", True, color)
        screen.blit(label, (rect.x, max(0, rect.y - 20)))

    # prompt banner
    prompt = font_med.render(f"STEAL: {state.target.upper()}", True, (255, 220, 60))
    screen.blit(prompt, (WIDTH / 2 - prompt.get_width() / 2, 20))

    # timer
    remaining = max(0.0, ROUND_TIME - (now - state.round_start))
    timer_color = (255, 255, 255) if remaining > 3 else (255, 90, 90)
    timer_label = font_med.render(f"{remaining:0.1f}s", True, timer_color)
    screen.blit(timer_label, (WIDTH / 2 - timer_label.get_width() / 2, 60))

    # score / round / lives
    score_label = font_small.render(f"Score: {state.score}", True, (255, 255, 255))
    screen.blit(score_label, (20, 20))
    round_label = font_small.render(f"Round {state.round_num}/{TOTAL_ROUNDS}", True, (255, 255, 255))
    screen.blit(round_label, (20, 45))
    for i in range(STARTING_LIVES):
        color = (230, 60, 60) if i < state.lives else (70, 70, 70)
        pygame.draw.circle(screen, color, (WIDTH - 30 - i * 34, 30), 12)

    if now < state.flash_until:
        flash = font_big.render(state.flash_text, True, state.flash_color)
        screen.blit(flash, (WIDTH / 2 - flash.get_width() / 2, HEIGHT / 2 - 200))

    if state.game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        if state.won:
            msg = font_big.render("HEIST SUCCESSFUL!", True, (80, 255, 120))
        else:
            msg = font_big.render("CAUGHT! GAME OVER", True, (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 80))
        sc = font_med.render(f"Final Score: {state.score}", True, (255, 255, 255))
        screen.blit(sc, (WIDTH / 2 - sc.get_width() / 2, HEIGHT / 2))
        hint = font_small.render("Press R to play again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 40))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    running = True
    frame_count = 0
    cached_detections = []

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
        if frame_count % YOLO_EVERY_N_FRAMES == 0:
            cached_detections = detect_objects(frame)

        if not state.game_over:
            update_game(cached_detections, frame.shape)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 60))
        screen.blit(dim, (0, 0))

        draw_hud(cached_detections, frame.shape)

        title = font_small.render("OBJECT HEIST - grab the named object and hold it in the zone!",
                                   True, (255, 255, 255))
        screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()