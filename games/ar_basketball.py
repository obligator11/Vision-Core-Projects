import math
import random
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

THROW_VELOCITY_THRESHOLD = 1.1     # normalized units/sec to detect a release
THROW_COOLDOWN = 0.8
GRAVITY = 1600.0                   # px/sec^2 (screen-space, tuned for feel)
BALL_RADIUS = 16

HOOP_FRAC = (0.5, 0.22)            # hoop rim center as a fraction of the window
HOOP_WIDTH_FRAC = 0.14

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("AR Basketball Hoop")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 56, bold=True)
font_med = pygame.font.SysFont("arial", 28, bold=True)
font_small = pygame.font.SysFont("arial", 18)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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


def make_swish(ms=250):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    noise = np.random.uniform(-1, 1, n)
    env = np.sin(np.pi * t / (ms / 1000)) ** 2
    return to_sound(noise * env, volume=0.35)


def make_clank(ms=180):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * 500 * t) * np.exp(-t * 12) + \
        np.sin(2 * np.pi * 750 * t) * np.exp(-t * 15)
    return to_sound(tone, volume=0.5)


def make_cheer(ms=600):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    noise = np.random.uniform(-1, 1, n) * np.exp(-t * 2)
    return to_sound(noise, volume=0.35)


snd_swish = make_swish()
snd_clank = make_clank()
snd_cheer = make_cheer()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def wrist_pos(landmarks):
    lm = landmarks[mp_hands.HandLandmark.WRIST]
    return lm.x, lm.y


def get_hoop_rect():
    cx, cy = HOOP_FRAC[0] * WIDTH, HOOP_FRAC[1] * HEIGHT
    hw = HOOP_WIDTH_FRAC * WIDTH
    return cx, cy, hw


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class Ball:
    def __init__(self, x, y, vx, vy):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.alive = True
        self.scored = False
        self.trail = []

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.trail.append((self.x, self.y))
        if len(self.trail) > 20:
            self.trail.pop(0)
        if self.y > HEIGHT + 100 or self.x < -100 or self.x > WIDTH + 100:
            self.alive = False


class GameState:
    def __init__(self):
        self.prev_pos = None
        self.prev_time = None
        self.cooldown_until = 0
        self.balls = []
        self.made = 0
        self.attempts = 0
        self.streak = 0
        self.best_streak = 0
        self.result_text = ""
        self.result_until = 0

    def reset(self):
        self.__init__()


state = GameState()


def try_launch(wrist_norm, velocity, direction, now):
    if now < state.cooldown_until:
        return
    state.cooldown_until = now + THROW_COOLDOWN
    state.attempts += 1

    launch_x = wrist_norm[0] * WIDTH
    launch_y = wrist_norm[1] * HEIGHT
    power = min(1.6, velocity / 1.5)
    vx = direction[0] * 900 * power
    vy = -abs(direction[1] * 900 * power) - 400   # always launch upward with some force
    state.balls.append(Ball(launch_x, launch_y, vx, vy))


def check_scoring(ball, now):
    hx, hy, hw = get_hoop_rect()
    rim_left = hx - hw / 2
    rim_right = hx + hw / 2
    if not ball.scored and ball.vy > 0 and abs(ball.y - hy) < 14 and rim_left < ball.x < rim_right:
        ball.scored = True
        ball.alive = False
        state.made += 1
        state.streak += 1
        state.best_streak = max(state.best_streak, state.streak)
        state.result_text = "SWISH!" if state.streak < 3 else f"{state.streak} STREAK!"
        state.result_until = now + 0.9
        snd_swish.play()
        if state.streak >= 3:
            snd_cheer.play()


def draw_hoop():
    hx, hy, hw = get_hoop_rect()
    pygame.draw.rect(screen, (200, 60, 40), (hx - hw / 2 - 10, hy - 60, hw + 20, 12))
    pygame.draw.line(screen, (255, 140, 40), (hx - hw / 2, hy), (hx + hw / 2, hy), 6)
    pygame.draw.circle(screen, (255, 140, 40), (int(hx - hw / 2), int(hy)), 5)
    pygame.draw.circle(screen, (255, 140, 40), (int(hx + hw / 2), int(hy)), 5)
    for i in range(5):
        fx = hx - hw / 2 + (i + 0.5) * (hw / 5)
        pygame.draw.line(screen, (255, 255, 255, 120), (fx, hy), (fx, hy + 40), 2)


def draw_balls():
    for ball in state.balls:
        for i, (tx, ty) in enumerate(ball.trail):
            alpha = int(200 * (i / max(1, len(ball.trail))))
            trail_surf = pygame.Surface((8, 8), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (255, 160, 60, alpha), (4, 4), 4)
            screen.blit(trail_surf, (tx - 4, ty - 4))
        pygame.draw.circle(screen, (230, 120, 30), (int(ball.x), int(ball.y)), BALL_RADIUS)
        pygame.draw.circle(screen, (20, 20, 20), (int(ball.x), int(ball.y)), BALL_RADIUS, 2)


def draw_hud(hand_pos, hand_seen):
    if hand_pos:
        color = (0, 255, 0)
        pygame.draw.circle(screen, color, hand_pos, 14, 3)

    if not hand_seen:
        warn = font_med.render("Show your hand and mime a throw!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT - 90))

    score_label = font_med.render(f"Made: {state.made}/{state.attempts}", True, (255, 255, 255))
    screen.blit(score_label, (20, 20))
    streak_label = font_small.render(f"Streak: {state.streak}  Best: {state.best_streak}",
                                      True, (255, 255, 255))
    screen.blit(streak_label, (20, 55))

    now = time.time()
    if now < state.result_until:
        color = (255, 210, 60) if "STREAK" in state.result_text else (80, 255, 120)
        result = font_big.render(state.result_text, True, color)
        screen.blit(result, (WIDTH / 2 - result.get_width() / 2, 40))

    title = font_small.render("AR BASKETBALL HOOP - mime throwing your hand upward at the hoop!",
                               True, (255, 255, 255))
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 26))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    running = True
    last_time = time.time()

    while running:
        now = time.time()
        dt = min(0.05, now - last_time)
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
                    state.reset()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_seen = False
        hand_screen_pos = None
        if results.multi_hand_landmarks:
            hand_seen = True
            landmarks = results.multi_hand_landmarks[0].landmark
            pos_norm = wrist_pos(landmarks)
            hand_screen_pos = (int(pos_norm[0] * WIDTH), int(pos_norm[1] * HEIGHT))

            if state.prev_pos is not None:
                pdt = max(1e-3, now - state.prev_time)
                dx = pos_norm[0] - state.prev_pos[0]
                dy = pos_norm[1] - state.prev_pos[1]
                velocity = math.hypot(dx, dy) / pdt
                if velocity > THROW_VELOCITY_THRESHOLD and dy < 0:
                    direction = (dx / max(1e-4, math.hypot(dx, dy)),
                                 dy / max(1e-4, math.hypot(dx, dy)))
                    try_launch(pos_norm, velocity, direction, now)

            state.prev_pos, state.prev_time = pos_norm, now
        else:
            state.prev_pos = None

        for ball in state.balls:
            ball.update(dt)
            check_scoring(ball, now)
        for ball in list(state.balls):
            if not ball.alive:
                if not ball.scored and ball.y >= HEIGHT - BALL_RADIUS - 5:
                    snd_clank.play()
                    state.streak = 0
                state.balls.remove(ball)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        screen.blit(dim, (0, 0))

        draw_hoop()
        draw_balls()
        draw_hud(hand_screen_pos, hand_seen)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    hands.close()
    pygame.quit()


if __name__ == "__main__":
    main()