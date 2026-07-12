
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

TIME_LIMIT = 60.0
STARTING_LIVES = 3
BALLOON_RADIUS = 34
SPAWN_INTERVAL_RANGE = (0.35, 0.9)
BOMB_CHANCE_BASE = 0.10
BOMB_CHANCE_MAX = 0.30

POPPER_LANDMARKS = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE", "NOSE"]

BALLOON_COLORS = [(255, 90, 90), (90, 180, 255), (255, 210, 60), (120, 220, 120), (220, 120, 255)]

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Balloon Pop Frenzy")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 56, bold=True)
font_med = pygame.font.SysFont("arial", 28, bold=True)
font_small = pygame.font.SysFont("arial", 18)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# SOUND
# ----------------------------------------------------------------------------
def to_sound(samples, volume=0.6):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_pop(freq, ms=110):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t * 25)
    click = np.random.uniform(-1, 1, n) * np.exp(-t * 60) * 0.4
    return to_sound(tone + click)


def make_boom(ms=400):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * 70 * t) * np.exp(-t * 6)
    noise = np.random.uniform(-1, 1, n) * np.exp(-t * 10)
    return to_sound(tone * 0.7 + noise * 0.6, volume=0.7)


POP_SOUNDS = [make_pop(f) for f in (440, 523, 659, 784, 988)]
snd_boom = make_boom()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def get_popper_points(landmarks):
    pts = []
    for name in POPPER_LANDMARKS:
        lm = landmarks[mp_pose.PoseLandmark[name].value]
        if lm.visibility > 0.4:
            pts.append((lm.x * WIDTH, lm.y * HEIGHT))
    return pts


# ----------------------------------------------------------------------------
# ENTITIES
# ----------------------------------------------------------------------------
class Balloon:
    def __init__(self, is_bomb):
        self.x = random.uniform(BALLOON_RADIUS, WIDTH - BALLOON_RADIUS)
        self.y = HEIGHT + BALLOON_RADIUS
        self.speed = random.uniform(70, 160)
        self.sway_phase = random.uniform(0, 6.28)
        self.sway_amp = random.uniform(10, 40)
        self.is_bomb = is_bomb
        self.color = (30, 30, 30) if is_bomb else random.choice(BALLOON_COLORS)
        self.alive = True

    def update(self, dt, now):
        self.y -= self.speed * dt
        self.x += np.sin(now * 2 + self.sway_phase) * self.sway_amp * dt
        if self.y < -BALLOON_RADIUS:
            self.alive = False


class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 6.28)
        speed = random.uniform(80, 260)
        self.x, self.y = x, y
        self.vx, self.vy = np.cos(angle) * speed, np.sin(angle) * speed
        self.life = 0.5
        self.age = 0.0
        self.color = color

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 300 * dt

    @property
    def alive(self):
        return self.age < self.life


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.balloons = []
        self.particles = []
        self.score = 0
        self.lives = STARTING_LIVES
        self.start_time = time.time()
        self.next_spawn = time.time() + random.uniform(*SPAWN_INTERVAL_RANGE)
        self.game_over = False
        self.hit_this_frame = set()
        self.flash_until = 0
        self.flash_color = (255, 255, 255)

    def reset(self):
        self.__init__()


state = GameState()


def spawn_balloons(now):
    if now < state.next_spawn or state.game_over:
        return
    elapsed = now - state.start_time
    bomb_chance = min(BOMB_CHANCE_MAX, BOMB_CHANCE_BASE + elapsed * 0.003)
    is_bomb = random.random() < bomb_chance
    state.balloons.append(Balloon(is_bomb))
    gap = max(0.15, SPAWN_INTERVAL_RANGE[0] - elapsed * 0.01)
    state.next_spawn = now + random.uniform(gap, SPAWN_INTERVAL_RANGE[1])


def spawn_particles(x, y, color, count=18):
    for _ in range(count):
        state.particles.append(Particle(x, y, color))


def check_pops(popper_points, now):
    for balloon in state.balloons:
        if not balloon.alive:
            continue
        for (px, py) in popper_points:
            dist = ((px - balloon.x) ** 2 + (py - balloon.y) ** 2) ** 0.5
            if dist <= BALLOON_RADIUS:
                balloon.alive = False
                if balloon.is_bomb:
                    state.lives -= 1
                    snd_boom.play()
                    spawn_particles(balloon.x, balloon.y, (255, 120, 40), 30)
                    state.flash_until = now + 0.2
                    state.flash_color = (255, 60, 60)
                    if state.lives <= 0:
                        state.game_over = True
                else:
                    state.score += 10
                    random.choice(POP_SOUNDS).play()
                    spawn_particles(balloon.x, balloon.y, balloon.color, 14)
                break


def draw_balloons():
    for b in state.balloons:
        if not b.alive:
            continue
        pygame.draw.ellipse(screen, b.color,
                             (b.x - BALLOON_RADIUS * 0.8, b.y - BALLOON_RADIUS,
                              BALLOON_RADIUS * 1.6, BALLOON_RADIUS * 2))
        pygame.draw.line(screen, (120, 120, 120), (b.x, b.y + BALLOON_RADIUS),
                          (b.x, b.y + BALLOON_RADIUS + 20), 2)
        if b.is_bomb:
            label = font_small.render("X", True, (255, 60, 60))
            screen.blit(label, (b.x - label.get_width() / 2, b.y - label.get_height() / 2))


def draw_particles():
    for p in state.particles:
        alpha = max(0, 255 * (1 - p.age / p.life))
        surf = pygame.Surface((10, 10), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*p.color, int(alpha)), (5, 5), 5)
        screen.blit(surf, (p.x - 5, p.y - 5))


def draw_hud(popper_points, person_seen, now):
    for (px, py) in popper_points:
        pygame.draw.circle(screen, (0, 220, 255), (int(px), int(py)), 10, 2)

    if not person_seen:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, 40))

    elapsed = time.time() - state.start_time
    time_left = max(0, TIME_LIMIT - elapsed)
    timer_label = font_med.render(f"Time: {int(time_left)}s", True, (255, 255, 255))
    screen.blit(timer_label, (20, 20))
    score_label = font_med.render(f"Score: {state.score}", True, (255, 255, 255))
    screen.blit(score_label, (WIDTH - score_label.get_width() - 20, 20))
    for i in range(STARTING_LIVES):
        color = (230, 60, 60) if i < state.lives else (70, 70, 70)
        pygame.draw.circle(screen, color, (WIDTH / 2 - 40 + i * 34, 30), 12)

    if now < state.flash_until:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((*state.flash_color, 70))
        screen.blit(overlay, (0, 0))

    if state.game_over or time_left <= 0:
        state.game_over = True
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        msg = font_big.render("TIME'S UP!" if state.lives > 0 else "BOOM! GAME OVER",
                               True, (255, 210, 60) if state.lives > 0 else (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 80))
        sc = font_med.render(f"Final Score: {state.score}", True, (255, 255, 255))
        screen.blit(sc, (WIDTH / 2 - sc.get_width() / 2, HEIGHT / 2))
        hint = font_small.render("Press R to play again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 40))
    else:
        title = font_small.render("BALLOON POP FRENZY - pop balloons, dodge the bombs!",
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
                elif event.key == pygame.K_r and state.game_over:
                    state.reset()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        popper_points = []
        person_seen = False
        if results.pose_landmarks:
            person_seen = True
            popper_points = get_popper_points(results.pose_landmarks.landmark)

        if not state.game_over:
            spawn_balloons(now)
            for b in state.balloons:
                b.update(dt, now)
            state.balloons = [b for b in state.balloons if b.alive]
            if popper_points:
                check_pops(popper_points, now)

        for p in state.particles:
            p.update(dt)
        state.particles = [p for p in state.particles if p.alive]

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 80))
        screen.blit(dim, (0, 0))

        draw_balloons()
        draw_particles()
        draw_hud(popper_points, person_seen, now)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()