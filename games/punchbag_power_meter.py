import math
import random
import sys
import time
from collections import deque

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

PUNCH_START_VELOCITY = 0.9     # normalized units/sec to consider a punch "started"
PUNCH_END_VELOCITY = 0.25      # velocity below this ends the swing window
COOLDOWN_AFTER_PUNCH = 0.35
MAX_POWER_THRESHOLD = 88.0

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
BASE_SIZE = (1000, 700)
screen = pygame.display.set_mode(BASE_SIZE, pygame.RESIZABLE)
pygame.display.set_caption("Punch Bag Power Meter")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 64, bold=True)
font_med = pygame.font.SysFont("arial", 30, bold=True)
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
def to_sound(samples, volume=0.8):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_impact(power_frac, ms=220):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    freq = 70 + power_frac * 40
    thud = np.sin(2 * np.pi * freq * t) * np.exp(-t * (10 - power_frac * 4))
    crack = np.random.uniform(-1, 1, n) * np.exp(-t * 40) * power_frac
    return to_sound(thud * 0.8 + crack * 0.6, volume=0.5 + power_frac * 0.5)


def make_max_power_boom(ms=500):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    boom = np.sin(2 * np.pi * 55 * t) * np.exp(-t * 4)
    noise = np.random.uniform(-1, 1, n) * np.exp(-t * 8)
    return to_sound(boom * 0.9 + noise * 0.4, volume=0.8)


snd_max_power = make_max_power_boom()
_impact_cache = {}


def get_impact_sound(power_pct):
    key = round(power_pct / 10)
    if key not in _impact_cache:
        _impact_cache[key] = make_impact(power_pct / 100)
    return _impact_cache[key]


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr, w, h):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (w, h))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def get_wrist_3d(landmarks, side="RIGHT"):
    lm = landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value]
    return np.array([lm.x, lm.y, lm.z])


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.prev_pos = None
        self.prev_time = None
        self.in_swing = False
        self.swing_start_pos = None
        self.swing_path_length = 0.0
        self.peak_velocity = 0.0
        self.last_punch_time = 0
        self.meter_value = 0.0       # live display value (eases toward target)
        self.meter_target = 0.0
        self.best_power = 0.0
        self.shake_until = 0
        self.shake_strength = 0
        self.result_text = ""
        self.result_until = 0

    def reset_best(self):
        self.best_power = 0.0


state = GameState()


def score_punch(peak_velocity, path_length, now):
    # normalize into a 0-100 power score (tuned empirically for typical punch speeds)
    velocity_component = min(60, peak_velocity * 30)
    distance_component = min(40, path_length * 250)
    power = velocity_component + distance_component
    power = max(0.0, min(100.0, power))

    state.meter_target = power
    state.best_power = max(state.best_power, power)

    sound = get_impact_sound(power)
    sound.set_volume(0.4 + power / 100 * 0.6)
    sound.play()

    if power >= MAX_POWER_THRESHOLD:
        state.result_text = "MAX POWER!!"
        state.shake_until = now + 0.35
        state.shake_strength = 14
        snd_max_power.play()
    elif power >= 60:
        state.result_text = "POWERFUL HIT!"
        state.shake_until = now + 0.15
        state.shake_strength = 6
    elif power >= 30:
        state.result_text = "Solid hit"
    else:
        state.result_text = "Weak jab"
    state.result_until = now + 1.0


def update_punch_tracking(wrist_pos, now):
    if state.prev_pos is None:
        state.prev_pos, state.prev_time = wrist_pos, now
        return

    dt = max(1e-3, now - state.prev_time)
    velocity = np.linalg.norm(wrist_pos - state.prev_pos) / dt

    if not state.in_swing and velocity > PUNCH_START_VELOCITY and now - state.last_punch_time > COOLDOWN_AFTER_PUNCH:
        state.in_swing = True
        state.swing_start_pos = wrist_pos
        state.swing_path_length = 0.0
        state.peak_velocity = velocity

    if state.in_swing:
        state.swing_path_length += np.linalg.norm(wrist_pos - state.prev_pos)
        state.peak_velocity = max(state.peak_velocity, velocity)

        if velocity < PUNCH_END_VELOCITY:
            state.in_swing = False
            state.last_punch_time = now
            score_punch(state.peak_velocity, state.swing_path_length, now)

    state.prev_pos, state.prev_time = wrist_pos, now


def draw_hud(w, h, wrist_screen_pos, person_seen):
    # ease meter toward target for a satisfying fill animation
    state.meter_value += (state.meter_target - state.meter_value) * 0.25

    meter_w, meter_h = int(w * 0.08), int(h * 0.55)
    meter_x, meter_y = int(w * 0.06), int(h * 0.22)
    pygame.draw.rect(screen, (40, 40, 40), (meter_x, meter_y, meter_w, meter_h), border_radius=10)
    fill_h = int(meter_h * (state.meter_value / 100))
    fill_color = (255, 90, 90) if state.meter_value >= MAX_POWER_THRESHOLD else \
        ((255, 200, 60) if state.meter_value >= 60 else (80, 220, 120))
    pygame.draw.rect(screen, fill_color,
                      (meter_x, meter_y + meter_h - fill_h, meter_w, fill_h), border_radius=10)
    pygame.draw.rect(screen, (255, 255, 255), (meter_x, meter_y, meter_w, meter_h), 3, border_radius=10)
    label = font_small.render("POWER", True, (255, 255, 255))
    screen.blit(label, (meter_x + meter_w / 2 - label.get_width() / 2, meter_y - 24))
    val_label = font_med.render(f"{int(state.meter_value)}", True, (255, 255, 255))
    screen.blit(val_label, (meter_x + meter_w / 2 - val_label.get_width() / 2, meter_y + meter_h + 8))

    best_label = font_small.render(f"Best: {int(state.best_power)}", True, (255, 255, 255))
    screen.blit(best_label, (w - best_label.get_width() - 20, 20))

    if wrist_screen_pos and person_seen:
        color = (255, 90, 90) if state.in_swing else (0, 200, 255)
        pygame.draw.circle(screen, color, wrist_screen_pos, 16, 3)

    if not person_seen:
        warn = font_med.render("Step into frame!", True, (255, 80, 80))
        screen.blit(warn, (w / 2 - warn.get_width() / 2, 40))

    now = time.time()
    if now < state.result_until:
        color = (255, 90, 90) if "MAX" in state.result_text else (255, 255, 255)
        result = font_big.render(state.result_text, True, color)
        screen.blit(result, (w / 2 - result.get_width() / 2, h / 2 - 150))

    title = font_small.render("PUNCH BAG POWER METER - throw a punch toward the camera!",
                               True, (255, 255, 255))
    screen.blit(title, (w / 2 - title.get_width() / 2, h - 26))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global screen
    w, h = BASE_SIZE
    running = True

    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                w, h = max(400, event.w), max(300, event.h)
                screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    state.reset_best()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        person_seen = False
        wrist_screen_pos = None
        if results.pose_landmarks:
            person_seen = True
            landmarks = results.pose_landmarks.landmark
            wrist_3d = get_wrist_3d(landmarks, "RIGHT")
            update_punch_tracking(wrist_3d, now)
            wrist_screen_pos = (int(wrist_3d[0] * w), int(wrist_3d[1] * h))

        # screen shake offset
        offset_x, offset_y = 0, 0
        if now < state.shake_until:
            offset_x = random.randint(-state.shake_strength, state.shake_strength)
            offset_y = random.randint(-state.shake_strength, state.shake_strength)

        surf = frame_to_surface(frame, w, h)
        screen.fill((0, 0, 0))
        screen.blit(surf, (offset_x, offset_y))
        dim = pygame.Surface((w, h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        screen.blit(dim, (0, 0))

        draw_hud(w, h, wrist_screen_pos, person_seen)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()