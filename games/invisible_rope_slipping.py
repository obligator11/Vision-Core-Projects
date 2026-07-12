import math
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

JUMP_MIN_HEIGHT = 0.018        # normalized hip rise above baseline to count as a jump
SYNC_TOLERANCE = 0.35          # seconds allowed between jump peak and wrist rotation
BASELINE_SMOOTHING = 0.02      # how quickly the "standing" hip baseline adapts

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Invisible Rope Skipping")
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
def to_sound(samples, volume=0.7):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_skip_sound(ms=150):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    whoosh = np.random.uniform(-1, 1, n) * np.exp(-t * 12)
    tap = np.sin(2 * np.pi * 700 * t) * np.exp(-t * 30)
    return to_sound(whoosh * 0.4 + tap * 0.8)


def make_miss_sound(ms=180):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * 160 * t) * np.exp(-t * 8)
    return to_sound(tone, volume=0.5)


snd_skip = make_skip_sound()
snd_miss = make_miss_sound()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def get_hip_y(landmarks):
    l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    return (l.y + r.y) / 2


def get_wrist_angle(landmarks):
    """Angle (radians) of the right wrist around the right shoulder - used to
    detect a full swinging rotation of the 'rope hand'."""
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    return math.atan2(wrist.y - shoulder.y, wrist.x - shoulder.x)


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.baseline_hip_y = None
        self.hip_history = deque(maxlen=5)
        self.was_rising = False
        self.jump_peak_times = deque(maxlen=10)

        self.unwrapped_angle = 0.0
        self.prev_angle = None
        self.last_rotation_time = None
        self.rotation_times = deque(maxlen=10)

        self.skips = 0
        self.misses = 0
        self.combo = 0
        self.best_combo = 0
        self.flash_text = ""
        self.flash_color = (255, 255, 255)
        self.flash_until = 0

    def reset(self):
        skips_kept = self.skips  # keep lifetime stats visible but reset combo/score if desired
        self.__init__()


state = GameState()


def update_jump_detection(hip_y, now):
    if state.baseline_hip_y is None:
        state.baseline_hip_y = hip_y
    else:
        # slowly adapt baseline toward current value only when near-standing
        state.baseline_hip_y += (hip_y - state.baseline_hip_y) * BASELINE_SMOOTHING

    height_above_baseline = state.baseline_hip_y - hip_y  # positive = higher (jumped)
    state.hip_history.append(height_above_baseline)

    rising = len(state.hip_history) >= 2 and state.hip_history[-1] > state.hip_history[-2]
    if state.was_rising and not rising and height_above_baseline > JUMP_MIN_HEIGHT:
        state.jump_peak_times.append(now)
    state.was_rising = rising


def update_wrist_rotation(angle, now):
    if state.prev_angle is not None:
        delta = angle - state.prev_angle
        # unwrap to handle the -pi/pi crossing
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        state.unwrapped_angle += delta

        if abs(state.unwrapped_angle) >= 2 * math.pi:
            state.unwrapped_angle -= math.copysign(2 * math.pi, state.unwrapped_angle)
            state.rotation_times.append(now)
    state.prev_angle = angle


def resolve_skips(now):
    while state.jump_peak_times:
        jump_t = state.jump_peak_times[0]
        if now - jump_t > SYNC_TOLERANCE + 0.05:
            # this jump has aged out - check if any rotation matched it
            matched = any(abs(jump_t - rt) <= SYNC_TOLERANCE for rt in state.rotation_times)
            if matched:
                state.skips += 1
                state.combo += 1
                state.best_combo = max(state.best_combo, state.combo)
                state.flash_text = f"SKIP! x{state.combo}"
                state.flash_color = (80, 255, 120)
                snd_skip.play()
            else:
                state.misses += 1
                state.combo = 0
                state.flash_text = "MISSED TIMING"
                state.flash_color = (255, 90, 90)
                snd_miss.play()
            state.flash_until = now + 0.6
            state.jump_peak_times.popleft()
        else:
            break

    # trim old rotation timestamps
    while state.rotation_times and now - state.rotation_times[0] > 3.0:
        state.rotation_times.popleft()


def draw_hud(hip_y, wrist_pos, person_seen):
    now = time.time()

    if not person_seen:
        warn = font_med.render("Step into frame - full body visible!", True, (255, 90, 90))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT / 2 - 20))
    else:
        # bounce meter
        if state.baseline_hip_y is not None:
            height = max(0.0, state.baseline_hip_y - hip_y)
            bar_h = min(200, int(height * HEIGHT * 4))
            bar_x, bar_y = 30, HEIGHT - 250
            pygame.draw.rect(screen, (40, 40, 40), (bar_x, bar_y, 30, 200), border_radius=6)
            pygame.draw.rect(screen, (60, 220, 90), (bar_x, bar_y + (200 - bar_h), 30, bar_h),
                              border_radius=6)
            pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, 30, 200), 2, border_radius=6)
            screen.blit(font_small.render("JUMP", True, (255, 255, 255)), (bar_x - 5, bar_y - 24))

        if wrist_pos:
            pygame.draw.circle(screen, (0, 200, 255), wrist_pos, 12, 3)

    score_label = font_med.render(f"Skips: {state.skips}", True, (255, 255, 255))
    screen.blit(score_label, (WIDTH - score_label.get_width() - 20, 20))
    combo_label = font_small.render(f"Combo: {state.combo}  Best: {state.best_combo}  Misses: {state.misses}",
                                     True, (255, 255, 255))
    screen.blit(combo_label, (WIDTH - combo_label.get_width() - 20, 60))

    if now < state.flash_until:
        flash = font_big.render(state.flash_text, True, state.flash_color)
        screen.blit(flash, (WIDTH / 2 - flash.get_width() / 2, 30))

    title = font_small.render("INVISIBLE ROPE SKIPPING - jump in time with your wrist swing",
                               True, (255, 255, 255))
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 26))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    running = True

    while running:
        now = time.time()

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
        results = pose.process(rgb)

        person_seen = False
        hip_y = 0.5
        wrist_pos = None
        if results.pose_landmarks:
            person_seen = True
            landmarks = results.pose_landmarks.landmark
            hip_y = get_hip_y(landmarks)
            angle = get_wrist_angle(landmarks)
            update_jump_detection(hip_y, now)
            update_wrist_rotation(angle, now)
            resolve_skips(now)

            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            wrist_pos = (int(wrist.x * WIDTH), int(wrist.y * HEIGHT))

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 70))
        screen.blit(dim, (0, 0))

        draw_hud(hip_y, wrist_pos, person_seen)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()