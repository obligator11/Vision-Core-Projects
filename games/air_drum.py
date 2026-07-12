import math
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

HIT_VELOCITY_THRESHOLD = 0.9   # normalized units/sec (downward) to trigger a hit
RETRIGGER_COOLDOWN = 0.18      # seconds before the same pad can trigger again
SAMPLE_RATE = 22050

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Air Drum Kit")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 48, bold=True)
font_med = pygame.font.SysFont("arial", 28, bold=True)
font_small = pygame.font.SysFont("arial", 18)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, model_complexity=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# SOUND SYNTHESIS
# ----------------------------------------------------------------------------
def _envelope(n, attack=0.02, decay=0.9):
    env = np.ones(n)
    a = max(1, int(n * attack))
    env[:a] = np.linspace(0, 1, a)
    env[a:] *= np.linspace(1, 0, n - a) ** decay
    return env


def to_sound(samples, volume=1.0):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_kick(ms=280):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    freq_sweep = 150 * np.exp(-t * 18) + 45
    phase = 2 * np.pi * np.cumsum(freq_sweep) / SAMPLE_RATE
    tone = np.sin(phase)
    click = np.exp(-t * 400) * np.random.uniform(-1, 1, n) * 0.3
    return to_sound((tone + click) * _envelope(n, 0.001, 1.2))


def make_snare(ms=200):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    noise = np.random.uniform(-1, 1, n)
    tone = np.sin(2 * np.pi * 190 * t) * 0.5
    return to_sound((noise * 0.7 + tone) * _envelope(n, 0.001, 1.0))


def make_hihat(ms=90):
    n = int(SAMPLE_RATE * ms / 1000)
    noise = np.random.uniform(-1, 1, n)
    # crude high-pass emphasis via first-difference
    hp = np.diff(noise, prepend=0)
    return to_sound(hp * _envelope(n, 0.001, 0.6), volume=0.8)


def make_cymbal(ms=900):
    n = int(SAMPLE_RATE * ms / 1000)
    noise = np.random.uniform(-1, 1, n)
    hp = np.diff(noise, prepend=0)
    return to_sound(hp * _envelope(n, 0.001, 2.0), volume=0.55)


snd_kick = make_kick()
snd_snare = make_snare()
snd_hihat = make_hihat()
snd_cymbal = make_cymbal()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def palm_center(landmarks):
    ids = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x for i in ids]
    ys = [landmarks[i].y for i in ids]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def get_pads():
    """Pad zones as fractions of the window, recomputed every frame so
    resizing the window keeps them proportionally placed."""
    return {
        "HI-HAT": {"rect": pygame.Rect(0, int(HEIGHT * 0.08), int(WIDTH * 0.32), int(HEIGHT * 0.42)),
                   "color": (255, 210, 60), "sound": snd_hihat},
        "CYMBAL": {"rect": pygame.Rect(int(WIDTH * 0.68), int(HEIGHT * 0.08), int(WIDTH * 0.32), int(HEIGHT * 0.42)),
                   "color": (0, 200, 255), "sound": snd_cymbal},
        "SNARE": {"rect": pygame.Rect(int(WIDTH * 0.28), int(HEIGHT * 0.55), int(WIDTH * 0.2), int(HEIGHT * 0.35)),
                  "color": (255, 100, 100), "sound": snd_snare},
        "KICK": {"rect": pygame.Rect(int(WIDTH * 0.55), int(HEIGHT * 0.55), int(WIDTH * 0.2), int(HEIGHT * 0.35)),
                 "color": (140, 120, 255), "sound": snd_kick},
    }


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class HandTrack:
    def __init__(self):
        self.prev_pos = None
        self.prev_time = None
        self.last_hit_pad = None
        self.last_hit_time = 0


class GameState:
    def __init__(self):
        self.hits = {"KICK": 0, "SNARE": 0, "HI-HAT": 0, "CYMBAL": 0}
        self.flash = {}   # pad name -> expiry time
        self.total_hits = 0
        self.start_time = time.time()


state = GameState()
hand_tracks = [HandTrack(), HandTrack()]


def trigger_hit(pad_name, pad, velocity, now):
    volume = min(1.0, 0.4 + velocity / 4.0)
    pad["sound"].set_volume(volume)
    pad["sound"].play()
    state.hits[pad_name] += 1
    state.total_hits += 1
    state.flash[pad_name] = now + 0.15


def update_hand(track, pos_norm, now, pads):
    if track.prev_pos is not None and track.prev_time is not None:
        dt = max(1e-3, now - track.prev_time)
        vy = (pos_norm[1] - track.prev_pos[1]) / dt   # positive = moving down
        px, py = pos_norm[0] * WIDTH, pos_norm[1] * HEIGHT

        hit_pad = None
        for name, pad in pads.items():
            if pad["rect"].collidepoint(px, py):
                hit_pad = name
                break

        if (hit_pad and vy > HIT_VELOCITY_THRESHOLD
                and (hit_pad != track.last_hit_pad or now - track.last_hit_time > RETRIGGER_COOLDOWN)):
            trigger_hit(hit_pad, pads[hit_pad], vy, now)
            track.last_hit_pad = hit_pad
            track.last_hit_time = now
        elif hit_pad is None:
            track.last_hit_pad = None

    track.prev_pos = pos_norm
    track.prev_time = now


def draw_pads(pads):
    now = time.time()
    for name, pad in pads.items():
        rect = pad["rect"]
        is_flashing = state.flash.get(name, 0) > now
        color = tuple(min(255, c + 90) for c in pad["color"]) if is_flashing else pad["color"]
        pygame.draw.rect(screen, color, rect, border_radius=16)
        pygame.draw.rect(screen, (255, 255, 255), rect, 3, border_radius=16)
        label = font_med.render(name, True, (20, 20, 20))
        screen.blit(label, (rect.centerx - label.get_width() / 2, rect.centery - 30))
        count = font_small.render(f"hits: {state.hits[name]}", True, (20, 20, 20))
        screen.blit(count, (rect.centerx - count.get_width() / 2, rect.centery + 6))


def draw_hud(hand_positions):
    for (px, py) in hand_positions:
        pygame.draw.circle(screen, (0, 255, 0), (int(px), int(py)), 16, 3)
        pygame.draw.circle(screen, (0, 255, 0), (int(px), int(py)), 4)

    elapsed = time.time() - state.start_time
    bpm_estimate = (state.total_hits / max(1.0, elapsed)) * 60
    stats = font_small.render(
        f"Total hits: {state.total_hits}   Hit rate: {bpm_estimate:0.0f}/min",
        True, (255, 255, 255))
    screen.blit(stats, (10, 10))

    title = font_small.render("AIR DRUM KIT - swing your hand down into a pad to hit it!",
                               True, (255, 255, 255))
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 26))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = max(400, event.w), max(300, event.h)
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        now = time.time()
        pads = get_pads()
        hand_positions = []

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                pos_norm = palm_center(hand_landmarks.landmark)
                hand_positions.append((pos_norm[0] * WIDTH, pos_norm[1] * HEIGHT))
                update_hand(hand_tracks[i], pos_norm, now, pads)
            # reset stale tracks if fewer than 2 hands seen this frame
            for j in range(len(results.multi_hand_landmarks), 2):
                hand_tracks[j].prev_pos = None
                hand_tracks[j].prev_time = None
        else:
            for t in hand_tracks:
                t.prev_pos = None
                t.prev_time = None

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 90))
        screen.blit(dim, (0, 0))

        draw_pads(pads)
        draw_hud(hand_positions)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    hands.close()
    pygame.quit()


if __name__ == "__main__":
    main()