
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
SAMPLE_RATE = 22050

HOLD_TO_CONFIRM = 0.4

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Hand Shadow Puppet Classifier")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 44, bold=True)
font_med = pygame.font.SysFont("arial", 26, bold=True)
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
def to_sound(samples, volume=0.6):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_bell(freq=1200, ms=300):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t * 6)
    return to_sound(tone)


def make_woof(ms=220):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    freq = 180 * np.exp(-t * 6) + 90
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    tone = np.sin(phase) * np.exp(-t * 5)
    return to_sound(tone)


def make_chirp(ms=250):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    freq = 1400 + 600 * np.sin(t * 40)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    tone = np.sin(phase) * np.exp(-t * 4)
    return to_sound(tone, volume=0.4)


SOUNDS = {"RABBIT": make_bell(1400), "DOG": make_woof(), "BIRD": make_chirp()}


# ----------------------------------------------------------------------------
# HAND GEOMETRY / CLASSIFICATION
# ----------------------------------------------------------------------------
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}


def finger_extended(landmarks, finger, wrist):
    tip = landmarks[FINGER_TIPS[finger]]
    pip = landmarks[FINGER_PIPS[finger]]
    tip_dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    pip_dist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
    return tip_dist > pip_dist * 1.15


def classify_hand(landmarks):
    wrist = landmarks[0]
    ext = {f: finger_extended(landmarks, f, wrist) for f in FINGER_TIPS}

    index_tip = landmarks[FINGER_TIPS["index"]]
    middle_tip = landmarks[FINGER_TIPS["middle"]]
    spread = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)

    # RABBIT: index + middle extended (the "ears"), ring + pinky curled
    if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "RABBIT"

    # DOG: thumb + index extended forming an L/snout, others curled
    if ext["thumb"] and ext["index"] and not ext["middle"] and not ext["ring"] and not ext["pinky"]:
        return "DOG"

    # BIRD: thumb, index, middle extended and spread (beak/head), ring+pinky curled
    if ext["thumb"] and ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"] and spread > 0.05:
        return "BIRD"

    return None


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def hand_center_px(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return sum(xs) / len(xs) * WIDTH, sum(ys) / len(ys) * HEIGHT


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.candidate = None
        self.candidate_since = None
        self.locked_animal = None
        self.locked_until = 0
        self.matches = {"RABBIT": 0, "DOG": 0, "BIRD": 0}


state = GameState()


def update_classification(animal, now):
    if animal != state.candidate:
        state.candidate = animal
        state.candidate_since = now

    if animal and now - state.candidate_since >= HOLD_TO_CONFIRM:
        if state.locked_animal != animal or now > state.locked_until:
            state.locked_animal = animal
            state.matches[animal] += 1
            SOUNDS[animal].play()
        state.locked_until = now + 1.2


def draw_silhouette(animal, center, now):
    cx, cy = center
    wiggle = math.sin(now * 6) * 6

    if animal == "RABBIT":
        pygame.draw.ellipse(screen, (20, 20, 20), (cx - 30, cy - 10, 60, 80))
        ear1 = [(cx - 18, cy - 10), (cx - 26 + wiggle, cy - 90), (cx - 6, cy - 15)]
        ear2 = [(cx + 18, cy - 10), (cx + 26 + wiggle, cy - 90), (cx + 6, cy - 15)]
        pygame.draw.polygon(screen, (20, 20, 20), ear1)
        pygame.draw.polygon(screen, (20, 20, 20), ear2)
    elif animal == "DOG":
        pygame.draw.ellipse(screen, (20, 20, 20), (cx - 40, cy - 20, 90, 55))
        pygame.draw.polygon(screen, (20, 20, 20),
                             [(cx - 40, cy - 15), (cx - 70, cy - 5), (cx - 40, cy + 15)])
        ear_wag = 6 + wiggle * 0.3
        pygame.draw.ellipse(screen, (20, 20, 20), (cx + 5, cy - 45 + ear_wag, 22, 30))
    elif animal == "BIRD":
        pygame.draw.ellipse(screen, (20, 20, 20), (cx - 25, cy - 20, 55, 45))
        pygame.draw.polygon(screen, (20, 20, 20),
                             [(cx + 25, cy - 5), (cx + 55, cy + wiggle * 0.5), (cx + 25, cy + 10)])
        wing_flap = abs(wiggle) * 1.5
        pygame.draw.polygon(screen, (20, 20, 20),
                             [(cx - 10, cy), (cx - 45, cy - 15 - wing_flap), (cx - 15, cy + 15)])


def draw_hud(animal, hand_seen, hand_center, now):
    if not hand_seen:
        warn = font_med.render("Show your hand to the camera!", True, (255, 90, 90))
        screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, 40))
    elif state.candidate and now - state.candidate_since < HOLD_TO_CONFIRM:
        progress = (now - state.candidate_since) / HOLD_TO_CONFIRM
        label = font_small.render(f"Holding {state.candidate}...", True, (255, 210, 60))
        screen.blit(label, (WIDTH / 2 - label.get_width() / 2, 40))
        pygame.draw.rect(screen, (255, 210, 60), (WIDTH / 2 - 60, 65, 120 * progress, 8))

    if state.locked_animal and now < state.locked_until and hand_center:
        draw_silhouette(state.locked_animal, hand_center, now)
        label = font_big.render(state.locked_animal, True, (80, 255, 120))
        screen.blit(label, (WIDTH / 2 - label.get_width() / 2, HEIGHT - 100))

    stats = font_small.render(
        f"Rabbit: {state.matches['RABBIT']}   Dog: {state.matches['DOG']}   Bird: {state.matches['BIRD']}",
        True, (255, 255, 255))
    screen.blit(stats, (20, 20))

    title = font_small.render(
        "SHADOW PUPPETS - Rabbit: index+middle up | Dog: thumb+index L-shape | Bird: 3 fingers spread",
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

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_seen = False
        hand_center = None
        animal = None
        if results.multi_hand_landmarks:
            hand_seen = True
            landmarks = results.multi_hand_landmarks[0].landmark
            animal = classify_hand(landmarks)
            hand_center = hand_center_px(landmarks)
            update_classification(animal, now)
        else:
            state.candidate = None

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 60))
        screen.blit(dim, (0, 0))

        draw_hud(animal, hand_seen, hand_center, now)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    hands.close()
    pygame.quit()


if __name__ == "__main__":
    main()