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

HOLD_TO_LOCK_IN = 0.7          # seconds a new prompt must be held to "lock in"
BREAK_GRACE_PERIOD = 0.6       # seconds an active prompt may be off-zone before failing
NEXT_PROMPT_DELAY = 1.0        # pause after locking in before the next prompt appears

ZONE_LABELS = ["TL", "TC", "TR", "ML", "C", "MR", "BL", "BC", "BR"]
BODY_PARTS = {
    "LEFT HAND": "LEFT_WRIST", "RIGHT HAND": "RIGHT_WRIST",
    "LEFT FOOT": "LEFT_ANKLE", "RIGHT FOOT": "RIGHT_ANKLE",
    "HEAD": "NOSE",
}

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Silhouette Twister")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 56, bold=True)
font_med = pygame.font.SysFont("arial", 26, bold=True)
font_small = pygame.font.SysFont("arial", 16)

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


def make_ding(freq=880, ms=180):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t * 6)
    return to_sound(tone)


def make_buzzer(ms=350):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sign(np.sin(2 * np.pi * 110 * t)) * np.exp(-t * 3)
    return to_sound(tone, volume=0.5)


snd_lock = make_ding(880)
snd_fail = make_buzzer()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def get_zone_rects():
    cell_w, cell_h = WIDTH / 3, HEIGHT / 3
    rects = {}
    for i, label in enumerate(ZONE_LABELS):
        row, col = divmod(i, 3)
        rects[label] = pygame.Rect(col * cell_w, row * cell_h, cell_w, cell_h)
    return rects


def landmark_px(landmarks, name):
    lm = landmarks[mp_pose.PoseLandmark[name].value]
    return lm.x * WIDTH, lm.y * HEIGHT


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class Prompt:
    def __init__(self, body_part, zone):
        self.body_part = body_part
        self.zone = zone
        self.locked_in = False
        self.hold_start = None
        self.break_since = None


class GameState:
    def __init__(self):
        self.active_prompts = []
        self.streak = 0
        self.best_streak = 0
        self.failed = False
        self.fail_message = ""
        self.spawn_next_prompt()

    def used_parts(self):
        return {p.body_part for p in self.active_prompts}

    def used_zones(self):
        return {p.zone for p in self.active_prompts}

    def spawn_next_prompt(self):
        available_parts = [p for p in BODY_PARTS if p not in self.used_parts()]
        available_zones = [z for z in ZONE_LABELS if z not in self.used_zones()]
        if not available_parts or not available_zones:
            # cycle: clear locked-in prompts to keep the game going indefinitely
            self.active_prompts = []
            available_parts = list(BODY_PARTS.keys())
            available_zones = list(ZONE_LABELS)
        part = random.choice(available_parts)
        zone = random.choice(available_zones)
        self.active_prompts.append(Prompt(part, zone))

    def reset(self):
        self.__init__()


state = GameState()


def update_game(landmarks, person_seen, now, zone_rects):
    if state.failed:
        return
    if not person_seen:
        # treat as all prompts breaking
        for p in state.active_prompts:
            if p.break_since is None:
                p.break_since = now
        _check_failures(now)
        return

    all_current_locked = True
    for p in state.active_prompts:
        landmark_name = BODY_PARTS[p.body_part]
        px, py = landmark_px(landmarks, landmark_name)
        in_zone = zone_rects[p.zone].collidepoint(px, py)

        if in_zone:
            p.break_since = None
            if not p.locked_in:
                all_current_locked = False
                if p.hold_start is None:
                    p.hold_start = now
                elif now - p.hold_start >= HOLD_TO_LOCK_IN:
                    p.locked_in = True
                    snd_lock.play()
        else:
            p.hold_start = None
            if p.break_since is None:
                p.break_since = now
            all_current_locked = False

    _check_failures(now)

    if state.failed:
        return

    if all(p.locked_in for p in state.active_prompts):
        state.streak += 1
        state.best_streak = max(state.best_streak, state.streak)
        state.spawn_next_prompt()


def _check_failures(now):
    for p in state.active_prompts:
        if p.break_since is not None and now - p.break_since > BREAK_GRACE_PERIOD:
            state.failed = True
            state.fail_message = f"Lost it on {p.body_part}!"
            snd_fail.play()
            return


def draw_zones(zone_rects):
    for label, rect in zone_rects.items():
        pygame.draw.rect(screen, (255, 255, 255), rect, 1)


def draw_prompts(zone_rects, now):
    for p in state.active_prompts:
        rect = zone_rects[p.zone]
        if p.locked_in:
            color = (60, 220, 90)
        elif p.break_since is not None:
            color = (255, 90, 90)
        elif p.hold_start is not None:
            color = (255, 210, 60)
        else:
            color = (0, 200, 255)

        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((*color, 70))
        screen.blit(overlay, rect.topleft)
        pygame.draw.rect(screen, color, rect, 4)

        label = font_small.render(p.body_part, True, (255, 255, 255))
        screen.blit(label, (rect.centerx - label.get_width() / 2, rect.centery - 8))

        if p.hold_start is not None and not p.locked_in:
            progress = min(1.0, (now - p.hold_start) / HOLD_TO_LOCK_IN)
            pygame.draw.arc(screen, color, (rect.centerx - 20, rect.centery + 10, 40, 40),
                             -1.57, -1.57 + progress * 6.28, 4)


def draw_hud():
    streak_label = font_med.render(f"Streak: {state.streak}   Best: {state.best_streak}",
                                    True, (255, 255, 255))
    screen.blit(streak_label, (20, 20))

    if state.failed:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        msg = font_big.render("BROKEN POSE!", True, (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 80))
        detail = font_med.render(state.fail_message, True, (255, 255, 255))
        screen.blit(detail, (WIDTH / 2 - detail.get_width() / 2, HEIGHT / 2 - 10))
        sc = font_small.render(f"Final streak: {state.streak}", True, (220, 220, 220))
        screen.blit(sc, (WIDTH / 2 - sc.get_width() / 2, HEIGHT / 2 + 30))
        hint = font_small.render("Press R to try again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 55))

    title = font_small.render("SILHOUETTE TWISTER - match every glowing zone at once!",
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
                elif event.key == pygame.K_r and state.failed:
                    state.reset()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        person_seen = bool(results.pose_landmarks)
        zone_rects = get_zone_rects()

        if not state.failed:
            update_game(results.pose_landmarks.landmark if person_seen else None,
                        person_seen, now, zone_rects)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 60))
        screen.blit(dim, (0, 0))

        draw_zones(zone_rects)
        draw_prompts(zone_rects, now)
        draw_hud()

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    pose.close()
    pygame.quit()


if __name__ == "__main__":
    main()