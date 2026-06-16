"""
╔══════════════════════════════════════════════════════════╗
║           CLONE FINGER ARMY  —  by Sam                  ║
║  Multi-finger tracking game: each finger = one unit      ║
║  Controls: show fingers on webcam, avoid red enemies     ║
║  Q = quit  |  C = chaos mode  |  R = restart             ║
╚══════════════════════════════════════════════════════════╝
"""

import pygame
import cv2
import numpy as np
import sys
import math
import random
import time
import threading

# ─── Try MediaPipe (new API) ─────────────────────────────────────────────────
try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
    _mp_draw  = mp.solutions.drawing_utils
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False

# ─── CONFIG ──────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1280, 720          # Adjustable game window
CAM_W,  CAM_H  = 320, 240        # Overlay camera size
CAM_X,  CAM_Y  = 10, WIN_H - CAM_H - 10

FINGER_TIPS  = [4, 8, 12, 16, 20]   # MediaPipe fingertip landmark ids
TRAIL_LEN    = 28                    # Trail history length
ENEMY_SPAWN  = 1.8                   # Enemy spawn interval (sec)
ENEMY_SPEED  = 2.5
CHAOS_MULT   = 3                     # Enemy multiplier in chaos mode
FPS          = 60

# Palette  (neon arcade)
C_BG         = (5, 5, 18)
C_GRID       = (15, 20, 45)
C_HUD        = (200, 200, 255)
C_ENEMY      = (255, 40, 60)
C_ENEMY_GLOW = (255, 80, 80)
C_WHITE      = (255, 255, 255)
C_CHAOS      = (255, 180, 0)

UNIT_COLORS = [
    (0, 230, 255),   # cyan
    (0, 255, 120),   # green
    (255, 80, 220),  # magenta
    (255, 200, 0),   # gold
    (80, 140, 255),  # blue
]


# ─── PCM AUDIO ───────────────────────────────────────────────────────────────
def _make_tone(freq=440, dur=0.06, vol=0.18, sr=44100, waveform="sine"):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    if waveform == "sine":
        wave = np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == "noise":
        wave = np.random.uniform(-1, 1, len(t))
    else:
        wave = np.sin(2 * np.pi * freq * t)
    wave = (wave * vol * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)

def build_sounds():
    move_snd = _make_tone(freq=320, dur=0.04, vol=0.06, waveform="sine")
    hit_snd  = _make_tone(freq=80,  dur=0.18, vol=0.35, waveform="noise")
    chaos_snd= _make_tone(freq=600, dur=0.10, vol=0.20, waveform="square")
    spawn_snd= _make_tone(freq=200, dur=0.05, vol=0.08, waveform="sine")
    return {"move": move_snd, "hit": hit_snd, "chaos": chaos_snd, "spawn": spawn_snd}


# ─── PARTICLE ────────────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, math.tau)
        speed = random.uniform(1.5, 5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.uniform(0.3, 0.7)
        self.max_life = self.life
        self.color = color
        self.size = random.randint(2, 5)

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.92
        self.vy *= 0.92
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        alpha = self.life / self.max_life
        r, g, b = self.color
        color = (int(r*alpha), int(g*alpha), int(b*alpha))
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), max(1, int(self.size*alpha)))


# ─── UNIT (one finger) ───────────────────────────────────────────────────────
class Unit:
    def __init__(self, idx):
        self.idx   = idx
        self.color = UNIT_COLORS[idx % len(UNIT_COLORS)]
        self.x     = WIN_W // 2
        self.y     = WIN_H // 2
        self.trail = []      # list of (x, y)
        self.alive = True
        self.pulse = 0.0

    def update(self, x, y, dt):
        self.x, self.y = x, y
        self.trail.append((x, y))
        if len(self.trail) > TRAIL_LEN:
            self.trail.pop(0)
        self.pulse = (self.pulse + dt * 4) % math.tau

    def draw(self, surf):
        # Trail
        for i, (tx, ty) in enumerate(self.trail):
            frac  = i / max(len(self.trail)-1, 1)
            alpha = int(frac * 160)
            r, g, b = self.color
            color = (r, g, b)
            size  = max(1, int(frac * 8))
            s = pygame.Surface((size*2+2, size*2+2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (size+1, size+1), size)
            surf.blit(s, (tx - size - 1, ty - size - 1))

        # Glow ring
        glow_r = 22 + int(math.sin(self.pulse) * 5)
        glow_surf = pygame.Surface((glow_r*2+4, glow_r*2+4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color, 55), (glow_r+2, glow_r+2), glow_r)
        surf.blit(glow_surf, (int(self.x) - glow_r - 2, int(self.y) - glow_r - 2))

        # Core circle
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), 12)
        pygame.draw.circle(surf, C_WHITE,    (int(self.x), int(self.y)), 12, 2)

        # Finger label
        font = pygame.font.SysFont("consolas", 11, bold=True)
        label = font.render(f"F{self.idx+1}", True, C_WHITE)
        surf.blit(label, (int(self.x) - label.get_width()//2, int(self.y) - 24))

    @property
    def rect(self):
        return pygame.Rect(self.x - 12, self.y - 12, 24, 24)


# ─── ENEMY ───────────────────────────────────────────────────────────────────
class Enemy:
    def __init__(self, chaos=False):
        edge = random.randint(0, 3)
        if edge == 0:   self.x, self.y = random.uniform(0, WIN_W), -20
        elif edge == 1: self.x, self.y = WIN_W + 20, random.uniform(0, WIN_H)
        elif edge == 2: self.x, self.y = random.uniform(0, WIN_W), WIN_H + 20
        else:           self.x, self.y = -20, random.uniform(0, WIN_H)
        self.speed   = ENEMY_SPEED * (1.6 if chaos else 1.0) * random.uniform(0.8, 1.4)
        self.size    = random.randint(10, 20)
        self.pulse   = random.uniform(0, math.tau)
        self.alive   = True

    def update(self, targets, dt):
        if not targets:
            return
        # Chase nearest unit
        nearest = min(targets, key=lambda u: math.hypot(u.x - self.x, u.y - self.y))
        dx = nearest.x - self.x
        dy = nearest.y - self.y
        dist = math.hypot(dx, dy) or 1
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
        self.pulse += dt * 5

    def draw(self, surf):
        glow_r = self.size + 8 + int(math.sin(self.pulse) * 4)
        glow_surf = pygame.Surface((glow_r*2+4, glow_r*2+4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*C_ENEMY_GLOW, 60), (glow_r+2, glow_r+2), glow_r)
        surf.blit(glow_surf, (int(self.x) - glow_r - 2, int(self.y) - glow_r - 2))
        pygame.draw.circle(surf, C_ENEMY, (int(self.x), int(self.y)), self.size)
        # X mark
        pygame.draw.line(surf, C_WHITE, (int(self.x)-6, int(self.y)-6), (int(self.x)+6, int(self.y)+6), 2)
        pygame.draw.line(surf, C_WHITE, (int(self.x)+6, int(self.y)-6), (int(self.x)-6, int(self.y)+6), 2)

    @property
    def rect(self):
        return pygame.Rect(self.x - self.size, self.y - self.size, self.size*2, self.size*2)


# ─── HAND TRACKER ────────────────────────────────────────────────────────────
# Fixes vs v1:
#   • Lower detection/tracking confidence thresholds (0.4 / 0.3)
#   • model_complexity=0  →  fastest model, fewer missed frames
#   • CLAHE pre-processing  →  works in poor / variable lighting
#   • Exponential smoothing (α=0.55) per fingertip  →  no jitter
#   • Last-known-position hold for up to HOLD_FRAMES frames before dropout
#   • Confidence score broadcast so HUD can show a tracking quality bar
#   • Debug dots burned into the overlay for every detected tip
class HandTracker:
    SMOOTH_ALPHA  = 0.55   # 0=no update, 1=raw (0.55 = responsive but stable)
    HOLD_FRAMES   = 6      # frames to keep last position when MP drops the hand

    def __init__(self):
        self.cap           = None
        self.frame_rgb     = None
        # smoothed positions: {finger_key: [sx, sy]}
        self._smooth       = {}
        # hold counters: {finger_key: int}
        self._hold         = {}
        self.fingertips    = []   # [(x_norm, y_norm), ...]
        self.confidence    = 0.0  # 0-1 tracking quality indicator
        self._lock         = threading.Lock()
        self._running      = False
        self.hands         = None
        self._clahe        = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self._open_cam()

    def _open_cam(self):
        self.cap = cv2.VideoCapture(0)
        # Ask for the highest resolution the cam supports; MP works better with more pixels
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.cap.set(cv2.CAP_PROP_FPS,            30)
        # Fall back to 640×480 if camera refused the higher res
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if actual_w < 640:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if MEDIAPIPE_OK:
            self.hands = _mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,            # fastest model → fewer missed frames
                min_detection_confidence=0.4,  # was 0.6 — catch partial/angled hands
                min_tracking_confidence=0.3,   # was 0.5 — keep tracking through movement
            )
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    # ── CLAHE brightness normalisation ───────────────────────────────────────
    def _enhance(self, bgr_frame):
        """Apply CLAHE to the luminance channel so tracking works in dim rooms."""
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ── Exponential smoothing ─────────────────────────────────────────────────
    def _smooth_tip(self, key, raw_x, raw_y):
        α = self.SMOOTH_ALPHA
        if key not in self._smooth:
            self._smooth[key] = [raw_x, raw_y]
        else:
            self._smooth[key][0] = α * raw_x + (1 - α) * self._smooth[key][0]
            self._smooth[key][1] = α * raw_y + (1 - α) * self._smooth[key][1]
        self._hold[key] = self.HOLD_FRAMES
        return tuple(self._smooth[key])

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame   = cv2.flip(frame, 1)
            enhanced = self._enhance(frame)
            tips    = []
            conf    = 0.0

            if MEDIAPIPE_OK and self.hands:
                rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False          # small speed-up for MP
                res = self.hands.process(rgb)
                rgb.flags.writeable = True

                detected_keys = set()
                if res.multi_hand_landmarks:
                    conf = 1.0
                    for hand_idx, hlm in enumerate(res.multi_hand_landmarks):
                        _mp_draw.draw_landmarks(
                            frame, hlm, _mp_hands.HAND_CONNECTIONS,
                            _mp_draw.DrawingSpec(color=(0,255,200), thickness=2, circle_radius=3),
                            _mp_draw.DrawingSpec(color=(0,180,255), thickness=2),
                        )
                        for tip_id in FINGER_TIPS:
                            lm  = hlm.landmark[tip_id]
                            key = (hand_idx, tip_id)
                            detected_keys.add(key)
                            sx, sy = self._smooth_tip(key, lm.x, lm.y)
                            tips.append((sx, sy))
                            # Draw debug dot on the raw frame overlay
                            px = int(lm.x * frame.shape[1])
                            py = int(lm.y * frame.shape[0])
                            cv2.circle(frame, (px, py), 7, (0, 255, 100), -1)
                            cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

                # Decay hold counters for keys not seen this frame
                for key in list(self._hold.keys()):
                    if key not in detected_keys:
                        self._hold[key] -= 1
                        if self._hold[key] > 0 and key in self._smooth:
                            # Inject last-known smoothed position so unit doesn't vanish
                            tips.append(tuple(self._smooth[key]))
                            conf = max(conf, 0.4)
                        else:
                            self._hold.pop(key, None)
                            self._smooth.pop(key, None)
            else:
                # ── Skin-colour fallback (works without MediaPipe) ──────────
                # Convert to YCrCb and isolate skin tone
                ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
                mask  = cv2.inRange(ycrcb,
                                    np.array([0, 133, 77],  np.uint8),
                                    np.array([255, 173, 127], np.uint8))
                mask  = cv2.GaussianBlur(mask, (5, 5), 0)
                _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                    if cv2.contourArea(c) < 300:
                        continue
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = M["m10"]/M["m00"] / frame.shape[1]
                        cy = M["m01"]/M["m00"] / frame.shape[0]
                        tips.append((cx, cy))
                        conf = 0.6
                        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

            # Build overlay image
            small = cv2.resize(frame, (CAM_W, CAM_H))
            small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            with self._lock:
                self.frame_rgb  = small
                self.fingertips = tips[:10]
                self.confidence = conf

    def get_frame_and_tips(self):
        with self._lock:
            return self.frame_rgb, list(self.fingertips), self.confidence

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()


# ─── DRAW GRID ───────────────────────────────────────────────────────────────
def draw_grid(surf):
    for x in range(0, WIN_W, 60):
        pygame.draw.line(surf, C_GRID, (x, 0), (x, WIN_H))
    for y in range(0, WIN_H, 60):
        pygame.draw.line(surf, C_GRID, (0, y), (WIN_W, y))


# ─── HUD ─────────────────────────────────────────────────────────────────────
def draw_hud(surf, score, lives, chaos, active_units, font_big, font_sm):
    # Score
    sc_txt = font_big.render(f"SCORE  {score:06d}", True, C_HUD)
    surf.blit(sc_txt, (WIN_W//2 - sc_txt.get_width()//2, 10))

    # Lives
    for i in range(lives):
        pygame.draw.circle(surf, (0, 255, 120), (WIN_W - 30 - i*28, 24), 9)

    # Active fingers
    info = font_sm.render(f"FINGERS: {active_units}", True, C_HUD)
    surf.blit(info, (10, 10))

    # Chaos badge
    if chaos:
        badge = font_big.render("⚡ CHAOS MODE ⚡", True, C_CHAOS)
        x = WIN_W//2 - badge.get_width()//2
        pygame.draw.rect(surf, (40, 20, 0), (x-8, WIN_H-46, badge.get_width()+16, 36), border_radius=8)
        surf.blit(badge, (x, WIN_H-44))


# ─── DRAW CAMERA OVERLAY ─────────────────────────────────────────────────────
def draw_cam_overlay(surf, cam_frame, confidence=0.0):
    if cam_frame is None:
        return
    cam_surf = pygame.surfarray.make_surface(cam_frame.swapaxes(0, 1))
    pygame.draw.rect(surf, (30, 30, 60), (CAM_X-3, CAM_Y-3, CAM_W+6, CAM_H+6), border_radius=8)
    surf.blit(cam_surf, (CAM_X, CAM_Y))
    pygame.draw.rect(surf, (80, 80, 180), (CAM_X-3, CAM_Y-3, CAM_W+6, CAM_H+6), 2, border_radius=8)
    font = pygame.font.SysFont("consolas", 10)
    lbl = font.render("LIVE CAM", True, (120, 120, 200))
    surf.blit(lbl, (CAM_X + 4, CAM_Y + 4))
    # Tracking confidence bar
    bar_w = int(CAM_W * confidence)
    bar_y = CAM_Y + CAM_H + 4
    pygame.draw.rect(surf, (30, 30, 60), (CAM_X, bar_y, CAM_W, 6), border_radius=3)
    bar_color = (int(255 * (1 - confidence)), int(200 * confidence), 60)
    if bar_w > 0:
        pygame.draw.rect(surf, bar_color, (CAM_X, bar_y, bar_w, 6), border_radius=3)
    status = "TRACKING" if confidence > 0.5 else ("PARTIAL" if confidence > 0 else "NO HAND")
    st_col = (0, 220, 100) if confidence > 0.5 else ((255, 180, 0) if confidence > 0 else (255, 60, 60))
    st_txt = font.render(status, True, st_col)
    surf.blit(st_txt, (CAM_X + CAM_W - st_txt.get_width() - 4, CAM_Y + 4))


# ─── DEATH SCREEN ────────────────────────────────────────────────────────────
def draw_death(surf, score, font_big, font_sm):
    overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surf.blit(overlay, (0, 0))
    t1 = font_big.render("ARMY WIPED OUT", True, C_ENEMY)
    t2 = font_sm.render(f"Final Score: {score:06d}", True, C_WHITE)
    t3 = font_sm.render("Press  R  to Restart   |   Q  to Quit", True, (150, 150, 255))
    surf.blit(t1, (WIN_W//2 - t1.get_width()//2, WIN_H//2 - 60))
    surf.blit(t2, (WIN_W//2 - t2.get_width()//2, WIN_H//2))
    surf.blit(t3, (WIN_W//2 - t3.get_width()//2, WIN_H//2 + 50))


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Clone Finger Army")
    clock  = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 28, bold=True)
    font_sm  = pygame.font.SysFont("consolas", 18)

    sounds = build_sounds()
    tracker = HandTracker()

    def reset():
        return {
            "units":       [],             # active Unit objects
            "enemies":     [],
            "particles":   [],
            "score":       0,
            "lives":       5,
            "chaos":       False,
            "dead":        False,
            "spawn_timer": 0.0,
            "move_timer":  0.0,
            "score_timer": 0.0,
        }

    state = reset()

    while True:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)

        # ── Events ────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                tracker.stop(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    tracker.stop(); pygame.quit(); sys.exit()
                if ev.key == pygame.K_r:
                    state = reset()
                if ev.key == pygame.K_c and not state["dead"]:
                    state["chaos"] = not state["chaos"]
                    sounds["chaos"].play()
            if ev.type == pygame.VIDEORESIZE:
                pass  # pygame RESIZABLE handles it

        if state["dead"]:
            screen.fill(C_BG)
            draw_grid(screen)
            draw_death(screen, state["score"], font_big, font_sm)
            cam_frame, _, conf = tracker.get_frame_and_tips()
            draw_cam_overlay(screen, cam_frame, conf)
            pygame.display.flip()
            continue

        # ── Get fingertips ─────────────────────────────────────────────────
        cam_frame, raw_tips, tracking_conf = tracker.get_frame_and_tips()

        # Map normalised fingertip coords → game window coords
        # Exclude area occupied by camera overlay
        active_positions = []
        for (nx, ny) in raw_tips:
            gx = int(nx * WIN_W)
            gy = int(ny * WIN_H)
            active_positions.append((gx, gy))

        # Sync units list with detected fingers
        while len(state["units"]) < len(active_positions):
            state["units"].append(Unit(len(state["units"])))
        # Mark excess units inactive
        for i, u in enumerate(state["units"]):
            u.alive = i < len(active_positions)

        for i, (gx, gy) in enumerate(active_positions):
            state["units"][i].update(gx, gy, dt)

        # ── Movement sound ────────────────────────────────────────────────
        state["move_timer"] += dt
        if active_positions and state["move_timer"] > 0.12:
            sounds["move"].play()
            state["move_timer"] = 0.0

        # ── Spawn enemies ─────────────────────────────────────────────────
        state["spawn_timer"] += dt
        spawn_every = ENEMY_SPAWN / (CHAOS_MULT if state["chaos"] else 1)
        if state["spawn_timer"] >= spawn_every:
            state["spawn_timer"] = 0.0
            count = random.randint(2, 5) if state["chaos"] else 1
            for _ in range(count):
                state["enemies"].append(Enemy(chaos=state["chaos"]))
            sounds["spawn"].play()

        # ── Update enemies ────────────────────────────────────────────────
        alive_units = [u for u in state["units"] if u.alive]
        for e in state["enemies"]:
            e.update(alive_units, dt)

        # ── Collision detection ───────────────────────────────────────────
        hit = False
        for e in state["enemies"][:]:
            for u in alive_units:
                if e.rect.colliderect(u.rect):
                    hit = True
                    state["lives"] -= 1
                    sounds["hit"].play()
                    # Spawn particles at collision
                    for _ in range(18):
                        state["particles"].append(Particle(u.x, u.y, u.color))
                    state["enemies"].remove(e)
                    break
            # Remove off-screen enemies (got lost)
            if e in state["enemies"]:
                if e.x < -60 or e.x > WIN_W+60 or e.y < -60 or e.y > WIN_H+60:
                    state["enemies"].remove(e)

        if hit and state["lives"] <= 0:
            state["dead"] = True

        # ── Score ──────────────────────────────────────────────────────────
        state["score_timer"] += dt
        if state["score_timer"] >= 0.5:
            state["score_timer"] = 0
            bonus = CHAOS_MULT if state["chaos"] else 1
            state["score"] += len(alive_units) * 10 * bonus

        # ── Update particles ───────────────────────────────────────────────
        state["particles"] = [p for p in state["particles"] if p.update(dt)]

        # ── Draw ──────────────────────────────────────────────────────────
        screen.fill(C_BG)
        draw_grid(screen)

        # Draw units
        for u in state["units"]:
            if u.alive:
                u.draw(screen)

        # Draw enemies
        for e in state["enemies"]:
            e.draw(screen)

        # Draw particles
        for p in state["particles"]:
            p.draw(screen)

        # Camera overlay
        draw_cam_overlay(screen, cam_frame, tracking_conf)

        # HUD
        draw_hud(screen, state["score"], max(state["lives"], 0),
                 state["chaos"], len(alive_units), font_big, font_sm)

        # No fingers detected — prompt
        if not active_positions:
            hint = font_sm.render("✋  Show your fingers to the camera!", True, (180, 180, 100))
            screen.blit(hint, (WIN_W//2 - hint.get_width()//2, WIN_H//2 - 14))

        pygame.display.flip()

    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()