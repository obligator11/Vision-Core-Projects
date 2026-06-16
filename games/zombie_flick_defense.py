#!/usr/bin/env python3
"""
Zombie Flick Defense
====================
Gesture-controlled zombie defense game using webcam hand tracking.

Controls:
  - Point with index finger → aim at zombies
  - Fast FLICK gesture      → attack (slash)
  - Pinch (thumb + index)   → special area attack

Requirements:
  pip install pygame opencv-python mediapipe numpy
"""

import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import sys
from collections import deque

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
GAME_W, GAME_H = 1280, 720
CAM_W,  CAM_H  = 240, 180          # overlay size (top-right)
CAM_PAD        = 12
FPS            = 60

# Colours
C_BG           = (10,  12,  8)
C_HUD          = (180, 230, 80)
C_COMBO        = (255, 200,  0)
C_FLASH_HIT    = (200,  0,   0, 120)
C_FLASH_COMBO  = (255, 200,  0,  80)
C_FLASH_SPEC   = (100, 200, 255, 100)
C_BLOOD        = [(180, 0, 0), (220, 20, 20), (140, 0, 0)]
C_PARTICLE     = [(255, 80, 0), (255, 160, 0), (200, 60, 0)]

# Gesture thresholds
FLICK_SPEED_THRESH   = 18          # px/frame smoothed velocity
PINCH_DIST_THRESH    = 0.07        # normalised hand distance
FLICK_COOLDOWN       = 0.25        # seconds between flick attacks
PINCH_COOLDOWN       = 2.0         # seconds between special attacks
GESTURE_HISTORY_LEN  = 8

# Zombie config
BASE_ZOMBIE_SPEED    = 0.9
SPEED_SCALE_PER_WAVE = 0.15
BASE_SPAWN_COUNT     = 6
SPAWN_PER_WAVE       = 3
SLASH_RADIUS         = 110         # melee range for flick
SPECIAL_RADIUS       = 220         # area blast radius

# Slow-mo
SLOTIME_DURATION     = 0.55        # seconds
SLOTIME_SCALE        = 0.22        # time multiplier

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a, b, t):
    return a + (b - a) * t


def rand_edge(w, h, margin=80):
    side = random.randint(0, 3)
    if side == 0: return random.randint(0, w), -margin
    if side == 1: return random.randint(0, w), h + margin
    if side == 2: return -margin, random.randint(0, h)
    return w + margin, random.randint(0, h)


# ─────────────────────────────────────────────
#  SOUND  (procedural, no files needed)
# ─────────────────────────────────────────────

class SoundEngine:
    def __init__(self):
        # Try mono first; fall back gracefully to whatever the system gives us
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        freq, size, channels = pygame.mixer.get_init()
        self._channels = channels   # 1 = mono, 2 = stereo
        self.slash   = self._make_slash()
        self.hit     = self._make_hit()
        self.combo   = self._make_combo()
        self.special = self._make_special()

    def _synth(self, mono):
        """Accept a 1-D int16 array; duplicate to stereo if mixer needs it."""
        arr = np.clip(mono, -32767, 32767).astype(np.int16)
        if self._channels == 2:
            arr = np.column_stack((arr, arr))   # shape (N, 2)
        try:
            return pygame.sndarray.make_sound(arr)
        except Exception:
            return None

    def _make_slash(self):
        t = np.linspace(0, 0.12, int(44100 * 0.12))
        freq = np.linspace(800, 200, len(t))
        wave = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 30) * 28000).astype(np.float32)
        noise = np.random.uniform(-1, 1, len(t)).astype(np.float32) * 8000 * np.exp(-t * 20)
        return self._synth((wave + noise).astype(np.int16))

    def _make_hit(self):
        t = np.linspace(0, 0.15, int(44100 * 0.15))
        freq = np.linspace(120, 60, len(t))
        wave = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 18) * 24000).astype(np.float32)
        noise = np.random.uniform(-1, 1, len(t)).astype(np.float32) * 6000 * np.exp(-t * 25)
        return self._synth((wave + noise).astype(np.int16))

    def _make_combo(self):
        t = np.linspace(0, 0.2, int(44100 * 0.2))
        freqs = [440, 550, 660]
        wave = np.zeros(len(t), dtype=np.float32)
        for i, f in enumerate(freqs):
            start = int(i * 0.05 * 44100)
            end   = min(len(t), start + int(0.12 * 44100))
            sub_t = t[:end - start]
            wave[start:end] += np.sin(2 * np.pi * f * sub_t) * np.exp(-sub_t * 15) * 18000
        return self._synth(wave.astype(np.int16))

    def _make_special(self):
        t = np.linspace(0, 0.4, int(44100 * 0.4))
        wave = np.zeros(len(t), dtype=np.float32)
        for f in [80, 160, 240]:
            wave += np.sin(2 * np.pi * f * t) * np.exp(-t * 8) * 10000
        noise = np.random.uniform(-1, 1, len(t)).astype(np.float32) * 5000 * np.exp(-t * 6)
        return self._synth((wave + noise).astype(np.int16))

    def play(self, name):
        try:
            snd = getattr(self, name, None)
            if snd:
                snd.play()
        except Exception:
            pass


# ─────────────────────────────────────────────
#  PARTICLES
# ─────────────────────────────────────────────

class Particle:
    def __init__(self, x, y, colour=None, blood=False):
        self.x, self.y = x, y
        angle  = random.uniform(0, 2 * math.pi)
        speed  = random.uniform(2, 10) if blood else random.uniform(3, 8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - (random.uniform(2, 5) if blood else 0)
        self.life  = random.uniform(0.3, 0.7)
        self.age   = 0
        self.size  = random.randint(3, 9) if blood else random.randint(2, 6)
        self.colour = colour or (random.choice(C_BLOOD) if blood else random.choice(C_PARTICLE))

    def update(self, dt):
        self.x  += self.vx * dt * 60
        self.y  += self.vy * dt * 60
        self.vy += 0.3 * dt * 60   # gravity
        self.age += dt

    def alive(self):
        return self.age < self.life

    def draw(self, surf):
        alpha = 1 - self.age / self.life
        r, g, b = self.colour
        colour = (int(r * alpha), int(g * alpha), int(b * alpha))
        sz = max(1, int(self.size * alpha))
        pygame.draw.circle(surf, colour, (int(self.x), int(self.y)), sz)


# ─────────────────────────────────────────────
#  DAMAGE NUMBERS
# ─────────────────────────────────────────────

class DmgNumber:
    def __init__(self, x, y, text, colour=(255, 60, 60)):
        self.x, self.y = x, y
        self.vy     = -2.5
        self.text   = text
        self.colour = colour
        self.life   = 1.0
        self.age    = 0

    def update(self, dt):
        self.y   += self.vy * dt * 60
        self.vy  += 0.05 * dt * 60
        self.age += dt

    def alive(self):
        return self.age < self.life

    def draw(self, surf, font):
        alpha = max(0, 1 - self.age / self.life)
        scale = 1 + (1 - self.age / self.life) * 0.4
        r, g, b = self.colour
        col   = (int(r * alpha), int(g * alpha), int(b * alpha))
        size  = max(8, int(28 * scale))
        f     = pygame.font.SysFont("Arial Black", size, bold=True)
        img   = f.render(self.text, True, col)
        surf.blit(img, img.get_rect(center=(int(self.x), int(self.y))))


# ─────────────────────────────────────────────
#  SLASH EFFECT
# ─────────────────────────────────────────────

class SlashEffect:
    def __init__(self, x, y, radius, special=False):
        self.x, self.y = x, y
        self.radius  = radius
        self.special = special
        self.age     = 0
        self.life    = 0.35

    def update(self, dt):
        self.age += dt

    def alive(self):
        return self.age < self.life

    def draw(self, surf):
        t = self.age / self.life
        alpha = int((1 - t) * 180)
        r_now = int(self.radius * (0.4 + t * 0.6))
        colour = (100, 200, 255) if self.special else (255, 255, 255)
        ring_surf = pygame.Surface((r_now * 2 + 4, r_now * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(ring_surf, (*colour, alpha), (r_now + 2, r_now + 2), r_now, max(1, int(4 * (1 - t))))
        surf.blit(ring_surf, (self.x - r_now - 2, self.y - r_now - 2))
        # inner glow
        glow_surf = pygame.Surface((r_now * 2, r_now * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*colour, int(alpha * 0.3)), (r_now, r_now), r_now)
        surf.blit(glow_surf, (self.x - r_now, self.y - r_now))


# ─────────────────────────────────────────────
#  ZOMBIE
# ─────────────────────────────────────────────

ZOMBIE_SHAPES = [
    # (body_w, body_h, head_r)
    (28, 40, 14),
    (24, 44, 12),
    (32, 36, 15),
]

class Zombie:
    _id = 0

    def __init__(self, wave):
        Zombie._id += 1
        self.id    = Zombie._id
        self.x, self.y = rand_edge(GAME_W, GAME_H)
        self.hp    = 1
        speed_mult = 1 + (wave - 1) * SPEED_SCALE_PER_WAVE
        self.speed = (BASE_ZOMBIE_SPEED + random.uniform(-0.15, 0.25)) * speed_mult
        self.shape = random.choice(ZOMBIE_SHAPES)
        # colour variation
        g = random.randint(80, 130)
        self.colour      = (random.randint(30, 60), g, random.randint(20, 50))
        self.eye_colour  = (255, random.randint(0, 80), 0)
        self.highlighted = False
        self.death_timer = 0
        self.dead        = False
        self.wobble      = random.uniform(0, math.pi * 2)
        self.wobble_sp   = random.uniform(3, 6)

    def update(self, dt, px, py, time_scale=1.0):
        if self.dead:
            self.death_timer += dt
            return
        eff_dt = dt * time_scale
        dx = px - self.x
        dy = py - self.y
        d  = math.hypot(dx, dy)
        if d > 1:
            self.x += (dx / d) * self.speed * eff_dt * 60
            self.y += (dy / d) * self.speed * eff_dt * 60
        self.wobble += self.wobble_sp * eff_dt

    def draw(self, surf):
        if self.dead:
            return
        bw, bh, hr = self.shape
        cx, cy = int(self.x), int(self.y)
        wobble_x = int(math.sin(self.wobble) * 2)

        # shadow
        pygame.draw.ellipse(surf, (0, 0, 0, 80),
                            (cx - bw // 2 + 2, cy + bh // 2 - 4, bw, 10))

        # highlight ring
        if self.highlighted:
            pygame.draw.circle(surf, (255, 80, 80), (cx, cy), hr + bw // 2 + 6, 3)

        # body
        body_col = (min(255, self.colour[0] + 40), self.colour[1], self.colour[2]) \
                   if self.highlighted else self.colour
        pygame.draw.rect(surf, body_col,
                         (cx - bw // 2 + wobble_x, cy - bh // 2, bw, bh), border_radius=4)

        # arms (raised)
        arm_y = cy - bh // 4
        pygame.draw.line(surf, self.colour,
                         (cx - bw // 2 + wobble_x, arm_y),
                         (cx - bw // 2 - 12 + wobble_x, arm_y - 18), 5)
        pygame.draw.line(surf, self.colour,
                         (cx + bw // 2 + wobble_x, arm_y),
                         (cx + bw // 2 + 12 + wobble_x, arm_y - 18), 5)

        # head
        pygame.draw.circle(surf, self.colour, (cx + wobble_x, cy - bh // 2 - hr), hr)

        # eyes
        ex = 5
        ey = cy - bh // 2 - hr - 2
        pygame.draw.circle(surf, self.eye_colour, (cx - ex + wobble_x, ey), 4)
        pygame.draw.circle(surf, self.eye_colour, (cx + ex + wobble_x, ey), 4)
        pygame.draw.circle(surf, (255, 255, 0), (cx - ex + wobble_x, ey), 2)
        pygame.draw.circle(surf, (255, 255, 0), (cx + ex + wobble_x, ey), 2)

        # mouth
        pygame.draw.arc(surf, (150, 0, 0),
                        (cx - 7 + wobble_x, ey + 6, 14, 8),
                        math.pi, 2 * math.pi, 2)


# ─────────────────────────────────────────────
#  HAND TRACKER
# ─────────────────────────────────────────────

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55
        )
        self.history  = deque(maxlen=GESTURE_HISTORY_LEN)
        self.smoothed = None          # (x, y) in game coords
        self.velocity = (0, 0)
        self.speed    = 0
        self.pinching = False

        # camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cam_frame = None

    def process(self, game_w, game_h):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        self.cam_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        raw_pos  = None
        pinching = False

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark

            # index fingertip (8) = aim point
            ix = lm[8].x
            iy = lm[8].y

            # map to game coords (flipped x since cam is mirrored back)
            gx = ix * game_w
            gy = iy * game_h
            raw_pos = (gx, gy)

            # pinch: thumb (4) vs index (8)
            thumb = (lm[4].x, lm[4].y)
            index = (lm[8].x, lm[8].y)
            pd = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
            pinching = pd < PINCH_DIST_THRESH

        self.pinching = pinching

        if raw_pos:
            self.history.append(raw_pos)
            if len(self.history) >= 2:
                # smoothing: exponential moving average
                if self.smoothed is None:
                    self.smoothed = raw_pos
                else:
                    alpha = 0.55
                    sx = lerp(self.smoothed[0], raw_pos[0], alpha)
                    sy = lerp(self.smoothed[1], raw_pos[1], alpha)
                    self.velocity = (sx - self.smoothed[0], sy - self.smoothed[1])
                    self.speed    = math.hypot(*self.velocity)
                    self.smoothed = (sx, sy)
        else:
            self.history.clear()
            if self.smoothed:
                self.speed = max(0, self.speed - 1)

    def get_cam_surface(self):
        if self.cam_frame is None:
            return None
        small = cv2.resize(self.cam_frame, (CAM_W, CAM_H))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf  = pygame.surfarray.make_surface(small.swapaxes(0, 1))
        return surf

    def release(self):
        self.cap.release()


# ─────────────────────────────────────────────
#  SCREEN FLASH
# ─────────────────────────────────────────────

class ScreenFlash:
    def __init__(self):
        self.flashes = []   # [(colour_rgba, life, age)]

    def add(self, colour, duration=0.15):
        self.flashes.append([colour, duration, 0])

    def update(self, dt):
        self.flashes = [[c, l, a + dt] for c, l, a in self.flashes if a + dt < l]

    def draw(self, surf):
        for colour, life, age in self.flashes:
            t = age / life
            r, g, b, a_max = colour
            a = int(a_max * (1 - t))
            overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            overlay.fill((r, g, b, a))
            surf.blit(overlay, (0, 0))


# ─────────────────────────────────────────────
#  MAIN GAME
# ─────────────────────────────────────────────

class ZombieFlickDefense:
    def __init__(self):
        pygame.init()
        # Resizable window; all game logic runs at fixed GAME_W x GAME_H,
        # then we scale-blit onto the actual window each frame.
        self.window = pygame.display.set_mode(
            (GAME_W, GAME_H),
            pygame.RESIZABLE
        )
        pygame.display.set_caption("Zombie Flick Defense")
        # Fixed-resolution surface we always render into
        self.screen = pygame.Surface((GAME_W, GAME_H))
        self.clock  = pygame.time.Clock()

        self.font_lg = pygame.font.SysFont("Arial Black", 64, bold=True)
        self.font_md = pygame.font.SysFont("Arial Black", 32, bold=True)
        self.font_sm = pygame.font.SysFont("Arial",       22)
        self.font_xs = pygame.font.SysFont("Arial",       16)

        self.sounds  = SoundEngine()
        self.tracker = HandTracker()
        self.flash   = ScreenFlash()

        self.reset()

    def reset(self):
        self.player_x = GAME_W // 2
        self.player_y = GAME_H // 2
        self.zombies   = []
        self.particles = []
        self.dmg_nums  = []
        self.effects   = []

        self.wave       = 1
        self.score      = 0
        self.combo      = 0
        self.combo_timer = 0
        self.COMBO_RESET = 2.5
        self.kills      = 0
        self.alive_flag = True
        self.game_over  = False

        self.flick_cd   = 0
        self.pinch_cd   = 0
        self.last_pinch = False

        self.slow_timer   = 0
        self.time_scale   = 1.0

        self.wave_clear_timer = 0
        self.spawning_wave    = False
        self._spawn_wave(self.wave)

        self.streak_surf = None
        self.streak_age  = 0

    def _spawn_wave(self, wave):
        count = BASE_SPAWN_COUNT + (wave - 1) * SPAWN_PER_WAVE
        for _ in range(count):
            self.zombies.append(Zombie(wave))
        self.spawning_wave = True

    # ── ATTACK LOGIC ────────────────────────────

    def _do_slash(self, special=False):
        if not self.tracker.smoothed:
            return
        ax, ay = self.tracker.smoothed
        radius = SPECIAL_RADIUS if special else SLASH_RADIUS
        colour = (100, 200, 255) if special else (255, 255, 220)

        self.effects.append(SlashEffect(ax, ay, radius, special=special))
        self.sounds.play("special" if special else "slash")
        self.flash.add((200, 200, 255, 60) if special else (255, 255, 255, 50), 0.1)

        hit_any = False
        for z in self.zombies:
            if z.dead:
                continue
            d = dist((ax, ay), (z.x, z.y))
            if d < radius:
                self._kill_zombie(z, special=special)
                hit_any = True

        if hit_any:
            self.sounds.play("hit")
            self.flash.add((200, 0, 0, 80), 0.12)
            # slow-mo on good hits
            if not special and self.combo >= 3:
                self.slow_timer = SLOTIME_DURATION

    def _kill_zombie(self, zombie, special=False):
        zombie.dead = True
        self.kills  += 1
        base_pts     = 10 * self.wave
        combo_mult   = 1 + self.combo * 0.5
        pts          = int(base_pts * combo_mult)
        self.score  += pts
        self.combo  += 1
        self.combo_timer = self.COMBO_RESET

        # particles
        for _ in range(20 if special else 12):
            self.particles.append(Particle(zombie.x, zombie.y, blood=True))
        for _ in range(6):
            col = random.choice(C_PARTICLE)
            self.particles.append(Particle(zombie.x, zombie.y, colour=col))

        # damage number
        col = (255, 200, 0) if self.combo > 3 else (255, 80, 80)
        self.dmg_nums.append(DmgNumber(zombie.x, zombie.y - 30, f"+{pts}", col))

        # combo streak visual at 5, 10, 15 …
        if self.combo % 5 == 0 and self.combo > 0:
            self.sounds.play("combo")
            self.flash.add(C_FLASH_COMBO, 0.2)
            self._make_streak_text(f"{self.combo}x COMBO!")

    def _make_streak_text(self, text):
        surf = self.font_lg.render(text, True, (255, 220, 0))
        self.streak_surf = surf
        self.streak_age  = 0

    # ── UPDATE ───────────────────────────────────

    def update(self, dt):
        if self.game_over:
            return

        # slow-mo
        if self.slow_timer > 0:
            self.slow_timer -= dt
            self.time_scale = lerp(self.time_scale, SLOTIME_SCALE, 0.2)
        else:
            self.time_scale = lerp(self.time_scale, 1.0, 0.15)

        self.tracker.process(GAME_W, GAME_H)

        # cooldowns
        self.flick_cd  = max(0, self.flick_cd - dt)
        self.pinch_cd  = max(0, self.pinch_cd - dt)
        self.combo_timer = max(0, self.combo_timer - dt)
        if self.combo_timer <= 0 and self.combo > 0:
            self.combo = 0

        # ── gesture detection ────────────────────
        if self.tracker.smoothed:
            # FLICK → attack
            if self.tracker.speed > FLICK_SPEED_THRESH and self.flick_cd <= 0:
                self._do_slash(special=False)
                self.flick_cd = FLICK_COOLDOWN

            # PINCH → special
            pinching_now = self.tracker.pinching
            if pinching_now and not self.last_pinch and self.pinch_cd <= 0:
                self._do_slash(special=True)
                self.pinch_cd  = PINCH_COOLDOWN
                self.flash.add(C_FLASH_SPEC, 0.25)
            self.last_pinch = pinching_now

        # ── highlight nearest zombie to aim ──────
        for z in self.zombies:
            z.highlighted = False
        if self.tracker.smoothed:
            nearest, nearest_d = None, 9999
            for z in self.zombies:
                if not z.dead:
                    d = dist(self.tracker.smoothed, (z.x, z.y))
                    if d < nearest_d:
                        nearest_d = d
                        nearest   = z
            if nearest and nearest_d < 150:
                nearest.highlighted = True

        # ── zombies ──────────────────────────────
        for z in self.zombies:
            z.update(dt, self.player_x, self.player_y, self.time_scale)
            if not z.dead and dist((z.x, z.y), (self.player_x, self.player_y)) < 38:
                self.game_over = True
                self.flash.add((200, 0, 0, 150), 0.4)
                return

        # remove dead zombies after brief delay
        self.zombies = [z for z in self.zombies if not (z.dead and z.death_timer > 0.3)]

        # ── wave progression ─────────────────────
        alive = [z for z in self.zombies if not z.dead]
        if len(alive) == 0:
            self.wave_clear_timer += dt
            if self.wave_clear_timer > 2.0:
                self.wave += 1
                self.wave_clear_timer = 0
                self._spawn_wave(self.wave)
                self.flash.add((80, 255, 80, 60), 0.3)
        else:
            self.wave_clear_timer = 0

        # ── particles / effects / nums ───────────
        for p in self.particles:
            p.update(dt * self.time_scale)
        self.particles = [p for p in self.particles if p.alive()]

        for e in self.effects:
            e.update(dt)
        self.effects = [e for e in self.effects if e.alive()]

        for n in self.dmg_nums:
            n.update(dt)
        self.dmg_nums = [n for n in self.dmg_nums if n.alive()]

        self.flash.update(dt)

        if self.streak_surf:
            self.streak_age += dt
            if self.streak_age > 1.8:
                self.streak_surf = None

    # ── DRAW ─────────────────────────────────────

    def draw(self):
        self.screen.fill(C_BG)
        self._draw_grid()

        # draw effects under zombies
        for e in self.effects:
            e.draw(self.screen)

        # zombies
        for z in self.zombies:
            z.draw(self.screen)

        # player
        self._draw_player()

        # particles
        for p in self.particles:
            p.draw(self.screen)

        # damage numbers
        for n in self.dmg_nums:
            n.draw(self.screen, self.font_sm)

        # aim cursor
        self._draw_cursor()

        # screen flash
        self.flash.draw(self.screen)

        # HUD
        self._draw_hud()

        # combo streak
        if self.streak_surf:
            t   = self.streak_age / 1.8
            alp = int(255 * (1 - t ** 2))
            sc  = 1 + math.sin(self.streak_age * 12) * 0.04 * (1 - t)
            img = pygame.transform.rotozoom(self.streak_surf, 0, sc)
            img.set_alpha(alp)
            self.screen.blit(img, img.get_rect(center=(GAME_W // 2, GAME_H // 2 - 80)))

        # cam overlay
        cam_surf = self.tracker.get_cam_surface()
        if cam_surf:
            cx = GAME_W - CAM_W - CAM_PAD
            cy = CAM_PAD
            pygame.draw.rect(self.screen, (40, 40, 40),
                             (cx - 2, cy - 2, CAM_W + 4, CAM_H + 4))
            self.screen.blit(cam_surf, (cx, cy))
            label = self.font_xs.render("📷 LIVE", True, (120, 255, 120))
            self.screen.blit(label, (cx + 4, cy + CAM_H + 4))

        # wave clear banner
        if self.wave_clear_timer > 0:
            self._draw_wave_banner()

        if self.game_over:
            self._draw_game_over()

        # Scale fixed-res game surface to whatever the window currently is
        win_w, win_h = self.window.get_size()
        # Letterbox: maintain aspect ratio
        game_ratio = GAME_W / GAME_H
        win_ratio  = win_w / win_h
        if win_ratio >= game_ratio:
            scaled_h = win_h
            scaled_w = int(win_h * game_ratio)
        else:
            scaled_w = win_w
            scaled_h = int(win_w / game_ratio)
        ox = (win_w - scaled_w) // 2
        oy = (win_h - scaled_h) // 2
        scaled = pygame.transform.smoothscale(self.screen, (scaled_w, scaled_h))
        self.window.fill((0, 0, 0))
        self.window.blit(scaled, (ox, oy))
        pygame.display.flip()

    def _draw_grid(self):
        grid_col = (20, 28, 16)
        for x in range(0, GAME_W, 80):
            pygame.draw.line(self.screen, grid_col, (x, 0), (x, GAME_H))
        for y in range(0, GAME_H, 80):
            pygame.draw.line(self.screen, grid_col, (0, y), (GAME_W, y))

    def _draw_player(self):
        cx, cy = self.player_x, self.player_y
        # glow ring
        for r in range(28, 20, -2):
            alpha = max(0, 255 - (28 - r) * 60)
            col   = (0, int(alpha * 0.4), 0)
            pygame.draw.circle(self.screen, col, (cx, cy), r)
        pygame.draw.circle(self.screen, (80, 255, 80), (cx, cy), 18)
        pygame.draw.circle(self.screen, (200, 255, 200), (cx, cy), 10)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 5)

    def _draw_cursor(self):
        if not self.tracker.smoothed:
            return
        ax, ay = int(self.tracker.smoothed[0]), int(self.tracker.smoothed[1])
        # outer ring
        pygame.draw.circle(self.screen, (255, 255, 0), (ax, ay), 20, 2)
        # crosshair
        pygame.draw.line(self.screen, (255, 255, 0), (ax - 28, ay), (ax - 22, ay), 2)
        pygame.draw.line(self.screen, (255, 255, 0), (ax + 22, ay), (ax + 28, ay), 2)
        pygame.draw.line(self.screen, (255, 255, 0), (ax, ay - 28), (ax, ay - 22), 2)
        pygame.draw.line(self.screen, (255, 255, 0), (ax, ay + 22), (ax, ay + 28), 2)
        # speed glow
        if self.tracker.speed > 6:
            intensity = min(255, int(self.tracker.speed * 6))
            pygame.draw.circle(self.screen, (intensity, intensity // 2, 0), (ax, ay), 24, 1)

        # pinch indicator
        if self.tracker.pinching:
            pygame.draw.circle(self.screen, (100, 200, 255), (ax, ay), 32, 3)
            txt = self.font_xs.render("SPECIAL!", True, (100, 200, 255))
            self.screen.blit(txt, (ax + 36, ay - 8))

    def _draw_hud(self):
        # top-left: score
        sc  = self.font_md.render(f"SCORE  {self.score:,}", True, C_HUD)
        self.screen.blit(sc, (20, 16))

        # wave
        wv = self.font_md.render(f"WAVE  {self.wave}", True, (180, 130, 60))
        self.screen.blit(wv, (20, 56))

        # kills
        kl = self.font_sm.render(f"KILLS: {self.kills}", True, (160, 160, 160))
        self.screen.blit(kl, (20, 96))

        # combo
        if self.combo >= 2:
            scale = 1 + math.sin(time.time() * 12) * 0.06
            combo_txt = f"x{self.combo} COMBO"
            cs  = pygame.font.SysFont("Arial Black", int(36 * scale), bold=True)
            img = cs.render(combo_txt, True, C_COMBO)
            self.screen.blit(img, img.get_rect(midtop=(GAME_W // 2, 12)))

        # special cooldown bar
        cd_w = 180
        cd_h = 14
        cx   = GAME_W // 2 - cd_w // 2
        cy   = GAME_H - 44
        pct  = max(0, 1 - self.pinch_cd / PINCH_COOLDOWN)
        pygame.draw.rect(self.screen, (40, 40, 40), (cx, cy, cd_w, cd_h), border_radius=7)
        bar_col = (100, 200, 255) if pct >= 1.0 else (60, 100, 160)
        pygame.draw.rect(self.screen, bar_col,
                         (cx, cy, int(cd_w * pct), cd_h), border_radius=7)
        lbl = self.font_xs.render("PINCH SPECIAL", True, (160, 210, 255))
        self.screen.blit(lbl, lbl.get_rect(midbottom=(GAME_W // 2, cy - 2)))

        # slow-mo indicator
        if self.slow_timer > 0:
            slt = self.font_sm.render("⬛ SLOW MOTION", True, (255, 180, 0))
            self.screen.blit(slt, slt.get_rect(midbottom=(GAME_W // 2, cy - 22)))

        # gesture hint bottom-left
        hints = [
            "🖐  FLICK → attack",
            "🤏  PINCH → special",
        ]
        for i, h in enumerate(hints):
            t = self.font_xs.render(h, True, (90, 110, 80))
            self.screen.blit(t, (14, GAME_H - 44 + i * 20))

    def _draw_wave_banner(self):
        t = min(1, self.wave_clear_timer / 0.5)
        alpha = int(255 * (1 - abs(t - 0.5) * 2))
        if self.wave_clear_timer > 1.5:
            alpha = int(255 * (2.0 - self.wave_clear_timer) / 0.5)
        alpha = max(0, min(255, alpha))
        txt = self.font_lg.render(f"WAVE {self.wave - 1} CLEAR!", True, (100, 255, 100))
        txt.set_alpha(alpha)
        self.screen.blit(txt, txt.get_rect(center=(GAME_W // 2, GAME_H // 2)))

    def _draw_game_over(self):
        overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        go = self.font_lg.render("GAME OVER", True, (255, 40, 40))
        self.screen.blit(go, go.get_rect(center=(GAME_W // 2, GAME_H // 2 - 80)))
        sc = self.font_md.render(f"Score: {self.score:,}   Wave: {self.wave}   Kills: {self.kills}",
                                 True, (255, 220, 80))
        self.screen.blit(sc, sc.get_rect(center=(GAME_W // 2, GAME_H // 2)))
        rs = self.font_sm.render("Press R to restart  /  ESC to quit", True, (200, 200, 200))
        self.screen.blit(rs, rs.get_rect(center=(GAME_W // 2, GAME_H // 2 + 60)))

    # ── MAIN LOOP ────────────────────────────────

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 0.05)   # clamp to avoid spiral of death

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    # pygame 2 handles this automatically with RESIZABLE,
                    # but we re-acquire window reference just in case
                    self.window = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset()
                    # debug: manual attack with SPACE
                    elif event.key == pygame.K_SPACE:
                        if self.flick_cd <= 0:
                            self._do_slash()
                            self.flick_cd = FLICK_COOLDOWN
                    elif event.key == pygame.K_q:
                        if self.pinch_cd <= 0:
                            self._do_slash(special=True)
                            self.pinch_cd = PINCH_COOLDOWN

            self.update(dt)
            self.draw()

        self.tracker.release()
        pygame.quit()
        sys.exit()


# ─────────────────────────────────────────────
#  ENTRY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    game = ZombieFlickDefense()
    game.run()