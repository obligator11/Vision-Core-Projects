"""
╔══════════════════════════════════════════════════════════════╗
║          ShadowStrike AI — Next-Gen Boxing Game              ║
║  Real-body motion detection via OpenCV Optical Flow          ║
║  Requires: Python, OpenCV, NumPy, Pygame                     ║
║  No external model files needed — fully self-contained!      ║
╚══════════════════════════════════════════════════════════════╝

CONTROLS (keyboard fallback if no camera):
  SPACE / V  — Punch (or throw real punches at camera)
  Z / B      — Dodge (or sway body left/right)
  1 / 2 / 3  — Select difficulty on menu
  ENTER      — Confirm / advance
  Q / ESC    — Quit

MOTION CONTROLS (when camera is detected):
  • Throw fast arm movements toward camera → PUNCH
  • Sway body left or right              → DODGE
  • Duck / crouch                        → DUCK DODGE
"""

import cv2
import numpy as np
import pygame
import math
import time
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
WIN_W, WIN_H   = 1280, 720
CAM_W, CAM_H   = 640,  480
GAME_FPS       = 60
ROUND_DURATION = 90        # seconds per round
MAX_ROUNDS     = 3

# Colours
C_BG     = (8,   12,  22)
C_RED    = (220, 50,  50)
C_BLUE   = (50,  120, 220)
C_GREEN  = (50,  200, 100)
C_YELLOW = (255, 220, 50)
C_WHITE  = (255, 255, 255)
C_ORANGE = (255, 140, 30)
C_PURPLE = (180, 60,  220)
C_CYAN   = (0,   230, 230)
C_DARK   = (15,  18,  30)
C_GREY   = (100, 100, 120)
C_FLASH  = (255, 255, 200)
C_GOLD   = (255, 200, 50)

# Stamina
STAMINA_MAX        = 100.0
STAMINA_REGEN      = 14.0    # per second
STAMINA_PUNCH_COST = 10.0
STAMINA_DODGE_COST = 8.0
COMBO_WINDOW       = 0.9     # seconds between punches to extend combo

# Motion thresholds
OPTICAL_FLOW_PUNCH_THRESH = 18.0   # magnitude threshold for punch detection
REGION_MOTION_PUNCH_THRESH = 0.08  # fraction of region pixels in motion
DODGE_CENTROID_SHIFT       = 0.07  # fraction of frame width for dodge
DODGE_DUCK_SHIFT           = 0.06  # fraction of frame height for duck


# ═══════════════════════════════════════════════════════════════
#  SOUND MANAGER  (procedurally generated — zero external files)
# ═══════════════════════════════════════════════════════════════
class SoundManager:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self._sounds: Dict[str, pygame.mixer.Sound] = {}
        self._gen_all()

    # ── ADSR wave synthesis ───────────────────────────────────
    def _synth(self, freq: float, dur: float, wave: str = "sine",
               vol: float = 0.7, attack: float = 0.01, decay: float = 0.08,
               sustain: float = 0.6, release: float = 0.15,
               dist: float = 0.0, noise: float = 0.0,
               freq_sweep: float = 0.0) -> pygame.mixer.Sound:
        sr  = 44100
        n   = int(sr * dur)
        t   = np.linspace(0, dur, n, False)

        if wave == "sine":
            w = np.sin(2 * np.pi * freq * t)
        elif wave == "square":
            w = np.sign(np.sin(2 * np.pi * freq * t))
        elif wave == "saw":
            w = 2 * (t * freq - np.floor(t * freq + 0.5))
        elif wave == "tri":
            w = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
        else:  # noise
            w = np.random.uniform(-1, 1, n)

        if freq_sweep != 0:
            sweepf = freq + freq_sweep * t / dur
            w += 0.5 * np.sin(2 * np.pi * sweepf * t)
            w /= 1.5

        if noise > 0:
            w = (1 - noise) * w + noise * np.random.uniform(-1, 1, n)

        # ADSR
        env = np.ones(n)
        na, nd, nr = int(attack*sr), int(decay*sr), int(release*sr)
        ns = max(0, n - na - nd - nr)
        if na: env[:na]            = np.linspace(0, 1, na)
        if nd: env[na:na+nd]       = np.linspace(1, sustain, nd)
        env[na+nd:na+nd+ns]        = sustain
        if nr: env[na+nd+ns:]      = np.linspace(sustain, 0, min(nr, n - na - nd - ns))

        w = w * env
        if dist > 0:
            w = np.tanh(w * (1 + dist * 6)) / np.tanh(1 + dist * 6)
        w = np.clip(w * vol, -1, 1)
        stereo = np.column_stack([w, w])
        return pygame.sndarray.make_sound((stereo * 32767).astype(np.int16))

    def _gen_all(self):
        s = self._sounds
        s["punch"]  = self._synth(160, 0.14, "noise",  0.55, 0.003, 0.06, 0.15, 0.04, dist=0.3, noise=0.75)
        s["hit"]    = self._synth(75,  0.20, "sine",   0.85, 0.002, 0.14, 0.20, 0.05, dist=0.95, noise=0.45)
        s["dodge"]  = self._synth(420, 0.10, "saw",    0.30, 0.003, 0.04, 0.35, 0.05, freq_sweep=-300)
        s["block"]  = self._synth(320, 0.10, "square", 0.45, 0.001, 0.04, 0.25, 0.06, dist=0.4)
        s["combo"]  = self._synth(1100,0.16, "sine",   0.65, 0.004, 0.04, 0.55, 0.12)
        s["bell"]   = self._synth(860, 1.30, "sine",   0.90, 0.002, 0.50, 0.55, 0.55)
        s["ko"]     = self._synth(110, 0.50, "sine",   0.80, 0.002, 0.20, 0.20, 0.15, dist=0.7, noise=0.25)
        s["crowd"]  = self._synth(55,  1.00, "noise",  0.12, 0.40,  0.20, 0.80, 0.35, noise=0.98)
        s["countdown"] = self._synth(660, 0.18, "tri", 0.70, 0.005, 0.05, 0.60, 0.10)

    def play(self, name: str, vol: float = 1.0):
        s = self._sounds.get(name)
        if s:
            s.set_volume(min(1.0, vol))
            s.play()

    def loop(self, name: str, vol: float = 1.0):
        s = self._sounds.get(name)
        if s:
            s.set_volume(min(1.0, vol))
            s.play(-1)

    def stop(self, name: str):
        s = self._sounds.get(name)
        if s:
            s.stop()


# ═══════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════
@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float; max_life: float
    color: Tuple; size: float
    kind: str = "spark"   # spark | ring | streak

class ParticleSystem:
    def __init__(self):
        self.pool: List[Particle] = []

    def emit_hit(self, x, y, col=C_FLASH, n=22):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(2.5, 10.0)
            li = random.uniform(0.22, 0.52)
            self.pool.append(Particle(x, y, math.cos(a)*sp, math.sin(a)*sp,
                                      li, li, col, random.uniform(2, 6)))
        self.pool.append(Particle(x, y, 0, 0, 0.20, 0.20, col, 36, "ring"))

    def emit_combo(self, x, y, n=10):
        for _ in range(n):
            a = random.uniform(-math.pi/2 - .5, -math.pi/2 + .5)
            sp = random.uniform(3, 8)
            li = random.uniform(0.4, 0.9)
            self.pool.append(Particle(x, y, math.cos(a)*sp, math.sin(a)*sp-2,
                                      li, li, C_GOLD, random.uniform(3, 7), "spark"))

    def emit_dodge(self, x, y, dirx=1, n=12):
        for _ in range(n):
            li = random.uniform(0.18, 0.38)
            self.pool.append(Particle(x, y, dirx*random.uniform(3, 9),
                                      random.uniform(-2, 2),
                                      li, li, C_CYAN, random.uniform(2, 4)))

    def emit_streak(self, x, y, vx, vy, col, n=5):
        for _ in range(n):
            li = random.uniform(0.08, 0.18)
            self.pool.append(Particle(x+random.gauss(0,4), y+random.gauss(0,4),
                                      vx+random.gauss(0,1), vy+random.gauss(0,1),
                                      li, li, col, random.uniform(1, 3), "streak"))

    def update(self, dt):
        alive = []
        for p in self.pool:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx; p.y += p.vy
                p.vy += 0.22
                alive.append(p)
        self.pool = alive

    def draw(self, surface):
        for p in self.pool:
            a = p.life / p.max_life
            r, g, b = p.color
            c = (int(r*a), int(g*a), int(b*a))
            if p.kind == "ring":
                rad = int(p.size * (1.0 - a) * 3.5)
                if rad > 0:
                    pygame.draw.circle(surface, c, (int(p.x), int(p.y)),
                                       rad, max(1, int(3*a)))
            else:
                sz = max(1, int(p.size * a))
                pygame.draw.circle(surface, c, (int(p.x), int(p.y)), sz)


# ═══════════════════════════════════════════════════════════════
#  FLOATING TEXT
# ═══════════════════════════════════════════════════════════════
@dataclass
class FloatText:
    text: str; x: float; y: float
    color: Tuple; life: float; max_life: float; size: int = 28

class FloatTextSystem:
    def __init__(self):
        self._cache: Dict[int, pygame.font.Font] = {}
        self.items: List[FloatText] = []

    def _f(self, sz):
        if sz not in self._cache:
            self._cache[sz] = pygame.font.SysFont("Arial Black", sz, bold=True)
        return self._cache[sz]

    def add(self, text, x, y, col=C_WHITE, life=1.0, size=28):
        self.items.append(FloatText(text, x, y, col, life, life, size))

    def update(self, dt):
        alive = []
        for t in self.items:
            t.life -= dt
            if t.life > 0:
                t.y -= 42 * dt
                alive.append(t)
        self.items = alive

    def draw(self, surface):
        for t in self.items:
            a = t.life / t.max_life
            r, g, b = t.color
            col  = (int(r*a), int(g*a), int(b*a))
            surf = self._f(t.size).render(t.text, True, col)
            surface.blit(surf, (int(t.x - surf.get_width()/2), int(t.y)))


# ═══════════════════════════════════════════════════════════════
#  OPENCV MOTION TRACKER  (optical flow + background subtraction)
# ═══════════════════════════════════════════════════════════════
class MotionTracker:
    """
    Detects punches and dodges using:
    1. Lucas-Kanade optical flow on feature points
    2. Per-region motion magnitude analysis
    3. Foreground centroid tracking for dodge/duck detection
    """
    LK_PARAMS = dict(
        winSize   = (15, 15),
        maxLevel  = 2,
        criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03)
    )
    FEAT_PARAMS = dict(maxCorners=80, qualityLevel=0.25, minDistance=8, blockSize=7)

    def __init__(self, w=CAM_W, h=CAM_H, smooth=5):
        self.w, self.h    = w, h
        self._prev_gray   = None
        self._prev_pts    = None          # LK feature points
        self._bg_sub      = cv2.createBackgroundSubtractorMOG2(
                                history=40, varThreshold=45, detectShadows=False)
        self._smooth      = smooth
        # Velocity history per zone (left, right, centre)
        self._vel_hist: Dict[str, deque] = {
            "left":   deque(maxlen=smooth),
            "right":  deque(maxlen=smooth),
            "centre": deque(maxlen=smooth),
        }
        # Centroid history for dodge detection
        self._centroid_hist: deque = deque(maxlen=12)
        self._ref_centroid: Optional[Tuple] = None
        self._calib_frames  = 0
        self.calibrated     = False

        # Smoothed outputs
        self.motion_zones: Dict[str, float] = {"left": 0., "right": 0., "centre": 0.}
        self.centroid: Optional[Tuple] = None
        self.flow_magnitude: float = 0.0
        self.body_visible: bool = False
        self._no_body_frames = 0

    # ── ZONE BOUNDARIES ────────────────────────────────────────
    def _zones(self):
        """Return ROI slices: left fist, right fist, centre torso."""
        w, h = self.w, self.h
        return {
            "right":  (slice(0,    h//2),  slice(0,    w//3)),    # right fist (camera-mirrored)
            "left":   (slice(0,    h//2),  slice(2*w//3, w)),     # left fist
            "centre": (slice(h//4, 3*h//4), slice(w//4, 3*w//4)),  # torso
        }

    # ── MAIN PROCESS ───────────────────────────────────────────
    def process(self, frame_bgr: np.ndarray) -> None:
        frame = cv2.resize(frame_bgr, (self.w, self.h))
        frame = cv2.flip(frame, 1)          # mirror so left/right match player
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (5, 5), 0)

        # Background subtraction for foreground mask
        fg_mask = self._bg_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                   np.ones((5,5), np.uint8))

        # Check if someone is in frame
        fg_ratio = np.sum(fg_mask > 0) / (self.w * self.h)
        self.body_visible = fg_ratio > 0.04

        if self.body_visible:
            self._no_body_frames = 0
            self._update_centroid(fg_mask)
        else:
            self._no_body_frames += 1

        # Optical flow for velocity
        if self._prev_gray is not None:
            self._update_optical_flow(gray, fg_mask)
        else:
            # First frame: seed feature points
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.FEAT_PARAMS)

        self._prev_gray = gray.copy()

        # Zone motion from background subtraction
        zones = self._zones()
        for name, (rs, cs) in zones.items():
            region = fg_mask[rs, cs]
            ratio  = np.sum(region > 0) / max(1, region.size)
            hist   = self._vel_hist[name]
            hist.append(ratio)
            self.motion_zones[name] = float(np.mean(hist)) if hist else 0.

        # Calibrate reference centroid (1.5s after first body seen)
        if self.body_visible and not self.calibrated:
            self._calib_frames += 1
            if self._calib_frames > 45:
                self.calibrated    = True
                self._ref_centroid = self.centroid
        elif self.calibrated and self._ref_centroid and self.centroid:
            # Slowly drift reference (prevents position lock)
            rx, ry = self._ref_centroid
            cx, cy = self.centroid
            self._ref_centroid = (rx * 0.992 + cx * 0.008,
                                   ry * 0.992 + cy * 0.008)

    def _update_optical_flow(self, gray, fg_mask):
        if self._prev_pts is None or len(self._prev_pts) < 5:
            self._prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray, mask=None, **self.FEAT_PARAMS)
        if self._prev_pts is None:
            self.flow_magnitude = 0.
            return
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self.LK_PARAMS)
        if pts1 is None or st is None:
            self.flow_magnitude = 0.
            return
        good_new = pts1[st == 1]
        good_old = self._prev_pts[st == 1]
        if len(good_new) == 0:
            self.flow_magnitude = 0.
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.FEAT_PARAMS)
            return
        flow   = good_new - good_old
        magnitudes = np.linalg.norm(flow, axis=1)
        self.flow_magnitude = float(np.percentile(magnitudes, 85))
        # Refresh points occasionally
        if len(good_new) < 15:
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.FEAT_PARAMS)
        else:
            self._prev_pts = good_new.reshape(-1, 1, 2)

    def _update_centroid(self, fg_mask):
        m = cv2.moments(fg_mask)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            self._centroid_hist.append((cx / self.w, cy / self.h))
            if self._centroid_hist:
                xs = [p[0] for p in self._centroid_hist]
                ys = [p[1] for p in self._centroid_hist]
                self.centroid = (sum(xs)/len(xs), sum(ys)/len(ys))

    # ── HIGH-LEVEL QUERIES ──────────────────────────────────────
    def punch_velocity(self) -> Tuple[float, float, float]:
        """Returns (left_zone_motion, right_zone_motion, flow_mag)"""
        return (self.motion_zones["left"],
                self.motion_zones["right"],
                self.flow_magnitude)

    def dodge_direction(self) -> Tuple[str, float]:
        """Returns (direction, confidence) — '' if no dodge."""
        if not self.calibrated or self._ref_centroid is None or self.centroid is None:
            return "", 0.
        rx, ry = self._ref_centroid
        cx, cy = self.centroid
        dx = cx - rx
        dy = cy - ry
        if abs(dx) > DODGE_CENTROID_SHIFT:
            return ("right" if dx > 0 else "left"), abs(dx)
        if dy > DODGE_DUCK_SHIFT:
            return "duck", dy
        return "", 0.

    # ── DEBUG OVERLAY ──────────────────────────────────────────
    def debug_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame_bgr, (self.w, self.h))
        frame = cv2.flip(frame, 1)
        zones = self._zones()
        cols  = {"left": (0,200,0), "right": (0,100,255), "centre": (255,200,0)}
        for name, (rs, cs) in zones.items():
            y1, y2 = rs.start, rs.stop
            x1, x2 = cs.start, cs.stop
            m = self.motion_zones[name]
            alpha = min(1., m * 8)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1,y1), (x2,y2), cols[name], -1)
            frame = cv2.addWeighted(overlay, alpha * 0.35, frame, 1 - alpha*0.35, 0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), cols[name], 2)
            cv2.putText(frame, f"{name[:1].upper()}:{m:.2f}",
                        (x1+4, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cols[name], 1)
        if self.centroid:
            cx = int(self.centroid[0] * self.w)
            cy = int(self.centroid[1] * self.h)
            cv2.circle(frame, (cx, cy), 10, (255, 80, 80), 2)
            cv2.putText(frame, f"FM:{self.flow_magnitude:.1f}",
                        (5, self.h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        return frame


# ═══════════════════════════════════════════════════════════════
#  PUNCH DETECTOR
# ═══════════════════════════════════════════════════════════════
class PunchDetector:
    def __init__(self, tracker: MotionTracker, sound: SoundManager):
        self.tracker     = tracker
        self.sound       = sound
        self._cd_left    = 0.0
        self._cd_right   = 0.0
        self._combo_ts: List[float] = []
        self.combo_count = 0

    def update(self, dt: float, now: float, stamina: float
               ) -> Tuple[bool, bool, int]:
        self._cd_left  = max(0., self._cd_left  - dt)
        self._cd_right = max(0., self._cd_right - dt)

        left_m, right_m, flow = self.tracker.punch_velocity()
        lp = rp = False

        def _try_punch(motion, flow_m, is_left):
            nonlocal lp, rp
            cond = (motion > REGION_MOTION_PUNCH_THRESH or
                    flow_m > OPTICAL_FLOW_PUNCH_THRESH)
            if cond and stamina >= STAMINA_PUNCH_COST:
                self.sound.play("punch", 0.65)
                if is_left:
                    lp = True
                    self._cd_left  = 0.36
                else:
                    rp = True
                    self._cd_right = 0.36
                self._combo_ts = [t for t in self._combo_ts if now - t < COMBO_WINDOW]
                self._combo_ts.append(now)
                self.combo_count = len(self._combo_ts)

        if self._cd_left  == 0.: _try_punch(left_m,  flow, True)
        if self._cd_right == 0.: _try_punch(right_m, flow, False)
        return lp, rp, self.combo_count


# ═══════════════════════════════════════════════════════════════
#  DODGE DETECTOR
# ═══════════════════════════════════════════════════════════════
class DodgeDetector:
    def __init__(self, tracker: MotionTracker, sound: SoundManager):
        self.tracker  = tracker
        self.sound    = sound
        self._cooldown = 0.0

    def update(self, dt: float, stamina: float) -> Tuple[bool, str]:
        self._cooldown = max(0., self._cooldown - dt)
        if self._cooldown > 0 or stamina < STAMINA_DODGE_COST:
            return False, ""
        direction, conf = self.tracker.dodge_direction()
        if direction:
            self._cooldown = 0.65
            self.sound.play("dodge", 0.50)
            return True, direction
        return False, ""


# ═══════════════════════════════════════════════════════════════
#  AI OPPONENT
# ═══════════════════════════════════════════════════════════════
@dataclass
class AttackPattern:
    name: str; telegraph: float; damage: int
    stamina_cost: float; dodge_window: float

class AIOpponent:
    PATTERNS = [
        AttackPattern("jab",        0.42, 8,   5,  0.28),
        AttackPattern("cross",      0.55, 13,  8,  0.34),
        AttackPattern("hook",       0.68, 17, 12,  0.44),
        AttackPattern("uppercut",   0.78, 21, 15,  0.50),
        AttackPattern("body_shot",  0.60, 11,  8,  0.34),
        AttackPattern("combo_jab",  0.32,  7,  5,  0.20),
        AttackPattern("overhand",   0.85, 24, 18,  0.55),
    ]

    def __init__(self, difficulty: int = 1):
        self.hp         = 100
        self.max_hp     = 100
        self.stamina    = STAMINA_MAX
        self.difficulty = difficulty
        self.state      = "idle"    # idle|telegraph|attacking|stunned|blocking|ko
        self._st_timer  = 0.0      # state timer
        self._idle_t    = 0.0
        self._cur_atk: Optional[AttackPattern] = None
        self._pending   = False
        self._dodge_t   = 0.0
        self._dodge_dir = 0

        # Animation
        self._bob   = 0.0
        self._anim  = 0.0
        self.hit_f  = 0.0   # hit flash
        self.blk_f  = 0.0   # block flash
        self.x      = WIN_W * 0.68
        self.y      = WIN_H * 0.40

        # AI tuning
        self._aggression = 0.30 + 0.14 * difficulty
        self._react      = max(0.06, 0.35 - difficulty * 0.07)

    def set_round(self, rnd: int):
        self._aggression = min(0.95, 0.30 + 0.14*self.difficulty + 0.07*rnd)
        self._react      = max(0.05, 0.35 - self.difficulty*0.07 - rnd*0.02)

    def update(self, dt: float, player_punching: bool,
               player_dodged: bool) -> Tuple[bool, int, bool]:
        """Returns (hit_player, damage, was_blocked)."""
        self.stamina = min(STAMINA_MAX, self.stamina + STAMINA_REGEN * dt)
        self._bob  += dt * 2.9
        self._anim += dt * 3.6
        self.hit_f  = max(0., self.hit_f - dt * 5)
        self.blk_f  = max(0., self.blk_f - dt * 5)
        self._dodge_t = max(0., self._dodge_t - dt)

        if self.state == "ko":
            return False, 0, False

        if self.state == "stunned":
            self._st_timer -= dt
            if self._st_timer <= 0: self.state = "idle"
            return False, 0, False

        if self.state == "blocking":
            self._st_timer -= dt
            if self._st_timer <= 0: self.state = "idle"
            return False, 0, False

        # React to player punch with dodge
        if player_punching and self.state == "idle" and self._dodge_t == 0.:
            if random.random() < self._react:
                self._dodge_dir = random.choice([-1, 1])
                self._dodge_t   = 0.38

        # Telegraph → attack
        if self.state == "telegraph":
            self._st_timer -= dt
            if self._st_timer <= 0:
                self.state     = "attacking"
                self._st_timer = 0.22
                self._pending  = True
            return False, 0, False

        if self.state == "attacking":
            self._st_timer -= dt
            if self._pending and self._st_timer <= 0:
                self._pending = False
                self.state    = "idle"
                return True, self._cur_atk.damage if self._cur_atk else 10, False

        elif self.state == "idle":
            self._idle_t -= dt
            if self._idle_t <= 0:
                r = random.random()
                if r < self._aggression and self.stamina > 12:
                    pat           = random.choice(self.PATTERNS)
                    self._cur_atk = pat
                    self.state    = "telegraph"
                    mult          = 1 + (self.difficulty - 1) * 0.18
                    self._st_timer = pat.telegraph / mult
                    self.stamina  -= pat.stamina_cost
                    self._idle_t  = random.uniform(0.18, 0.75) / self.difficulty
                elif r < self._aggression + 0.18:
                    self.state    = "blocking"
                    self._st_timer = random.uniform(0.28, 0.70)
                    self._idle_t  = random.uniform(0.4, 1.4)
                else:
                    self._idle_t = random.uniform(0.25, 1.1) / self.difficulty

        return False, 0, False

    def receive_hit(self, dmg: int) -> bool:
        """Returns True if blocked."""
        if self.state == "blocking":
            block_chance = 0.50 + self.difficulty * 0.10
            if random.random() < block_chance:
                self.blk_f = 1.0
                self.hp    = max(0, self.hp - dmg // 3)
                return True
        self.hp    = max(0, self.hp - dmg)
        self.hit_f = 1.0
        self.state = "stunned"
        self._st_timer = 0.20
        if self.hp <= 0:
            self.state = "ko"
        return False

    # ── SKELETON RENDERER ──────────────────────────────────────
    def draw(self, surface: pygame.Surface):
        if self.state == "ko":
            self._draw_ko(surface); return
        x  = int(self.x)
        y  = int(self.y) + int(math.sin(self._bob) * 7)

        # Flash colour
        if self.hit_f > 0:
            v  = int(self.hit_f * 210)
            bc = (255, 255-v, 255-v)
        elif self.blk_f > 0:
            v  = int(self.blk_f * 180)
            bc = (255-v//2, 255-v//2, 255)
        else:
            bc = C_WHITE

        sw   = math.sin(self._anim) * 20
        HEAD = 28; TORSO = 92; ARM = 76; LEG = 88

        # Glow backdrop (telegraph)
        if self.state == "telegraph":
            gr  = int(45 + 25 * abs(math.sin(self._anim * 4)))
            gsurf = pygame.Surface((gr*2, gr*2), pygame.SRCALPHA)
            pygame.draw.circle(gsurf, (255, 80, 0, 90), (gr, gr), gr)
            surface.blit(gsurf, (x - gr, y - TORSO//2 - gr))

        def P(dx, dy): return (x+dx, y+dy)

        # Head
        pygame.draw.circle(surface, bc, P(0, -TORSO-HEAD), HEAD, 3)
        pygame.draw.circle(surface, C_RED, P(8, -TORSO-HEAD-3), 5)   # eye

        # Torso
        pygame.draw.line(surface, bc, P(0, -TORSO), P(0, 0), 4)

        # Shoulders
        pygame.draw.line(surface, bc, P(-42, -TORSO+6), P(42, -TORSO+6), 3)

        # Arm reach for attack
        reach = 0
        if self.state == "attacking":
            reach = int(40 * (1 - self._st_timer / 0.22))

        lel = P(int(-42+sw*.55), int(-TORSO+6+ARM*.48))
        rel = P(int(42+reach),   int(-TORSO+6+ARM*.48 - abs(reach)*.4))
        lwr = P(int(-42-sw),     int(-TORSO+6+ARM + sw*.3))
        rwr = P(int(42+reach+12), int(-TORSO+6+ARM - abs(reach)*.6))

        pygame.draw.line(surface, bc, P(-42, -TORSO+6), lel, 3)
        pygame.draw.line(surface, bc, lel, lwr, 3)
        pygame.draw.line(surface, bc, P(42, -TORSO+6), rel, 3)
        pygame.draw.line(surface, bc, rel, rwr, 3)

        pygame.draw.circle(surface, C_RED, lwr, 10)
        pygame.draw.circle(surface, C_RED, rwr, 10)

        # Hips
        pygame.draw.line(surface, bc, P(-26, 0), P(26, 0), 3)

        # Legs
        ls = math.sin(self._bob * 1.2) * 15
        lk = P(int(-26+ls*.4), int(LEG*.55))
        rk = P(int(26-ls*.4),  int(LEG*.55))
        lf = P(int(-28+ls),    LEG)
        rf = P(int(28-ls),     LEG)
        pygame.draw.line(surface, bc, P(-26,0), lk, 3)
        pygame.draw.line(surface, bc, lk, lf, 3)
        pygame.draw.line(surface, bc, P(26,0),  rk, 3)
        pygame.draw.line(surface, bc, rk, rf, 3)

        # State indicators
        labels = {"telegraph": ("!", C_ORANGE, 34),
                  "blocking":  ("B", C_BLUE,   26),
                  "stunned":   ("*", C_YELLOW, 26)}
        if self.state in labels:
            t, col, sz = labels[self.state]
            fnt = pygame.font.SysFont("Arial Black", sz, bold=True)
            lbl = fnt.render(t, True, col)
            surface.blit(lbl, (x - lbl.get_width()//2, y - TORSO - HEAD*2 - 30))

        # Dodge lean
        if self._dodge_t > 0:
            lean_x = int(self._dodge_dir * 18 * (self._dodge_t / 0.38))
            self.x = WIN_W*0.68 + lean_x

    def _draw_ko(self, surface):
        x, y = int(self.x), int(self.y)
        c    = (160, 50, 50)
        # Lying body
        pygame.draw.line(surface, c, (x-120, y+18), (x+90, y+18), 5)
        pygame.draw.circle(surface, c, (x+100, y+8), 24, 3)
        pygame.draw.line(surface, c, (x-80, y+10), (x-80, y+55), 3)
        pygame.draw.line(surface, c, (x-40, y+10), (x-40, y+55), 3)
        # Stars above head
        for i in range(3):
            a   = self._bob + i * math.tau/3
            sx  = int(x+100 + math.cos(a)*22)
            sy  = int(y - 20 + math.sin(a)*12)
            pygame.draw.circle(surface, C_GOLD, (sx, sy), 4)

    def get_hitbox(self) -> pygame.Rect:
        return pygame.Rect(int(self.x)-62, int(self.y)-238, 124, 248)


# ═══════════════════════════════════════════════════════════════
#  PLAYER
# ═══════════════════════════════════════════════════════════════
class Player:
    def __init__(self):
        self.hp         = 100
        self.max_hp     = 100
        self.stamina    = STAMINA_MAX
        self.is_dodging = False
        self.dodge_dir  = ""
        self._dodge_t   = 0.
        self.hit_f      = 0.
        self.last_combo = 0

    def update(self, dt: float):
        self.stamina  = min(STAMINA_MAX, self.stamina + STAMINA_REGEN * dt)
        self.hit_f    = max(0., self.hit_f - dt * 3)
        if self._dodge_t > 0:
            self._dodge_t -= dt
            self.is_dodging = self._dodge_t > 0
        else:
            self.is_dodging = False

    def do_punch(self) -> bool:
        if self.stamina >= STAMINA_PUNCH_COST:
            self.stamina -= STAMINA_PUNCH_COST
            return True
        return False

    def do_dodge(self, direction: str):
        if self.stamina >= STAMINA_DODGE_COST:
            self.stamina  -= STAMINA_DODGE_COST
            self.is_dodging = True
            self.dodge_dir  = direction
            self._dodge_t   = 0.48

    def receive_hit(self, dmg: int):
        if self.is_dodging:
            return
        self.hp    = max(0, self.hp - dmg)
        self.hit_f = 1.


# ═══════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════
class HUD:
    def __init__(self):
        self._fb  = pygame.font.SysFont("Arial Black", 52, bold=True)
        self._fm  = pygame.font.SysFont("Arial Black", 30, bold=True)
        self._fs  = pygame.font.SysFont("Arial",       20)
        self._ft  = pygame.font.SysFont("Arial",       15)
        self._fco = pygame.font.SysFont("Arial Black", 56, bold=True)

    @staticmethod
    def _bar(surf, x, y, w, h, val, mx, fg, bg=(35,35,55), bdr=C_GREY):
        pygame.draw.rect(surf, bg,  (x, y, w, h), border_radius=7)
        fw = int(w * max(0., val) / mx)
        if fw > 0:
            pygame.draw.rect(surf, fg, (x, y, fw, h), border_radius=7)
        pygame.draw.rect(surf, bdr, (x, y, w, h), 2, border_radius=7)

    def draw(self, surf, player: Player, ai: AIOpponent,
             rnd: int, t_left: float, combo: int):
        BW = 340
        # ── Player HP
        self._bar(surf, 28, 26, BW, 24, player.hp, player.max_hp, C_GREEN)
        lbl = self._fs.render(f"YOU  {player.hp}", True, C_WHITE)
        surf.blit(lbl, (34, 30))
        # ── Player Stamina
        self._bar(surf, 28, 56, BW, 13, player.stamina, STAMINA_MAX, C_YELLOW)
        surf.blit(self._ft.render("STA", True, C_YELLOW), (34, 57))
        # ── AI HP
        self._bar(surf, WIN_W-28-BW, 26, BW, 24, ai.hp, ai.max_hp, C_RED)
        albl = self._fs.render(f"SHADOW  {ai.hp}", True, C_WHITE)
        surf.blit(albl, (WIN_W-28-BW+4, 30))
        # ── AI Stamina
        self._bar(surf, WIN_W-28-BW, 56, BW, 13, ai.stamina, STAMINA_MAX, C_ORANGE)
        # ── Timer
        rl = self._fm.render(f"RND {rnd}", True, C_CYAN)
        surf.blit(rl, (WIN_W//2 - rl.get_width()//2, 18))
        m, s  = divmod(int(t_left), 60)
        tcol  = C_RED if t_left < 10 else C_WHITE
        tl    = self._fb.render(f"{m}:{s:02d}", True, tcol)
        surf.blit(tl, (WIN_W//2 - tl.get_width()//2, 46))
        # ── Combo
        if combo >= 2:
            cols = [C_YELLOW, C_ORANGE, C_RED, C_PURPLE]
            cc   = cols[min(combo-2, 3)]
            cl   = self._fco.render(f"{combo}x COMBO!", True, cc)
            surf.blit(cl, (WIN_W//2 - cl.get_width()//2, 112))
        # ── Difficulty badge
        dn = {1:"ROOKIE", 2:"FIGHTER", 3:"CHAMPION"}
        dc = {1:C_GREEN,  2:C_YELLOW,  3:C_RED}
        dl = self._ft.render(dn.get(ai.difficulty,""), True, dc.get(ai.difficulty, C_WHITE))
        surf.blit(dl, (WIN_W-28-dl.get_width(), 72))
        # ── Dodge indicator
        if player.is_dodging:
            ddir = player.dodge_dir.upper()
            ds   = self._fm.render(f"◄ DODGE {ddir} ►", True, C_CYAN)
            surf.blit(ds, (WIN_W//2 - ds.get_width()//2, WIN_H - 82))
        # ── Hit flash overlay
        if player.hit_f > 0:
            ov = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            ov.fill((220, 28, 28, int(player.hit_f * 95)))
            surf.blit(ov, (0, 0))


# ═══════════════════════════════════════════════════════════════
#  SLOW-MO REPLAY
# ═══════════════════════════════════════════════════════════════
class ReplaySystem:
    def __init__(self, cap=90):
        self._buf: deque = deque(maxlen=cap)
        self.playing = False
        self._frames: List = []
        self._idx    = 0
        self._fnt    = None

    def record(self, surf: pygame.Surface):
        t = pygame.transform.scale(surf, (WIN_W//4, WIN_H//4))
        self._buf.append(t.copy())

    def trigger(self):
        self._frames = list(self._buf)
        self._idx    = 0
        self.playing = True

    def update_draw(self, surf: pygame.Surface) -> bool:
        if not self.playing: return False
        if self._idx >= len(self._frames):
            self.playing = False; return False
        frame = pygame.transform.scale(self._frames[self._idx], (WIN_W, WIN_H))
        surf.blit(frame, (0, 0))
        tint = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        tint.fill((0, 0, 60, 55))
        surf.blit(tint, (0, 0))
        if self._fnt is None:
            self._fnt = pygame.font.SysFont("Arial Black", 44, bold=True)
        lbl = self._fnt.render("K  O  !    REPLAY", True, C_GOLD)
        surf.blit(lbl, (WIN_W//2 - lbl.get_width()//2, WIN_H//2 - 30))
        self._idx += 1
        return True


# ═══════════════════════════════════════════════════════════════
#  ARENA
# ═══════════════════════════════════════════════════════════════
class Arena:
    def __init__(self):
        self._bg = self._build()

    def _build(self) -> pygame.Surface:
        s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        # Floor oval
        pygame.draw.ellipse(s, (25, 18, 44), (WIN_W//2-370, WIN_H//2-28, 740, 195))
        # Ring ropes (3 levels)
        for i, yo in enumerate([-62, -38, -14]):
            pygame.draw.ellipse(s, (170, 35, 35),
                                (WIN_W//2-385+i*10, WIN_H//2+yo-88+i*10,
                                 770-i*20, 98), 3)
        # Corner posts
        for cx in [WIN_W//2-390, WIN_W//2+390]:
            pygame.draw.rect(s, (110, 110, 130), (cx-9, WIN_H//2-185, 18, 210))
        # Spotlights
        for cx, a in [(WIN_W//2, 55), (int(WIN_W*.33), 25), (int(WIN_W*.67), 25)]:
            gr = 340
            gs = pygame.Surface((gr*2, gr*2), pygame.SRCALPHA)
            pygame.draw.circle(gs, (255, 245, 190, a), (gr, gr), gr)
            s.blit(gs, (cx-gr, WIN_H//2-310))
        return s

    def draw(self, surf: pygame.Surface, fc: int):
        surf.fill(C_BG)
        surf.blit(self._bg, (0, 0))
        # Crowd silhouettes
        for i in range(0, WIN_W, 42):
            h  = 30 + (i % 22)
            y0 = WIN_H - 78 - h + int(math.sin(fc * 0.038 + i * 0.28) * 4)
            pygame.draw.rect(surf, (28+(i%14), 18+(i%9), 42+(i%18)), (i, y0, 38, h+78))


# ═══════════════════════════════════════════════════════════════
#  SCREEN STATES
# ═══════════════════════════════════════════════════════════════
class S:
    MENU      = "menu"
    FIGHTING  = "fighting"
    ROUND_END = "round_end"
    GAME_OVER = "game_over"


# ═══════════════════════════════════════════════════════════════
#  MAIN GAME
# ═══════════════════════════════════════════════════════════════
class ShadowStrikeAI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("ShadowStrike AI")
        self.clock  = pygame.time.Clock()

        # Subsystems
        self.sound     = SoundManager()
        self.particles = ParticleSystem()
        self.floats    = FloatTextSystem()
        self.hud       = HUD()
        self.arena     = Arena()
        self.replay    = ReplaySystem()

        # Fonts
        self._ftitle = pygame.font.SysFont("Arial Black", 76, bold=True)
        self._fbig   = pygame.font.SysFont("Arial Black", 50, bold=True)
        self._fmed   = pygame.font.SysFont("Arial Black", 30, bold=True)
        self._fsm    = pygame.font.SysFont("Arial",       20)
        self._ftiny  = pygame.font.SysFont("Arial",       15)

        # Camera / motion
        self._cap     = None
        self._cam_ok  = False
        self._cam_surf: Optional[pygame.Surface] = None
        self.tracker  = MotionTracker(CAM_W, CAM_H, smooth=5)
        self._try_camera()

        # Global state
        self.state        = S.MENU
        self.difficulty   = 1
        self.round_num    = 1
        self._p_wins      = 0
        self._ai_wins     = 0
        self._round_timer = float(ROUND_DURATION)
        self._round_msg   = ""
        self._round_msg_t = 0.
        self._result_txt  = ""
        self._slo_mo_done = False
        self._fc          = 0
        self._prev_t      = time.time()

        # Keyboard fallback state
        self._kb: Dict[str, float] = {}

        self._init_round()
        self.sound.loop("crowd", 0.07)

    # ── CAMERA ───────────────────────────────────────────────
    def _try_camera(self):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self._cap   = cap
                self._cam_ok = True
        except Exception:
            pass

    def _read_camera(self, dt: float):
        if not self._cam_ok or self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            return
        self.tracker.process(frame)
        dbg   = self.tracker.debug_frame(frame)
        rgb   = cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB)
        surf  = pygame.surfarray.make_surface(np.rot90(rgb))
        self._cam_surf = surf

    # ── ROUND INIT ────────────────────────────────────────────
    def _init_round(self):
        self.player = Player()
        self.ai     = AIOpponent(self.difficulty)
        self.ai.set_round(self.round_num)
        self._round_timer = float(ROUND_DURATION)
        self._slo_mo_done = False
        self.particles.pool.clear()
        self.floats.items.clear()
        self.tracker.calibrated    = False
        self.tracker._calib_frames = 0
        self._punch = PunchDetector(self.tracker, self.sound)
        self._dodge = DodgeDetector(self.tracker, self.sound)

    # ── KEYBOARD FALLBACK ─────────────────────────────────────
    def _kb_inputs(self, dt: float, keys, now: float
                   ) -> Tuple[bool, bool, int, bool, str]:
        lp = rp = dodged = False
        ddir = ""
        combo = getattr(self._punch, 'combo_count', 0)

        # Punch: Space or V
        if (keys[pygame.K_SPACE] or keys[pygame.K_v]):
            if self._kb.get("punch_cd", 0.) <= 0:
                self._kb["punch_cd"] = 0.36
                rp = True
                self._punch._combo_ts = [t for t in self._punch._combo_ts
                                          if now - t < COMBO_WINDOW]
                self._punch._combo_ts.append(now)
                self._punch.combo_count = len(self._punch._combo_ts)
                combo = self._punch.combo_count
                self.sound.play("punch", 0.6)
        self._kb["punch_cd"] = max(0., self._kb.get("punch_cd", 0.) - dt)

        # Dodge: Z or B
        if (keys[pygame.K_z] or keys[pygame.K_b]):
            if self._kb.get("dodge_cd", 0.) <= 0:
                self._kb["dodge_cd"] = 0.65
                dodged = True
                ddir   = "left" if keys[pygame.K_LEFT] else "right"
        self._kb["dodge_cd"] = max(0., self._kb.get("dodge_cd", 0.) - dt)

        return lp, rp, combo, dodged, ddir

    # ── FIGHT UPDATE ─────────────────────────────────────────
    def _update_fight(self, dt: float, now: float, keys):
        self._round_timer -= dt

        # Camera
        self._read_camera(dt)

        # Inputs
        if self._cam_ok and self.tracker.body_visible:
            lp, rp, combo, dodged, ddir = (
                *self._punch.update(dt, now, self.player.stamina),
                *self._dodge.update(dt, self.player.stamina)
            )
        else:
            lp, rp, combo, dodged, ddir = self._kb_inputs(dt, keys, now)

        punching = lp or rp

        # Player punch
        if punching and self.player.do_punch():
            # Check hit on AI
            dmg = self._calc_hit_dmg()
            if dmg > 0:
                blocked = self.ai.receive_hit(dmg)
                if blocked:
                    self.sound.play("block")
                    self.floats.add("BLOCKED", self.ai.x, self.ai.y - 210,
                                    C_BLUE, size=26)
                    self.particles.emit_hit(int(self.ai.x), int(self.ai.y-100),
                                            C_BLUE, 10)
                else:
                    bonus = max(1, combo-1) * 3
                    total = dmg + bonus
                    self.sound.play("hit", 0.9)
                    self.particles.emit_hit(int(self.ai.x),
                                            int(self.ai.y - 110), C_ORANGE, 24)
                    self.floats.add(f"-{total}",
                                    self.ai.x + random.randint(-30,30),
                                    self.ai.y - 230, C_RED, size=30)

        # Combo feedback
        if combo >= 2 and combo != self.player.last_combo:
            self.player.last_combo = combo
            self.sound.play("combo")
            self.particles.emit_combo(int(WIN_W*.28), int(WIN_H*.42))
            self.floats.add(f"{combo}× COMBO!", WIN_W*.28, WIN_H*.40,
                            C_GOLD, size=34)

        # Dodge
        if dodged and ddir:
            self.player.do_dodge(ddir)
            dirx = 1 if ddir == "right" else -1
            self.particles.emit_dodge(int(WIN_W*.28), int(WIN_H*.42), dirx)

        # AI update
        ai_hit, ai_dmg, _ = self.ai.update(dt, punching, self.player.is_dodging)
        if ai_hit and ai_dmg > 0:
            self.player.receive_hit(ai_dmg)
            if not self.player.is_dodging:
                self.sound.play("hit", 0.6)
                self.particles.emit_hit(int(WIN_W*.30), int(WIN_H*.40),
                                        C_RED, 16)
                self.floats.add(f"-{ai_dmg}", WIN_W*.28, WIN_H*.36,
                                C_RED, size=28)
            else:
                self.sound.play("dodge")
                self.floats.add("DODGED!", WIN_W*.28, WIN_H*.36,
                                C_CYAN, size=30)

        self.player.update(dt)
        self.particles.update(dt)
        self.floats.update(dt)

        # KO / round check
        outcome = self._check_outcome()
        if outcome and not self._slo_mo_done:
            self._slo_mo_done = True
            if self.ai.hp <= 0:
                self.replay.trigger()
                self.sound.play("ko")
                self.sound.play("bell")
        if outcome and not self.replay.playing:
            self._end_round(outcome)

    def _calc_hit_dmg(self) -> int:
        """Determine hit damage based on motion intensity."""
        _, _, flow = self.tracker.punch_velocity()
        # When using keyboard, flow is 0 — use a fixed value
        intensity = min(1., flow / 35.) if flow > 0 else 0.8
        base = int(8 + intensity * 14)
        return max(6, base) if random.random() > 0.15 else 0  # 15% miss

    def _check_outcome(self) -> Optional[str]:
        if self.player.hp <= 0: return "ai_wins"
        if self.ai.hp     <= 0: return "player_wins"
        if self._round_timer <= 0:
            if   self.player.hp > self.ai.hp: return "player_wins"
            elif self.ai.hp > self.player.hp: return "ai_wins"
            else:                             return "draw"
        return None

    def _end_round(self, outcome: str):
        msgs = {"player_wins": f"ROUND {self.round_num} — YOU WIN!",
                "ai_wins":     f"ROUND {self.round_num} — SHADOW WINS!",
                "draw":        f"ROUND {self.round_num} — DRAW!"}
        self._result_txt = msgs[outcome]
        if outcome == "player_wins": self._p_wins  += 1
        elif outcome == "ai_wins":   self._ai_wins  += 1
        self.sound.play("bell")
        self.state = S.ROUND_END

    # ── CAMERA PANEL ─────────────────────────────────────────
    def _draw_cam_panel(self):
        TW, TH = CAM_W//2, CAM_H//2
        px, py  = 18, WIN_H - TH - 18
        if self._cam_surf:
            thumb = pygame.transform.scale(self._cam_surf, (TW, TH))
            self.screen.blit(thumb, (px, py))
            # Glow border
            bdr_col = C_GREEN if self.tracker.body_visible else C_GREY
            pygame.draw.rect(self.screen, bdr_col, (px-2, py-2, TW+4, TH+4), 2)
            lbl = self._ftiny.render(
                "MOTION TRACKING" if self.tracker.body_visible else "STAND IN FRAME",
                True, bdr_col)
            self.screen.blit(lbl, (px+4, py+4))

            # Calibration hint
            if self._cam_ok and not self.tracker.calibrated:
                hint = self._fsm.render("◉ Calibrating motion baseline…", True, C_YELLOW)
                self.screen.blit(hint, (px+4, py+TH-24))

        else:
            pygame.draw.rect(self.screen, (18, 22, 38), (px, py, TW, TH))
            t = self._fsm.render("NO CAMERA — Keyboard: SPACE=Punch  Z=Dodge",
                                  True, C_GREY)
            self.screen.blit(t, (px+6, py+TH//2 - 10))

        # Motion bars (mini)
        bx, by = px + TW + 10, py
        l, r, flow = self.tracker.punch_velocity()
        for i, (val, name, col) in enumerate([(l,"L",C_GREEN),(r,"R",C_BLUE),(min(1.,flow/40.),"F",C_ORANGE)]):
            bh = int(TH * min(1., val * 6))
            pygame.draw.rect(self.screen, (30,30,50), (bx+i*20, by+TH-TH, 14, TH))
            pygame.draw.rect(self.screen, col,       (bx+i*20, by+TH-bh, 14, bh))
            t = self._ftiny.render(name, True, col)
            self.screen.blit(t, (bx+i*20+1, by+TH+2))

    # ── SCREENS ──────────────────────────────────────────────
    def _draw_menu(self):
        self.arena.draw(self.screen, self._fc)

        # Title with glow
        glow = self._ftitle.render("SHADOWSTRIKE  AI", True, C_RED)
        main = self._ftitle.render("SHADOWSTRIKE  AI", True, C_WHITE)
        tx   = WIN_W//2 - main.get_width()//2
        self.screen.blit(glow, (tx+3, 98+3))
        self.screen.blit(main, (tx,   98))
        sub = self._fmed.render("Real-Body Motion Boxing · AI Opponent", True, C_CYAN)
        self.screen.blit(sub, (WIN_W//2 - sub.get_width()//2, 188))

        # Difficulty buttons
        dl = self._fmed.render("DIFFICULTY:", True, C_WHITE)
        self.screen.blit(dl, (WIN_W//2 - 205, 295))
        dnames = {1:"ROOKIE", 2:"FIGHTER", 3:"CHAMPION"}
        dcols  = {1:C_GREEN, 2:C_YELLOW,  3:C_RED}
        for d in (1,2,3):
            bx = WIN_W//2 - 125 + (d-1)*135
            if d == self.difficulty:
                pygame.draw.rect(self.screen, dcols[d], (bx,290,124,44), border_radius=8)
                t = self._fmed.render(dnames[d], True, C_DARK)
            else:
                pygame.draw.rect(self.screen, (38,38,58), (bx,290,124,44), border_radius=8)
                pygame.draw.rect(self.screen, dcols[d],   (bx,290,124,44), 2, border_radius=8)
                t = self._fmed.render(dnames[d], True, dcols[d])
            self.screen.blit(t, (bx+62-t.get_width()//2, 300))

        # Key: 1/2/3 to change diff
        kl = self._fsm.render("Press  1 / 2 / 3  to change difficulty", True, C_GREY)
        self.screen.blit(kl, (WIN_W//2 - kl.get_width()//2, 344))

        # Instructions
        instr = [
            ("🥊", "PUNCH",   "Throw both fists fast toward the camera"),
            ("🏃", "DODGE",   "Lean body left / right or duck"),
            ("⚡", "STAMINA", "Rapid punches drain your stamina — pace yourself"),
            ("🎯", "COMBO",   "Land punches quickly for multiplied damage"),
            ("🎮", "KEYS",    "SPACE=Punch  Z=Dodge  1-3=Difficulty  ESC=Quit"),
        ]
        for i, (icon, key, desc) in enumerate(instr):
            y = 385 + i * 32
            ks = self._fmed.render(f"{icon} {key}", True, C_CYAN)
            ds = self._fsm.render(desc, True, C_GREY)
            self.screen.blit(ks, (WIN_W//2 - 300, y))
            self.screen.blit(ds, (WIN_W//2 - 100, y + 3))

        # Start pulse
        pulse = abs(math.sin(self._fc * 0.055))
        sc    = (int(45+205*pulse), int(145+55*pulse), 45)
        pygame.draw.rect(self.screen, sc, (WIN_W//2-145, 558, 290, 64), border_radius=10)
        sl = self._fbig.render("PRESS  ENTER", True, C_DARK)
        self.screen.blit(sl, (WIN_W//2 - sl.get_width()//2, 571))

        # Camera status indicator
        cam_lbl = ("📷 Camera ready" if self._cam_ok else "⚠ No camera — keyboard mode")
        cam_col = C_GREEN if self._cam_ok else C_YELLOW
        cl = self._fsm.render(cam_lbl, True, cam_col)
        self.screen.blit(cl, (WIN_W//2 - cl.get_width()//2, 638))

    def _draw_fight(self):
        self.arena.draw(self.screen, self._fc)
        self.ai.draw(self.screen)
        self._draw_cam_panel()
        self.particles.draw(self.screen)
        self.floats.draw(self.screen)
        self.hud.draw(self.screen, self.player, self.ai,
                      self.round_num, self._round_timer,
                      self._punch.combo_count)

        # Round start banner
        if self._round_msg and self._round_msg_t > 0:
            fnt = pygame.font.SysFont("Arial Black", 88, bold=True)
            lbl = fnt.render(self._round_msg, True, C_GOLD)
            self.screen.blit(lbl, (WIN_W//2 - lbl.get_width()//2, WIN_H//2 - 55))
            self._round_msg_t -= 0.022

    def _draw_round_end(self):
        self.arena.draw(self.screen, self._fc)
        col  = C_GREEN if "WIN!" in self._result_txt else (C_YELLOW if "DRAW" in self._result_txt else C_RED)
        rl   = self._ftitle.render(self._result_txt, True, col)
        self.screen.blit(rl, (WIN_W//2 - rl.get_width()//2, 175))
        sl   = self._fmed.render(f"Player {self._p_wins}  –  {self._ai_wins}  Shadow",
                                  True, C_WHITE)
        self.screen.blit(sl, (WIN_W//2 - sl.get_width()//2, 295))
        cl   = self._fsm.render("Press  ENTER  to continue", True, C_GREY)
        self.screen.blit(cl, (WIN_W//2 - cl.get_width()//2, 380))

    def _draw_game_over(self):
        self.arena.draw(self.screen, self._fc)
        won  = self._p_wins > self._ai_wins
        msg, col = ("VICTORY!", C_GREEN) if won else ("DEFEATED!", C_RED)
        tl   = self._ftitle.render(msg, True, col)
        self.screen.blit(tl, (WIN_W//2 - tl.get_width()//2, 155))
        fl   = self._fmed.render(f"Final: Player {self._p_wins} – {self._ai_wins} Shadow",
                                  True, C_WHITE)
        self.screen.blit(fl, (WIN_W//2 - fl.get_width()//2, 275))
        bl   = self._fsm.render("ENTER = Play Again    Q / ESC = Quit", True, C_GREY)
        self.screen.blit(bl, (WIN_W//2 - bl.get_width()//2, 370))

    # ── MAIN LOOP ─────────────────────────────────────────────
    def run(self):
        running = True
        while running:
            now = time.time()
            dt  = min(now - self._prev_t, 0.05)
            self._prev_t = now
            self._fc    += 1
            keys = pygame.key.get_pressed()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

                if ev.type == pygame.KEYDOWN:
                    k = ev.key
                    if k in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    elif self.state == S.MENU:
                        if k == pygame.K_1: self.difficulty = 1
                        if k == pygame.K_2: self.difficulty = 2
                        if k == pygame.K_3: self.difficulty = 3
                        if k == pygame.K_RETURN:
                            self.round_num  = 1
                            self._p_wins    = 0
                            self._ai_wins   = 0
                            self._init_round()
                            self.state       = S.FIGHTING
                            self._round_msg  = f"ROUND  {self.round_num}"
                            self._round_msg_t = 3.0
                            self.sound.play("bell")

                    elif self.state == S.ROUND_END:
                        if k == pygame.K_RETURN:
                            total_rounds = self.round_num >= MAX_ROUNDS
                            decisive     = self._p_wins >= 2 or self._ai_wins >= 2
                            if total_rounds or decisive:
                                self.state = S.GAME_OVER
                            else:
                                self.round_num += 1
                                self._init_round()
                                self.state       = S.FIGHTING
                                self._round_msg  = f"ROUND  {self.round_num}"
                                self._round_msg_t = 3.0
                                self.sound.play("bell")

                    elif self.state == S.GAME_OVER:
                        if k == pygame.K_RETURN:
                            self.state    = S.MENU
                            self.round_num = 1
                            self._p_wins  = 0
                            self._ai_wins = 0

            # ── UPDATE ────────────────────────────────────────
            if self.state == S.FIGHTING and not self.replay.playing:
                self._update_fight(dt, now, keys)

            # ── DRAW ──────────────────────────────────────────
            if   self.state == S.MENU:      self._draw_menu()
            elif self.state == S.FIGHTING:
                self._draw_fight()
                if not self.replay.update_draw(self.screen):
                    self.replay.record(self.screen)
            elif self.state == S.ROUND_END: self._draw_round_end()
            elif self.state == S.GAME_OVER: self._draw_game_over()

            pygame.display.flip()
            self.clock.tick(GAME_FPS)

        # Cleanup
        if self._cap:
            self._cap.release()
        pygame.quit()


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ShadowStrike AI — Boxing Game")
    print("=" * 60)
    print("  Controls (keyboard fallback):")
    print("    SPACE / V  → Punch")
    print("    Z / B      → Dodge  (+ LEFT arrow = dodge left)")
    print("    1 / 2 / 3  → Difficulty")
    print("    ENTER      → Start / Continue")
    print("    Q / ESC    → Quit")
    print()
    print("  Motion controls (when camera detected):")
    print("    Fast arm movement toward camera → Punch")
    print("    Sway body left or right          → Dodge")
    print("    Duck / crouch                    → Duck")
    print("=" * 60)
    game = ShadowStrikeAI()
    game.run()