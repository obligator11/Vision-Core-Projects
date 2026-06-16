"""
PRECISION SNIPER AI
===================
Finger-tracked precision aiming game.
- Index finger  = crosshair
- Pinch gesture = shoot
- Hold pinch    = zoom (scope)

Requirements:
    pip install opencv-python mediapipe pygame numpy
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import random
import time
import sys
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────
#  RESIZABLE WINDOW — W/H are always live
# ─────────────────────────────────────────────
W, H = 1280, 720
CAM_W, CAM_H = 280, 210
FPS = 60

# ── Palette: tactical desert / iron-sight aesthetic ──────────────────
C_BG          = (18, 16, 12)        # near-black warm
C_GROUND      = (72, 58, 38)        # desert tan
C_GROUND2     = (55, 44, 28)        # darker band
C_MOUNTAIN    = (38, 32, 22)        # distant ridge
C_SKY_TOP     = (12, 18, 28)        # deep dusk blue
C_SKY_BOT     = (38, 28, 18)        # warm horizon
C_RETICLE     = (210, 240, 180)     # pale tactical green
C_RETICLE_Z   = (255, 210, 80)      # amber when scoped
C_HUD         = (200, 220, 160)     # muted NVG green
C_RED         = (200, 50,  40)
C_ORANGE      = (255, 130, 30)
C_WHITE       = (230, 228, 220)
C_YELLOW      = (255, 220, 60)
C_TRAIL       = (255, 240, 180)
C_SPARK       = (255, 160, 40)
C_SLOWMO      = (160, 200, 255)
C_DRONE_BODY  = (80, 90, 100)       # gunmetal
C_DRONE_ARM   = (60, 68, 75)
C_DRONE_EYE   = (220, 60, 40)       # red sensor
C_DRONE_CORE  = (40, 160, 220)      # blue reactor
C_TARGET_BODY = (100, 85, 70)       # sand-coloured silhouette target
C_TARGET_RING = (200, 60, 40)

PINCH_DIST    = 0.055
ZOOM_HOLD     = 0.35
SLOW_DURATION = 1.5
RELOAD_TIME   = 1.5
AMMO_MAX      = 6

# ─────────────────────────────────────────────
#  SOUND SYNTHESIS
# ─────────────────────────────────────────────
def _make_sound(sr: int, data: np.ndarray) -> pygame.mixer.Sound:
    data   = np.clip(data, -1, 1)
    stereo = np.stack([data, data], axis=-1)
    buf    = (stereo * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(buf)

def build_sounds(sr: int = 44100) -> dict:
    t1    = np.linspace(0, 0.22, int(sr * 0.22), endpoint=False)
    noise = np.random.randn(len(t1)) * np.exp(-t1 * 22)
    crack = np.sin(2 * np.pi * 120 * t1) * np.exp(-t1 * 18)
    shot  = noise * 0.65 + crack * 0.9

    t2   = np.linspace(0, 0.3, int(sr * 0.3), endpoint=False)
    ping = (np.sin(2 * np.pi * 1100 * t2) * np.exp(-t2 * 22) +
            np.sin(2 * np.pi * 1700 * t2) * np.exp(-t2 * 35)) * 0.55

    t3   = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    head = (np.sin(2 * np.pi * 650 * t3) * np.exp(-t3 * 9) +
            np.random.randn(len(t3)) * np.exp(-t3 * 8) * 0.18) * 0.85

    t4     = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False)
    click1 = np.sin(2 * np.pi * 320 * t4) * np.exp(-t4 * 55) * (t4 < 0.02)
    click2 = np.sin(2 * np.pi * 280 * t4) * np.exp(-(t4 - 0.30) * 55) * (t4 > 0.30)
    reload = (click1 + click2) * 0.9

    t5   = np.linspace(0, 0.14, int(sr * 0.14), endpoint=False)
    miss = np.random.randn(len(t5)) * np.exp(-t5 * 30) * 0.25

    return {
        "shot":   _make_sound(sr, shot),
        "hit":    _make_sound(sr, ping),
        "head":   _make_sound(sr, head),
        "reload": _make_sound(sr, reload),
        "miss":   _make_sound(sr, miss),
    }

# ─────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float; max_life: float
    color: Tuple[int,int,int]
    size: float

@dataclass
class BulletTrail:
    sx: float; sy: float
    ex: float; ey: float
    life: float = 0.28

@dataclass
class HitMarker:
    x: float; y: float
    headshot: bool
    life: float = 0.65
    score: int  = 0

# Target types: "drone" or "panel" (flat range target)
@dataclass
class Target:
    x: float; y: float
    vx: float; vy: float
    kind: str           # "drone" | "panel" | "runner"
    size: float         # base radius / half-width
    hp: int
    is_alive: bool = True
    flash: float   = 0.0
    angle: float   = 0.0   # rotation (drones spin)
    heat: float    = 0.0   # damage glow

    def W_bound(self): return W
    def H_bound(self): return H

    def update(self, dt: float, slow: float = 1.0):
        self.x += self.vx * dt * slow
        self.y += self.vy * dt * slow
        if self.kind == "drone":
            self.angle += dt * slow * (2.0 + abs(self.vx) * 0.015)
        margin = self.size + 8
        top    = int(H * 0.28)
        if self.x < margin or self.x > W - margin:
            self.vx *= -1
        if self.y < top or self.y > H - 80:
            self.vy *= -1
        self.x = max(margin, min(W - margin, self.x))
        self.y = max(float(top), min(H - 80.0, self.y))
        self.flash = max(0.0, self.flash - dt)
        self.heat  = max(0.0, self.heat  - dt * 0.8)

    @property
    def head_zone(self) -> Tuple[float, float, float]:
        """Returns (hx, hy, hr) — the critical hit zone."""
        if self.kind == "drone":
            return self.x, self.y, self.size * 0.38
        elif self.kind == "panel":
            return self.x, self.y - self.size * 0.55, self.size * 0.28
        else:  # runner
            return self.x, self.y - self.size * 0.9, self.size * 0.3

# ─────────────────────────────────────────────
#  GAME
# ─────────────────────────────────────────────
class PrecisionSniperAI:
    def __init__(self):
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        pygame.display.set_caption("PRECISION SNIPER AI")
        self.clock  = pygame.time.Clock()

        self._rebuild_fonts()
        self.sounds = build_sounds()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_hands  = mp.solutions.hands
        self.hands_det = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self._reset()

    def _rebuild_fonts(self):
        scale = min(W, H) / 720
        self.font_lg = pygame.font.SysFont("Courier New", max(18, int(46 * scale)), bold=True)
        self.font_md = pygame.font.SysFont("Courier New", max(14, int(26 * scale)), bold=True)
        self.font_sm = pygame.font.SysFont("Courier New", max(11, int(17 * scale)), bold=True)
        self.font_xs = pygame.font.SysFont("Courier New", max(9,  int(13 * scale)))

    # ── reset ──────────────────────────────────────────────────────────
    def _reset(self):
        global W, H
        W, H = self.screen.get_size()

        self.cx, self.cy   = W // 2, H // 2
        self.pinch_time    = 0.0
        self.pinch_active  = False
        self.zoomed        = False
        self.shot_cooldown = 0.0
        self.just_shot     = False

        self.targets:     List[Target]      = []
        self.particles:   List[Particle]    = []
        self.trails:      List[BulletTrail] = []
        self.hit_markers: List[HitMarker]   = []

        self.score       = 0
        self.shots_fired = 0
        self.shots_hit   = 0
        self.headshots   = 0
        self.combo       = 0
        self.best_combo  = 0

        self.ammo         = AMMO_MAX
        self.reloading    = False
        self.reload_timer = 0.0

        self.slow_timer = 0.0
        self.wave       = 1
        self.wave_timer = 30.0
        self.total_time = 0.0
        self.started    = False
        self.game_over  = False

        # env parallax seed
        self.env_seed = random.randint(0, 9999)

        self._spawn_wave(self.wave)

        self.cam_frame: Optional[np.ndarray] = None
        self.cam_lock   = threading.Lock()
        self._cam_running = True
        self._cam_thread  = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

    # ── camera ─────────────────────────────────────────────────────────
    def _cam_loop(self):
        while self._cam_running:
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                with self.cam_lock:
                    self.cam_frame = frame

    # ── spawn ──────────────────────────────────────────────────────────
    def _spawn_wave(self, wave: int):
        self.targets.clear()
        kinds  = ["drone"] * max(1, wave // 2) + ["panel"] * 2 + ["runner"] * (1 + wave // 3)
        n      = min(2 + wave, 10)
        for i in range(n):
            kind = kinds[i % len(kinds)]
            spd  = random.uniform(55 + wave * 16, 105 + wave * 24)
            ang  = random.uniform(0, math.tau)
            size = {"drone": 28, "panel": 22, "runner": 18}[kind]
            self.targets.append(Target(
                x    = random.uniform(80, W - 80),
                y    = random.uniform(int(H * 0.3), H - 120),
                vx   = math.cos(ang) * spd,
                vy   = math.sin(ang) * spd * 0.5,
                kind = kind,
                size = float(size),
                hp   = 1 + wave // 3,
                angle= random.uniform(0, math.tau),
            ))

    # ── hand tracking ──────────────────────────────────────────────────
    def _process_hand(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands_det.process(rgb)
        if not res.multi_hand_landmarks:
            return None, False
        lm  = res.multi_hand_landmarks[0].landmark
        ix  = int(lm[8].x * W)
        iy  = int(lm[8].y * H)
        tx, ty = lm[4].x, lm[4].y
        fx, fy = lm[8].x, lm[8].y
        dist   = math.hypot(fx - tx, fy - ty)
        return (ix, iy), dist < PINCH_DIST

    # ── shoot ──────────────────────────────────────────────────────────
    def _shoot(self):
        if self.reloading or self.ammo <= 0 or self.shot_cooldown > 0:
            return
        self.ammo        -= 1
        self.shots_fired += 1
        self.shot_cooldown = 0.16
        cx, cy = self.cx, self.cy

        hit_any  = False
        headshot = False
        hit_t: Optional[Target] = None

        for t in self.targets:
            if not t.is_alive:
                continue
            hx, hy, hr = t.head_zone
            slop = 7 if self.zoomed else 4
            if math.hypot(cx - hx, cy - hy) <= hr + slop:
                t.hp    -= 2
                headshot = True
                hit_any  = True
                hit_t    = t
                break
            elif math.hypot(cx - t.x, cy - t.y) <= t.size + slop:
                t.hp   -= 1
                hit_any = True
                hit_t   = t
                break

        # trail from random screen edge toward crosshair (sniper feel)
        edge_x = random.choice([random.randint(-40, -10), random.randint(W+10, W+40)])
        edge_y = random.randint(0, int(H * 0.4))
        self.trails.append(BulletTrail(sx=float(edge_x), sy=float(edge_y),
                                        ex=float(cx),     ey=float(cy)))

        if hit_any and hit_t:
            self.shots_hit += 1
            self.combo     += 1
            self.best_combo = max(self.best_combo, self.combo)
            base = 300 if headshot else 100
            mul  = 1 + (self.combo - 1) * 0.3
            pts  = int(base * mul)
            self.score += pts
            hit_t.flash = 0.22
            hit_t.heat  = 1.0
            self._spawn_sparks(cx, cy, headshot, hit_t.kind)
            if headshot:
                self.headshots += 1
                self.sounds["head"].play()
                self.slow_timer = SLOW_DURATION
            else:
                self.sounds["hit"].play()
            self.hit_markers.append(HitMarker(x=float(cx), y=float(cy),
                                               headshot=headshot, score=pts))
            if hit_t.hp <= 0:
                hit_t.is_alive = False
                self._spawn_explosion(hit_t.x, hit_t.y, hit_t.kind)
        else:
            self.combo = 0
            self.sounds["miss"].play()
            # dust puff at crosshair (missed, hit terrain)
            for _ in range(6):
                ang = random.uniform(0, math.tau)
                spd = random.uniform(20, 80)
                self.particles.append(Particle(
                    x=float(cx), y=float(cy),
                    vx=math.cos(ang)*spd, vy=math.sin(ang)*spd - 20,
                    life=0.35, max_life=0.35,
                    color=(140, 120, 90), size=3.0,
                ))

        self.sounds["shot"].play()
        if self.ammo == 0:
            self._start_reload()

    def _start_reload(self):
        self.reloading    = True
        self.reload_timer = RELOAD_TIME
        self.sounds["reload"].play()

    # ── particles ──────────────────────────────────────────────────────
    def _spawn_sparks(self, x: float, y: float, headshot: bool, kind: str):
        n   = 30 if headshot else 14
        col = C_SPARK if kind == "drone" else C_RED
        for _ in range(n):
            ang  = random.uniform(0, math.tau)
            spd  = random.uniform(80, 350 if headshot else 180)
            life = random.uniform(0.2, 0.6)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(ang)*spd, vy=math.sin(ang)*spd,
                life=life, max_life=life,
                color=col, size=random.uniform(1.5, 5.0 if headshot else 3.0),
            ))

    def _spawn_explosion(self, x: float, y: float, kind: str):
        for _ in range(40):
            ang  = random.uniform(0, math.tau)
            spd  = random.uniform(40, 280)
            life = random.uniform(0.3, 0.9)
            col  = random.choice([C_SPARK, C_ORANGE, C_YELLOW, (200,200,200)])
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(ang)*spd, vy=math.sin(ang)*spd - 30,
                life=life, max_life=life,
                color=col, size=random.uniform(2, 8),
            ))

    # ── main loop ──────────────────────────────────────────────────────
    def run(self):
        global W, H
        dt = 0.0
        while True:
            dt = self.clock.tick(FPS) / 1000.0

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._quit(); return
                if ev.type == pygame.VIDEORESIZE:
                    W, H = ev.w, ev.h
                    self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
                    self._rebuild_fonts()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self._quit(); return
                    if ev.key == pygame.K_r and not self.reloading and self.ammo < AMMO_MAX:
                        self._start_reload()
                    if ev.key == pygame.K_SPACE and self.game_over:
                        stop = self._cam_running
                        self._cam_running = False
                        self._reset(); continue

            if self.game_over:
                self._draw_game_over(); pygame.display.flip(); continue

            # hand tracking
            with self.cam_lock:
                frame = self.cam_frame.copy() if self.cam_frame is not None else None

            pos, pinching = None, False
            if frame is not None:
                pos, pinching = self._process_hand(frame)

            if pos:
                self.cx, self.cy = pos
                self.started = True

            if pinching:
                self.pinch_time += dt
                if not self.pinch_active:
                    self.pinch_active = True
                    self._shoot()
                self.zoomed = self.pinch_time >= ZOOM_HOLD
            else:
                self.pinch_active = False
                self.pinch_time   = 0.0
                self.zoomed       = False

            slow = 0.25 if self.slow_timer > 0 else 1.0
            self.slow_timer = max(0.0, self.slow_timer - dt)

            if self.started:
                self.wave_timer -= dt
                self.total_time += dt
                if self.wave_timer <= 0:
                    self.wave      += 1
                    self.wave_timer = max(16.0, 30.0 - self.wave * 1.5)
                    self._spawn_wave(self.wave)

            self.shot_cooldown = max(0.0, self.shot_cooldown - dt)

            if self.reloading:
                self.reload_timer -= dt
                if self.reload_timer <= 0:
                    self.ammo      = AMMO_MAX
                    self.reloading = False

            # update targets
            alive = [t for t in self.targets if t.is_alive]
            for t in alive:
                t.update(dt, slow)
            if not alive:
                self._spawn_wave(self.wave)

            # update particles
            for p in self.particles:
                p.x   += p.vx * dt * slow
                p.y   += p.vy * dt * slow
                p.vy  += 220 * dt * slow
                p.life -= dt
            self.particles = [p for p in self.particles if p.life > 0]

            # update trails
            for tr in self.trails:
                tr.life -= dt
            self.trails = [tr for tr in self.trails if tr.life > 0]

            # update hit markers
            for hm in self.hit_markers:
                hm.life -= dt
                hm.y    -= 42 * dt
            self.hit_markers = [hm for hm in self.hit_markers if hm.life > 0]

            self._draw(frame, slow)
            pygame.display.flip()

    # ── DRAWING ────────────────────────────────────────────────────────
    def _draw(self, frame, slow):
        s = self.screen
        s.fill(C_BG)
        self._draw_environment(s)

        for t in self.targets:
            if t.is_alive:
                self._draw_target(s, t)

        self._draw_trails(s)
        self._draw_particles(s)
        self._draw_hit_markers(s)
        self._draw_crosshair(s)
        self._draw_hud(s, slow)

        if frame is not None:
            self._draw_cam(s, frame)

        if self.zoomed:
            self._draw_vignette(s)
        if slow < 1.0:
            self._draw_slowmo(s)

    # ── environment: desert canyon range ───────────────────────────────
    def _draw_environment(self, s):
        rng = random.Random(self.env_seed)

        # sky gradient
        horizon_y = int(H * 0.30)
        for y in range(horizon_y):
            t   = y / horizon_y
            r   = int(C_SKY_TOP[0] + (C_SKY_BOT[0] - C_SKY_TOP[0]) * t)
            g   = int(C_SKY_TOP[1] + (C_SKY_BOT[1] - C_SKY_TOP[1]) * t)
            b   = int(C_SKY_TOP[2] + (C_SKY_BOT[2] - C_SKY_TOP[2]) * t)
            pygame.draw.line(s, (r, g, b), (0, y), (W, y))

        # distant mountain ridge
        pts = [(0, horizon_y)]
        x   = 0
        while x <= W:
            pts.append((x, horizon_y - rng.randint(18, 80)))
            x += rng.randint(40, 120)
        pts.append((W, horizon_y)); pts.append((W, horizon_y + 2)); pts.append((0, horizon_y + 2))
        pygame.draw.polygon(s, C_MOUNTAIN, pts)

        # ground bands
        pygame.draw.rect(s, C_GROUND,  (0, horizon_y, W, H - horizon_y))
        pygame.draw.rect(s, C_GROUND2, (0, horizon_y, W, int((H - horizon_y) * 0.18)))

        # distance markers / target lanes — horizontal lines
        for i, frac in enumerate([0.38, 0.55, 0.72, 0.88]):
            y  = int(H * frac)
            lc = (max(20, C_GROUND[0] - i*8), max(15, C_GROUND[1] - i*8),
                  max(8,  C_GROUND[2] - i*8))
            pygame.draw.line(s, lc, (0, y), (W, y), 1)
            dist_m = (4 - i) * 100
            lbl    = self.font_xs.render(f"{dist_m}m", True, (100, 88, 62))
            s.blit(lbl, (6, y - 14))

        # range poles
        for px in [W // 5, W * 2 // 5, W * 3 // 5, W * 4 // 5]:
            pygame.draw.line(s, (55, 45, 30),
                             (int(px), horizon_y + 4),
                             (int(px), int(H * 0.85)), 2)

        # sun / moon low on horizon
        sun_x = int(W * 0.78)
        pygame.draw.circle(s, (160, 110, 60), (sun_x, horizon_y - 6), 18)

    # ── target drawing ─────────────────────────────────────────────────
    def _draw_target(self, s, t: Target):
        flash = t.flash > 0
        heat  = t.heat

        if t.kind == "drone":
            self._draw_drone(s, t, flash, heat)
        elif t.kind == "panel":
            self._draw_panel(s, t, flash, heat)
        else:
            self._draw_runner(s, t, flash, heat)

        # HP pips above target
        if t.hp > 1:
            for i in range(t.hp):
                px = int(t.x) - (t.hp - 1) * 5 + i * 10
                pygame.draw.circle(s, C_RED, (px, int(t.y) - int(t.size) - 12), 3)

    def _draw_drone(self, s, t: Target, flash: bool, heat: float):
        """Quad-rotor drone silhouette."""
        cx, cy = int(t.x), int(t.y)
        ang    = t.angle
        sz     = int(t.size)
        # heat glow
        if heat > 0.1:
            gc = (int(255 * heat), int(100 * heat), 20)
            gs = pygame.Surface((sz*4, sz*4), pygame.SRCALPHA)
            pygame.draw.circle(gs, (*gc, int(60 * heat)), (sz*2, sz*2), sz*2)
            s.blit(gs, (cx - sz*2, cy - sz*2))

        bc = (220, 220, 220) if flash else C_DRONE_BODY
        # four arms at 45°, 135°, 225°, 315°
        arm_len = int(sz * 1.0)
        for a in [ang + math.pi/4, ang + 3*math.pi/4,
                  ang + 5*math.pi/4, ang + 7*math.pi/4]:
            ax = cx + int(math.cos(a) * arm_len)
            ay = cy + int(math.sin(a) * arm_len)
            pygame.draw.line(s, bc, (cx, cy), (ax, ay), 3)
            # rotor disc
            pygame.draw.circle(s, C_DRONE_ARM, (ax, ay), int(sz * 0.42), 2)

        # body hexagon
        body_pts = [
            (cx + int(math.cos(ang + i * math.pi/3) * sz * 0.55),
             cy + int(math.sin(ang + i * math.pi/3) * sz * 0.55))
            for i in range(6)
        ]
        pygame.draw.polygon(s, bc, body_pts)
        pygame.draw.polygon(s, (30, 38, 44), body_pts, 2)

        # reactor core
        cc = (220, 220, 220) if flash else C_DRONE_CORE
        pygame.draw.circle(s, cc, (cx, cy), int(sz * 0.28))

        # red sensor eye
        ex = cx + int(math.cos(ang) * sz * 0.28)
        ey = cy + int(math.sin(ang) * sz * 0.28)
        pygame.draw.circle(s, C_DRONE_EYE, (ex, ey), max(3, int(sz * 0.14)))

        # critical zone indicator (faint ring)
        hx, hy, hr = t.head_zone
        pygame.draw.circle(s, (200, 60, 40, 60), (int(hx), int(hy)), int(hr), 1)

    def _draw_panel(self, s, t: Target, flash: bool, heat: float):
        """Classic IPSC-style cardboard silhouette on a pole."""
        cx, cy = int(t.x), int(t.y)
        w      = int(t.size * 1.1)
        h      = int(t.size * 2.0)
        bc     = (220, 200, 160) if flash else C_TARGET_BODY

        # pole
        pygame.draw.line(s, (55, 45, 30), (cx, cy + h // 2),
                         (cx, cy + h // 2 + int(H * 0.08)), 3)

        # torso rectangle
        pygame.draw.rect(s, bc, (cx - w//2, cy - h//4, w, int(h * 0.65)), border_radius=4)
        pygame.draw.rect(s, (40, 32, 20), (cx - w//2, cy - h//4, w, int(h * 0.65)), 2, border_radius=4)

        # shoulder wedge
        pts = [(cx - w//2 - 4, cy - h//4),
               (cx + w//2 + 4, cy - h//4),
               (cx + w//3, cy - int(h * 0.52)),
               (cx - w//3, cy - int(h * 0.52))]
        pygame.draw.polygon(s, bc, pts)

        # head oval
        hx, hy, hr = t.head_zone
        hc = (220, 200, 160) if flash else (180, 160, 120)
        pygame.draw.ellipse(s, hc,
                            (int(hx) - int(hr), int(hy) - int(hr * 1.2),
                             int(hr * 2), int(hr * 2.4)))
        pygame.draw.ellipse(s, (40, 32, 20),
                            (int(hx) - int(hr), int(hy) - int(hr * 1.2),
                             int(hr * 2), int(hr * 2.4)), 2)

        # scoring rings on torso
        rc = C_TARGET_RING if not flash else (255, 255, 255)
        for rr in [int(w * 0.28), int(w * 0.18)]:
            pygame.draw.circle(s, rc, (cx, cy), rr, 1)

        # heat damage marks
        if heat > 0.3:
            for _ in range(3):
                bx = cx + random.randint(-w//3, w//3)
                by = cy + random.randint(-h//6, h//4)
                pygame.draw.circle(s, (60, 40, 20), (bx, by), random.randint(2, 5))

    def _draw_runner(self, s, t: Target, flash: bool, heat: float):
        """Fast-moving mechanical runner target (side-view legs)."""
        cx, cy = int(t.x), int(t.y)
        sz     = int(t.size)
        bc     = (180, 170, 150) if flash else (90, 80, 68)
        run_phase = math.sin(time.time() * 8 + t.x * 0.01) * 0.4

        # legs
        for side in [-1, 1]:
            lx  = cx + side * int(sz * 0.3)
            knee_y = cy + int(sz * 0.7) + int(run_phase * side * sz * 0.5)
            foot_y = cy + int(sz * 1.4)
            pygame.draw.line(s, bc, (cx, cy + int(sz * 0.3)), (lx, knee_y), 3)
            pygame.draw.line(s, bc, (lx, knee_y), (cx + side * int(sz * 0.1), foot_y), 3)

        # torso
        pygame.draw.rect(s, bc,
                         (cx - sz//2, cy - sz//3, sz, int(sz * 0.75)), border_radius=3)

        # arms
        for side in [-1, 1]:
            ax = cx + side * int(sz * 0.7)
            ay = cy + int(run_phase * -side * sz * 0.4)
            pygame.draw.line(s, bc, (cx, cy - sz//6), (ax, ay), 2)

        # head
        hx, hy, hr = t.head_zone
        hc = (200, 190, 170) if flash else (75, 68, 55)
        pygame.draw.circle(s, hc, (int(hx), int(hy)), int(hr * 1.2))
        # visor
        pygame.draw.rect(s, C_DRONE_EYE,
                         (int(hx) - int(hr * 0.6), int(hy) - 2,
                          int(hr * 1.2), 4), border_radius=2)

    # ── crosshair ──────────────────────────────────────────────────────
    def _draw_crosshair(self, s):
        cx, cy = self.cx, self.cy
        col    = C_RETICLE_Z if self.zoomed else C_RETICLE
        gap    = 20 if self.zoomed else 13
        ln     = 32 if self.zoomed else 20
        w      = 2

        if self.zoomed:
            # full-screen thin crosshair lines
            faint = tuple(max(0, v - 130) for v in col)
            pygame.draw.line(s, faint, (0, cy), (W, cy), 1)
            pygame.draw.line(s, faint, (cx, 0), (cx, H), 1)
            # double rings
            for r, thk in [(54, 2), (86, 1)]:
                pygame.draw.circle(s, col, (cx, cy), r, thk)
            # mil-dots at 8 positions on outer ring
            for deg in range(0, 360, 45):
                rad = math.radians(deg)
                mx  = cx + int(86 * math.cos(rad))
                my  = cy + int(86 * math.sin(rad))
                pygame.draw.circle(s, col, (mx, my), 2)
            # range indicator ticks on vertical
            for i in range(1, 5):
                ty2 = cy + i * 22
                pygame.draw.line(s, col, (cx - 5, ty2), (cx + 5, ty2), 1)
                ty2 = cy - i * 22
                pygame.draw.line(s, col, (cx - 5, ty2), (cx + 5, ty2), 1)

        # cardinal lines
        pygame.draw.line(s, col, (cx - gap - ln, cy), (cx - gap, cy), w)
        pygame.draw.line(s, col, (cx + gap, cy),      (cx + gap + ln, cy), w)
        pygame.draw.line(s, col, (cx, cy - gap - ln), (cx, cy - gap), w)
        pygame.draw.line(s, col, (cx, cy + gap),      (cx, cy + gap + ln), w)
        # centre dot
        pygame.draw.circle(s, col, (cx, cy), 3 if self.zoomed else 2)

        # shot-flash
        if self.just_shot:
            pygame.draw.circle(s, C_YELLOW, (cx, cy), 10, 2)

    # ── trails ─────────────────────────────────────────────────────────
    def _draw_trails(self, s):
        for tr in self.trails:
            a   = tr.life / 0.28
            col = tuple(int(v * a) for v in C_TRAIL)
            pygame.draw.line(s, col,
                             (int(tr.sx), int(tr.sy)),
                             (int(tr.ex), int(tr.ey)), 2)
            # impact flash
            if tr.life > 0.18:
                pygame.draw.circle(s, C_YELLOW, (int(tr.ex), int(tr.ey)), 4)

    # ── particles ──────────────────────────────────────────────────────
    def _draw_particles(self, s):
        for p in self.particles:
            a   = p.life / p.max_life
            col = tuple(int(v * a) for v in p.color)
            r   = max(1, int(p.size * a))
            pygame.draw.circle(s, col, (int(p.x), int(p.y)), r)

    # ── hit markers ────────────────────────────────────────────────────
    def _draw_hit_markers(self, s):
        for hm in self.hit_markers:
            a   = hm.life / 0.65
            col = C_YELLOW if hm.headshot else C_WHITE
            col = tuple(int(v * a) for v in col)
            sz  = 15 if hm.headshot else 9
            pygame.draw.line(s, col, (int(hm.x)-sz, int(hm.y)-sz),
                             (int(hm.x)+sz, int(hm.y)+sz), 3)
            pygame.draw.line(s, col, (int(hm.x)+sz, int(hm.y)-sz),
                             (int(hm.x)-sz, int(hm.y)+sz), 3)
            lbl = self.font_sm.render(f"+{hm.score}", True, col)
            s.blit(lbl, (int(hm.x) + 18, int(hm.y) - 10))
            if hm.headshot:
                hs = self.font_sm.render("CRITICAL", True, col)
                s.blit(hs, (int(hm.x) + 18, int(hm.y) + 12))

    # ── HUD ────────────────────────────────────────────────────────────
    def _draw_hud(self, s, slow):
        # score
        sc = self.font_lg.render(f"{self.score:08d}", True, C_HUD)
        s.blit(sc, (18, 10))

        acc = (self.shots_hit / max(1, self.shots_fired)) * 100
        ac  = self.font_sm.render(f"ACC {acc:.1f}%", True, C_HUD)
        s.blit(ac, (18, 10 + sc.get_height() + 4))

        if self.combo > 1:
            t_  = time.time()
            cc  = tuple(min(255, int(v * (0.55 + 0.45 * math.sin(t_ * 9))))
                        for v in C_YELLOW)
            cm  = self.font_md.render(f"× {self.combo}  COMBO", True, cc)
            s.blit(cm, (18, 10 + sc.get_height() + ac.get_height() + 10))

        wv = self.font_sm.render(
            f"WAVE {self.wave}   {max(0, self.wave_timer):.0f}s", True, C_HUD)
        s.blit(wv, (W - wv.get_width() - 18, 10))

        hs = self.font_sm.render(f"HS {self.headshots}", True, C_RED)
        s.blit(hs, (W - hs.get_width() - 18, 10 + wv.get_height() + 4))

        # ammo pips bottom right (leave room for camera overlay)
        cam_space = CAM_W + 20
        for i in range(AMMO_MAX):
            filled = i < self.ammo
            col    = C_HUD if filled else (45, 55, 40)
            rx     = W - cam_space - 26 - (AMMO_MAX - 1 - i) * 20
            ry     = H - 38
            pygame.draw.rect(s, col, (rx, ry, 13, 26), 0 if filled else 2, 3)

        if self.reloading:
            prog  = 1.0 - self.reload_timer / RELOAD_TIME
            bw    = int(W * 0.18)
            bx    = W // 2 - bw // 2
            by    = H - 48
            pygame.draw.rect(s, (30, 45, 25), (bx, by, bw, 12), 2, 4)
            pygame.draw.rect(s, C_HUD,        (bx, by, int(bw * prog), 12), 0, 4)
            rl = self.font_sm.render("RELOADING", True, C_HUD)
            s.blit(rl, (W // 2 - rl.get_width() // 2, by - 20))

        if self.pinch_time > 0 and not self.zoomed:
            prog = min(1.0, self.pinch_time / ZOOM_HOLD)
            bw   = 90
            bx   = self.cx - bw // 2
            by   = self.cy + 55
            pygame.draw.rect(s, (35, 50, 25), (bx, by, bw, 7), 1, 3)
            pygame.draw.rect(s, C_RETICLE_Z,  (bx, by, int(bw * prog), 7), 0, 3)
            zl = self.font_xs.render("ZOOM", True, C_RETICLE_Z)
            s.blit(zl, (bx + bw // 2 - zl.get_width() // 2, by - 14))

        if not self.started:
            msg = self.font_md.render("▶  RAISE YOUR INDEX FINGER", True, C_HUD)
            if math.sin(time.time() * 2.5) > 0:
                s.blit(msg, (W // 2 - msg.get_width() // 2, H // 2 + 50))

        if self.ammo < AMMO_MAX and not self.reloading:
            rh = self.font_xs.render("[R] RELOAD", True, (90, 120, 70))
            s.blit(rh, (W // 2 - rh.get_width() // 2, H - 16))

    # ── camera overlay ─────────────────────────────────────────────────
    def _draw_cam(self, s, frame):
        small = cv2.resize(frame, (CAM_W, CAM_H))
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        cam_s = pygame.surfarray.make_surface(small.swapaxes(0, 1))
        x     = W - CAM_W - 10
        y     = H - CAM_H - 10
        # corner brackets instead of solid border
        blen = 14
        bt   = 2
        bclr = C_HUD
        s.blit(cam_s, (x, y))
        for bx, by in [(x, y), (x + CAM_W, y), (x, y + CAM_H), (x + CAM_W, y + CAM_H)]:
            dx = 1 if bx == x else -1
            dy = 1 if by == y else -1
            pygame.draw.line(s, bclr, (bx, by), (bx + dx * blen, by), bt)
            pygame.draw.line(s, bclr, (bx, by), (bx, by + dy * blen), bt)
        lbl = self.font_xs.render("CAM", True, C_HUD)
        s.blit(lbl, (x + 4, y + 3))

    # ── scope vignette ─────────────────────────────────────────────────
    def _draw_vignette(self, s):
        vig = pygame.Surface((W, H), pygame.SRCALPHA)
        cx, cy = self.cx, self.cy
        # radial dark rings from edge inward
        for i in range(55, 0, -3):
            rr = int(min(W, H) * 0.34 + i * 6)
            al = min(255, int(4.0 * i))
            pygame.draw.circle(vig, (0, 0, 0, al), (cx, cy), rr, 10)
        s.blit(vig, (0, 0))
        pygame.draw.circle(s, C_RETICLE_Z, (cx, cy), min(W, H) // 2 - 16, 3)

    # ── slow-mo label ──────────────────────────────────────────────────
    def _draw_slowmo(self, s):
        tint = pygame.Surface((W, H), pygame.SRCALPHA)
        tint.fill((160, 200, 255, 22))
        s.blit(tint, (0, 0))
        lbl = self.font_md.render("⬤  SLOW MOTION", True, C_SLOWMO)
        s.blit(lbl, (W // 2 - lbl.get_width() // 2, 56))

    # ── game over ──────────────────────────────────────────────────────
    def _draw_game_over(self):
        s = self.screen
        s.fill(C_BG)
        self._draw_environment(s)

        acc = (self.shots_hit / max(1, self.shots_fired)) * 100
        lines = [
            ("DEBRIEF", self.font_lg, C_HUD),
            ("", None, None),
            (f"SCORE          {self.score:,}",       self.font_md, C_WHITE),
            (f"ACCURACY       {acc:.1f}%",            self.font_md, C_WHITE),
            (f"CRITICAL HITS  {self.headshots}",      self.font_md, C_YELLOW),
            (f"BEST COMBO     × {self.best_combo}",   self.font_md, C_ORANGE),
            (f"WAVE REACHED   {self.wave}",            self.font_md, C_HUD),
            ("", None, None),
            ("[SPACE] PLAY AGAIN   [ESC] QUIT", self.font_sm, C_HUD),
        ]

        total_h = sum((f.get_height() + 8) if f else 20 for _, f, _ in lines)
        y = H // 2 - total_h // 2 - 30
        for text, font, color in lines:
            if font is None:
                y += 20; continue
            r = font.render(text, True, color)
            s.blit(r, (W // 2 - r.get_width() // 2, y))
            y += r.get_height() + 8

    def _quit(self):
        self._cam_running = False
        self.cap.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = PrecisionSniperAI()
    game.run()