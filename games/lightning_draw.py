import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import sys
import threading
from collections import deque

# ─── INIT ──────────────────────────────────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

BASE_W, BASE_H = 1280, 720
screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
pygame.display.set_caption("⚡ Lightning Draw Duel: Advanced Spellforce")
clock = pygame.time.Clock()

# ─── AUDIO SYNTHESIS ───────────────────────────────────────────────────────────
SAMPLE_RATE = 44100

def make_electric_sound():
    frames = int(SAMPLE_RATE * 0.4)
    t = np.linspace(0, 0.4, frames, False)
    data = np.sin(2 * np.pi * 180 * t) * 0.3
    data += np.random.uniform(-0.5, 0.5, frames) * np.exp(-3 * t / 0.4)
    data += np.sin(2 * np.pi * 60 * t) * 0.4 * np.exp(-5 * t / 0.4)
    env = np.exp(-4 * t / 0.4)
    data = (data * env * 0.7 * 32767).astype(np.int16)
    stereo = np.column_stack([data, data])
    return pygame.sndarray.make_sound(stereo)

def make_impact_sound():
    frames = int(SAMPLE_RATE * 0.3)
    t = np.linspace(0, 0.3, frames, False)
    data = np.random.uniform(-1, 1, frames) * np.exp(-8 * t / 0.3)
    data += np.sin(2 * np.pi * 80 * t) * 0.6 * np.exp(-6 * t / 0.3)
    data = (data * 0.8 * 32767).astype(np.int16)
    stereo = np.column_stack([data, data])
    return pygame.sndarray.make_sound(stereo)

def make_shield_sound():
    frames = int(SAMPLE_RATE * 0.35)
    t = np.linspace(0, 0.35, frames, False)
    data = np.sin(2 * np.pi * 520 * t) * 0.5
    data += np.sin(2 * np.pi * 780 * t) * 0.3
    env = np.exp(-3 * t / 0.35)
    data = (data * env * 32767).astype(np.int16)
    stereo = np.column_stack([data, data])
    return pygame.sndarray.make_sound(stereo)

def make_slash_sound():
    frames = int(SAMPLE_RATE * 0.2)
    t = np.linspace(0, 0.2, frames, False)
    freq = np.linspace(800, 200, frames)
    data = np.sin(2 * np.pi * freq * t / SAMPLE_RATE)
    noise = np.random.uniform(-0.3, 0.3, frames)
    env = np.exp(-6 * t / 0.2)
    data = ((data * 0.5 + noise) * env * 32767).astype(np.int16)
    stereo = np.column_stack([data, data])
    return pygame.sndarray.make_sound(stereo)

def make_combo_sound():
    frames = int(SAMPLE_RATE * 0.5)
    t = np.linspace(0, 0.5, frames, False)
    freqs = [300, 450, 600, 900]
    data = np.zeros(frames)
    for i, f in enumerate(freqs):
        seg = frames // 4
        start = i * seg
        end = min((i+1)*seg, frames)
        tt = t[start:end] - t[start]
        data[start:end] += np.sin(2 * np.pi * f * tt) * np.exp(-4 * tt / (0.5/4))
    data = (data * 0.6 * 32767).astype(np.int16)
    stereo = np.column_stack([data, data])
    return pygame.sndarray.make_sound(stereo)

SND_ELECTRIC = make_electric_sound()
SND_IMPACT   = make_impact_sound()
SND_SHIELD   = make_shield_sound()
SND_SLASH    = make_slash_sound()
SND_COMBO    = make_combo_sound()

# ─── COLOURS ───────────────────────────────────────────────────────────────────
C_BG        = (5,  5, 18)
C_LIGHTNING = (180, 230, 255)
C_SLASH     = (255, 100, 200)
C_SHIELD    = (80,  200, 255)
C_ENEMY_HP  = (220,  50,  50)
C_PLAYER_HP = (50,  220, 120)
C_GOLD      = (255, 215,   0)
C_WHITE     = (255, 255, 255)

# ─── FONTS ─────────────────────────────────────────────────────────────────────
try:
    FONT_BIG   = pygame.font.SysFont("Segoe UI", 52, bold=True)
    FONT_MED   = pygame.font.SysFont("Segoe UI", 30, bold=True)
    FONT_SM    = pygame.font.SysFont("Segoe UI", 20)
    FONT_TITLE = pygame.font.SysFont("Segoe UI", 90, bold=True)
except:
    FONT_BIG   = pygame.font.SysFont(None, 52)
    FONT_MED   = pygame.font.SysFont(None, 30)
    FONT_SM    = pygame.font.SysFont(None, 20)
    FONT_TITLE = pygame.font.SysFont(None, 90)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def draw_text_shadow(surf, text, font, color, pos, shadow=(2,2)):
    sx, sy = pos[0]+shadow[0], pos[1]+shadow[1]
    s = font.render(text, True, (0,0,0))
    surf.blit(s, (sx, sy))
    t = font.render(text, True, color)
    surf.blit(t, pos)

CACHED_GLOWS = {}
def get_cached_glow(radius, color):
    key = (radius, tuple(color[:3]))
    if key in CACHED_GLOWS:
        return CACHED_GLOWS[key]
    s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
    for r in range(radius, 0, -4):
        a = int(180 * (1 - r/radius)**1.5)
        c = (*color[:3], max(0, min(255, a)))
        pygame.draw.circle(s, c, (radius, radius), r)
    CACHED_GLOWS[key] = s
    return s

# ─── ASYNCHRONOUS THREADED VIDEO CAPTURE PIPELINE ────────────────────────────
class ThreadedVideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        self.frame = None
        self.tracking_tip = None
        self.finger_up = False
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)
            
            tip_norm = None
            index_up = False
            
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                
                tip = lm.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                mcp = lm.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                wrist = lm.landmark[self.mp_hands.HandLandmark.WRIST]
                
                tip_norm = (tip.x, tip.y)
                
                hand_scale = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y) or 1.0
                extension_distance = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
                
                index_up = (extension_distance > hand_scale * 0.6) and (tip.y < mcp.y)

            with self.lock:
                self.frame = frame
                self.tracking_tip = tip_norm
                self.finger_up = index_up

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, None, False
            return self.frame.copy(), self.tracking_tip, self.finger_up

    def stop(self):
        self.stopped = True
        self.cap.release()
        self.hands.close()

# ─── PARTICLES & VFX ───────────────────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, color, vx=None, vy=None, life=None, size=None):
        self.x, self.y = float(x), float(y)
        self.color = color
        self.vx = vx if vx is not None else random.uniform(-4, 4)
        self.vy = vy if vy is not None else random.uniform(-6, -1)
        self.life = life if life is not None else random.uniform(0.3, 0.7)
        self.max_life = self.life
        self.size = size if size is not None else random.uniform(2, 5)

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        t = self.life / self.max_life
        r = max(1, int(self.size * t))
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), r)

class LightningBolt:
    def __init__(self, x1, y1, x2, y2, branches=2, color=None):
        self.color = color or C_LIGHTNING
        self.segs = self._gen(x1, y1, x2, y2, branches)
        self.life = 0.20
        self.max_life = 0.20

    def _gen(self, x1, y1, x2, y2, depth):
        if depth == 0 or dist((x1,y1),(x2,y2)) < 15:
            return [((x1,y1),(x2,y2))]
        mx = (x1+x2)/2 + random.uniform(-20,20)
        my = (y1+y2)/2 + random.uniform(-20,20)
        segs = self._gen(x1,y1,mx,my,depth-1) + self._gen(mx,my,x2,y2,depth-1)
        if random.random() < 0.2:
            bx = mx + random.uniform(-40,40)
            by = my + random.uniform(-40,40)
            segs += self._gen(mx,my,bx,by,depth-1 if depth>1 else 0)
        return segs

    def update(self, dt):
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        w = max(1, int(3 * (self.life / self.max_life)))
        for p1, p2 in self.segs:
            pygame.draw.line(surf, self.color, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), w)

class Explosion:
    def __init__(self, x, y, color, count=25):
        self.particles = []
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(2, 9)
            self.particles.append(Particle(
                x, y, color,
                math.cos(angle)*speed,
                math.sin(angle)*speed,
                random.uniform(0.3, 0.6),
                random.uniform(2, 6)
            ))
        self.bolts = [LightningBolt(x, y, x + random.uniform(-80,80), y + random.uniform(-80,80), 1, color) for _ in range(3)]

    def update(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)]
        self.bolts = [b for b in self.bolts if b.update(dt)]
        return self.particles or self.bolts

    def draw(self, surf):
        for b in self.bolts: b.draw(surf)
        for p in self.particles: p.draw(surf)

# ─── SPELL TRAIL ───────────────────────────────────────────────────────────────
class SpellTrail:
    def __init__(self):
        self.points = deque(maxlen=40)
        self.color = C_SLASH

    def add(self, pt):
        self.points.append(pt)

    def clear(self):
        self.points.clear()

    def draw(self, surf):
        if len(self.points) < 2: return
        pts = list(self.points)
        for i in range(1, len(pts)):
            w = max(1, int(5 * (i / len(pts))))
            pygame.draw.line(surf, self.color, pts[i-1], pts[i], w)
        if pts:
            gl = get_cached_glow(14, self.color)
            surf.blit(gl, (pts[-1][0]-14, pts[-1][1]-14))

# ─── ROBUST GESTURE RECOGNIZER ─────────────────────────────────────────────────
class GestureRecognizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.points = []
        self.smoothed = []

    def add(self, pt):
        self.points.append(pt)
        if len(self.points) > 3:
            xs = [p[0] for p in self.points[-4:]]
            ys = [p[1] for p in self.points[-4:]]
            self.smoothed.append((int(np.mean(xs)), int(np.mean(ys))))
        else:
            self.smoothed.append(pt)

    def recognize(self):
        pts = self.smoothed
        if len(pts) < 6:
            return None
        return self._classify(pts)

    def _classify(self, pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        n = len(pts)

        if w < 40 and h < 40: 
            return None

        cx, cy = np.mean(xs), np.mean(ys)
        radii = [math.hypot(p[0]-cx, p[1]-cy) for p in pts]
        mean_radius = np.mean(radii)
        
        if mean_radius > 20:
            variance = np.mean([abs(r - mean_radius) for r in radii]) / mean_radius
            start_end_dist = dist(pts[0], pts[-1])
            if variance < 0.35 and (start_end_dist < mean_radius * 1.8 or n > 15):
                return 'SHIELD'

        if n >= 8:
            dirs = []
            step = max(1, n // 10)
            for i in range(step, n, step):
                dx = pts[i][0] - pts[i-step][0]
                if abs(dx) > 5:
                    dirs.append(1 if dx > 0 else -1)
            changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
            if changes >= 2 and w > h * 0.4:
                return 'LIGHTNING'

        if n >= 4:
            total_len = dist(pts[0], pts[-1])
            if total_len > 60:
                dx = pts[-1][0] - pts[0][0]
                dy = pts[-1][1] - pts[0][1]
                devs = []
                for p in pts:
                    t2 = ((p[0]-pts[0][0])*dx + (p[1]-pts[0][1])*dy) / (total_len**2 + 1e-5)
                    px = pts[0][0] + t2*dx
                    py = pts[0][1] + t2*dy
                    devs.append(dist(p, (px,py)))
                if devs and max(devs) < total_len * 0.40:
                    return 'SLASH'

        return None

# ─── ENEMY ─────────────────────────────────────────────────────────────────────
ENEMY_SPELLS = ['FIRE_BALL', 'DARK_SLASH', 'VOID_BEAM']

class Enemy:
    def __init__(self, sw, sh):
        self.max_hp = 200
        self.hp     = 200
        self.x      = sw * 0.72
        self.y      = sh * 0.38
        self.sw, self.sh = sw, sh
        self.bob_t  = 0.0
        self.anim_t = 0.0
        self.spell_timer = 3.5
        self.current_spell = None
        self.casting_t = 0.0
        self.cast_dur  = 1.8
        self.shake_t   = 0.0
        self.charge_particles = []
        self.dead = False

    def resize(self, sw, sh):
        self.x = sw * 0.72
        self.y = sh * 0.38
        self.sw, self.sh = sw, sh

    def update(self, dt):
        self.bob_t  += dt * 1.5
        self.anim_t += dt
        self.shake_t = max(0, self.shake_t - dt)
        self.spell_timer -= dt

        if self.current_spell:
            self.casting_t += dt
            if random.random() < 0.3:
                angle = random.uniform(0, math.tau)
                r = random.uniform(20, 50)
                self.charge_particles.append(Particle(self.x + math.cos(angle)*r, self.y + math.sin(angle)*r, (220, 80, 80), 0, 0, 0.3, 3))
            for p in self.charge_particles: p.update(dt)
            self.charge_particles = [p for p in self.charge_particles if p.life > 0]

        if self.spell_timer <= 0 and not self.current_spell:
            self.current_spell = random.choice(ENEMY_SPELLS)
            self.casting_t = 0.0
            self.spell_timer = random.uniform(4, 7)
            return None

        if self.current_spell and self.casting_t >= self.cast_dur:
            spell = self.current_spell
            self.current_spell = None
            self.charge_particles.clear()
            return spell
        return None

    def take_hit(self, dmg):
        self.hp = max(0, self.hp - dmg)
        self.shake_t = 0.18
        if self.hp == 0:
            self.dead = True

    def draw(self, surf):
        bob = math.sin(self.bob_t) * 8
        ox = math.sin(self.shake_t * 40) * 6 if self.shake_t > 0 else 0
        cx = int(self.x + ox)
        cy = int(self.y + bob)
        sz = 68

        shadow = pygame.Surface((sz*2, 28), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0,0,0,50), (0,0,sz*2,28))
        surf.blit(shadow, (cx-sz, cy+sz+10))

        body_pts = [(cx-42,cy+sz), (cx+42,cy+sz), (cx+28,cy-sz//2), (cx-28,cy-sz//2)]
        pygame.draw.polygon(surf, (60,0,80), body_pts)
        pygame.draw.polygon(surf, (100,0,130), body_pts, 2)

        pygame.draw.circle(surf, (40,0,55), (cx,cy-sz//2-22), 30)
        pygame.draw.circle(surf, (80,0,100), (cx,cy-sz//2-22), 30, 2)

        for ex, ey in [(cx-12, cy-sz//2-22), (cx+12, cy-sz//2-22)]:
            gl = get_cached_glow(10, (220, 50, 50))
            surf.blit(gl, (ex-10, ey-10))
            pygame.draw.circle(surf, (255,80,80), (ex,ey), 5)

        for side in [-1, 1]:
            hx = cx + side * (48 + math.cos(self.anim_t)*6)
            hy = cy + 10 + math.sin(self.anim_t + side)*8
            pygame.draw.circle(surf, (80,0,100), (int(hx), int(hy)), 12)
            if self.current_spell:
                gl = get_cached_glow(14, (200,50,50))
                surf.blit(gl, (int(hx)-14, int(hy)-14))

        if self.current_spell:
            bw = 140
            bx = cx - bw//2
            by = cy - sz - 60
            pygame.draw.rect(surf, (60,0,0), (bx-1, by-1, bw+2, 16), border_radius=6)
            fill = int(bw * (self.casting_t / self.cast_dur))
            if fill > 0:
                pygame.draw.rect(surf, (220,50,50), (bx, by, fill, 14), border_radius=6)
            txt = FONT_SM.render(self.current_spell.replace('_',' '), True, (255,150,150))
            surf.blit(txt, (bx, by-22))

        for p in self.charge_particles: p.draw(surf)

        bw = 160
        bx = cx - bw//2
        by = cy - sz - 90
        pygame.draw.rect(surf, (40,0,0), (bx-2, by-2, bw+4, 18), border_radius=8)
        hp_w = int(bw * self.hp / self.max_hp)
        if hp_w > 0:
            pygame.draw.rect(surf, C_ENEMY_HP, (bx, by, hp_w, 14), border_radius=7)
        draw_text_shadow(surf, f"DARK MAGE  {self.hp}/{self.max_hp}", FONT_SM, (255,120,120), (bx, by-20))

# ─── INCOMING ATTACK ───────────────────────────────────────────────────────────
class IncomingAttack:
    def __init__(self, spell, ex, ey, tx, ty):
        self.spell = spell
        self.x, self.y = float(ex), float(ey)
        self.tx, self.ty = float(tx), float(ty)
        dx = tx - ex; dy = ty - ey
        d = math.hypot(dx, dy) or 1
        speed = 280
        self.vx = dx/d * speed
        self.vy = dy/d * speed
        self.life = 3.0
        self.color = {
            'FIRE_BALL': (255, 120, 30),
            'DARK_SLASH': (160, 40, 255),
            'VOID_BEAM':  (0, 180, 255),
        }.get(spell, (200, 200, 200))
        self.particles = []
        self.radius = 16

    def update(self, dt, sw, sh):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        if random.random() < 0.4:
            self.particles.append(Particle(self.x, self.y, self.color, random.uniform(-1, 1), random.uniform(-1, 1), 0.25, 4))
        self.particles = [p for p in self.particles if p.update(dt)]
        if self.x < 0 or self.x > sw or self.y < 0 or self.y > sh:
            return False
        return self.life > 0

    def hit_player(self, px, py, radius=50):
        return dist((self.x, self.y), (px, py)) < radius + self.radius

    def draw(self, surf):
        for p in self.particles: p.draw(surf)
        gl = get_cached_glow(int(self.radius * 1.8), self.color)
        surf.blit(gl, (int(self.x) - int(self.radius * 1.8), int(self.y) - int(self.radius * 1.8)))
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surf, C_WHITE, (int(self.x), int(self.y)), self.radius // 2)

# ─── HUD FLOATER ───────────────────────────────────────────────────────────────
class FloatText:
    def __init__(self, x, y, text, color, size=28):
        self.x, self.y = float(x), float(y)
        self.text = text
        self.color = color
        self.life = 1.2
        self.max_life = 1.2
        self.font = pygame.font.SysFont("Segoe UI", size, bold=True)

    def update(self, dt):
        self.y -= 50 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        t = self.life / self.max_life
        a = int(255 * min(t * 2, 1))
        txt = self.font.render(self.text, True, self.color)
        txt.set_alpha(a)
        surf.blit(txt, (int(self.x) - txt.get_width()//2, int(self.y)))

# ─── STAR FIELD ────────────────────────────────────────────────────────────────
class StarField:
    def __init__(self, w, h, n=80):
        self.stars = [(random.randint(0,w), random.randint(0,h), random.uniform(0.5,2.0), random.uniform(0.3,1.0)) for _ in range(n)]
        self.t = 0

    def update(self, dt):
        self.t += dt

    def draw(self, surf):
        for x, y, size, phase in self.stars:
            a = int(120 + 60 * math.sin(self.t * 2 + phase * 10))
            c = (a, a, min(255, a+40))
            pygame.draw.circle(surf, c, (x,y), int(size))

# ─── MAIN GAME ORCHESTRATION ───────────────────────────────────────────────────
class Game:
    def __init__(self):
        self.sw, self.sh = BASE_W, BASE_H
        self.state = 'TITLE'
        self.reset_game()

        self.stream = ThreadedVideoStream(0).start()
        self.cam_w, self.cam_h = 240, 180
        self.cam_frame = None
        self.title_t = 0.0

    def reset_game(self):
        self.player_hp    = 200
        self.player_max   = 200
        self.player_x     = self.sw * 0.28
        self.player_y     = self.sh * 0.55
        self.shield_t     = 0.0
        self.shield_active = False

        self.enemy        = Enemy(self.sw, self.sh)
        self.trail        = SpellTrail()
        self.gesture      = GestureRecognizer()
        self.drawing      = False
        self.draw_cooldown = 0.0

        self.attacks      = []
        self.effects      = []
        self.float_texts  = []
        self.particles    = []
        self.combo        = 0
        self.combo_t      = 0.0
        self.last_spell   = None

        self.stars        = StarField(self.sw, self.sh)
        self.score        = 0
        self.shake_t      = 0.0
        self.flash_t      = 0.0
        self.flash_color  = (255,255,255)

        self.tracking_tip  = None
        self.finger_up     = False

    def resize(self, w, h):
        self.sw, self.sh = w, h
        self.player_x = w * 0.28
        self.player_y = h * 0.55
        self.enemy.resize(w, h)
        self.stars = StarField(w, h)

    def process_camera_inputs(self):
        frame, tip_norm, index_up = self.stream.read()
        if frame is None:
            return

        self.cam_frame = cv2.resize(frame, (self.cam_w, self.cam_h))
        self.finger_up = index_up

        if tip_norm:
            gx = int(tip_norm[0] * self.sw)
            gy = int(tip_norm[1] * self.sh)
            self.tracking_tip = (gx, gy)
        else:
            self.tracking_tip = None

    def handle_gesture_input(self):
        tip = self.tracking_tip
        if tip is None:
            if self.drawing:
                self._finish_gesture()
            return

        if self.finger_up and not self.drawing and self.draw_cooldown <= 0:
            self.drawing = True
            self.gesture.reset()
            self.trail.clear()

        if self.finger_up and self.drawing:
            self.gesture.add(tip)
            self.trail.add(tip)
            if len(self.gesture.smoothed) > 6:
                self.trail.color = C_LIGHTNING
            else:
                self.trail.color = C_SLASH

        if not self.finger_up and self.drawing:
            self._finish_gesture()

    def _finish_gesture(self):
        self.drawing = False
        spell = self.gesture.recognize()
        if spell:
            self._cast_spell(spell)
        self.gesture.reset()
        self.trail.clear()
        self.draw_cooldown = 0.25

    def _cast_spell(self, spell):
        pts = self.gesture.smoothed
        if not pts: return

        if spell == self.last_spell:
            self.combo += 1
        else:
            self.combo = 1
        self.last_spell = spell
        self.combo_t = 2.5
        combo_mult = 1 + (self.combo - 1) * 0.5

        if spell == 'SLASH':
            dmg = int(25 * combo_mult)
            SND_SLASH.play()
            self.enemy.take_hit(dmg)
            if len(pts) >= 2:
                self.effects.append(LightningBolt(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], 2, C_SLASH))
            self.effects.append(Explosion(self.enemy.x, self.enemy.y, C_SLASH, 20))
            self.float_texts.append(FloatText(self.enemy.x, self.enemy.y-40, f"SLASH -{dmg}", C_SLASH))
            self.score += dmg

        elif spell == 'LIGHTNING':
            dmg = int(45 * combo_mult)
            SND_ELECTRIC.play()
            self.enemy.take_hit(dmg)
            for i in range(0, len(pts)-1, max(1, len(pts)//6)):
                j = min(i+max(1,len(pts)//6), len(pts)-1)
                self.effects.append(LightningBolt(pts[i][0], pts[i][1], pts[j][0], pts[j][1], 1, C_LIGHTNING))
            self.effects.append(LightningBolt(pts[-1][0], pts[-1][1], int(self.enemy.x), int(self.enemy.y), 2, C_LIGHTNING))
            self.effects.append(Explosion(self.enemy.x, self.enemy.y, C_LIGHTNING, 30))
            self.float_texts.append(FloatText(self.enemy.x, self.enemy.y-40, f"⚡ LIGHTNING -{dmg}!", C_LIGHTNING, 34))
            self.flash_t = 0.06
            self.flash_color = (180, 230, 255)
            self.score += dmg

        elif spell == 'SHIELD':
            SND_SHIELD.play()
            self.shield_active = True
            self.shield_t = 4.0
            if len(pts) >= 4:
                self.effects.append(LightningBolt(pts[0][0], pts[0][1], pts[len(pts)//2][0], pts[len(pts)//2][1], 1, C_SHIELD))
                self.effects.append(LightningBolt(pts[len(pts)//2][0], pts[len(pts)//2][1], pts[-1][0], pts[-1][1], 1, C_SHIELD))
            self.float_texts.append(FloatText(self.player_x, self.player_y-60, "🛡 SHIELD ACTIVE", C_SHIELD, 26))
            self.score += 10

        if self.combo >= 3:
            SND_COMBO.play()
            self.float_texts.append(FloatText(self.sw//2, self.sh//3, f"✦ COMBO x{self.combo}! ✦", C_GOLD, 38))
            self.effects.append(Explosion(self.sw//2, self.sh//2, C_GOLD, 40))

    def update(self, dt):
        if self.state == 'TITLE':
            self.title_t += dt
            frame, _, _ = self.stream.read()
            return
        if self.state in ('GAME_OVER', 'WIN'):
            return

        self.process_camera_inputs()
        self.handle_gesture_input()

        self.stars.update(dt)
        self.draw_cooldown = max(0, self.draw_cooldown - dt)
        self.shake_t       = max(0, self.shake_t - dt)
        self.flash_t       = max(0, self.flash_t - dt)
        self.shield_t      = max(0, self.shield_t - dt)
        if self.shield_t <= 0: self.shield_active = False
        self.combo_t       = max(0, self.combo_t - dt)
        if self.combo_t <= 0: self.combo = 0

        # ADAPTIVE DIFFICULTY ENEMY AI TWEAKS
        # The enemy mages channel spell triggers 25% faster for every 150 points you bank!
        speed_modifier = min(2.2, 1.0 + (self.score / 150.0) * 0.25)
        enemy_spell = self.enemy.update(dt * speed_modifier)
        if enemy_spell:
            tx = self.player_x + random.uniform(-40, 40)
            ty = self.player_y + random.uniform(-40, 40)
            self.attacks.append(IncomingAttack(enemy_spell, self.enemy.x, self.enemy.y, tx, ty))

        alive = []
        for atk in self.attacks:
            if atk.update(dt, self.sw, self.sh):
                if atk.hit_player(self.player_x, self.player_y):
                    if self.shield_active:
                        SND_SHIELD.play()
                        self.effects.append(Explosion(int(atk.x), int(atk.y), C_SHIELD, 15))
                        self.float_texts.append(FloatText(self.player_x, self.player_y-60, "BLOCKED!", C_SHIELD))
                        self.score += 15
                    else:
                        dmg = 20
                        self.player_hp = max(0, self.player_hp - dmg)
                        SND_IMPACT.play()
                        self.effects.append(Explosion(int(atk.x), int(atk.y), atk.color, 15))
                        self.float_texts.append(FloatText(self.player_x, self.player_y-60, f"-{dmg}", C_ENEMY_HP))
                        self.shake_t = 0.15
                        self.flash_t = 0.05
                        self.flash_color = atk.color
                else:
                    alive.append(atk)
        self.attacks = alive

        self.effects      = [e for e in self.effects if e.update(dt)]
        self.float_texts  = [f for f in self.float_texts if f.update(dt)]

        if self.player_hp <= 0: self.state = 'GAME_OVER'
        if self.enemy.dead: self.state = 'WIN'

    def draw(self):
        ox = int(math.sin(self.shake_t * 35) * 7) if self.shake_t > 0 else 0
        oy = int(math.cos(self.shake_t * 28) * 5) if self.shake_t > 0 else 0
        canvas = pygame.Surface((self.sw, self.sh))
        canvas.fill(C_BG)

        if self.state == 'TITLE':
            self._draw_title(canvas)
        elif self.state in ('PLAYING', 'GAME_OVER', 'WIN'):
            self._draw_game(canvas)
            if self.state == 'GAME_OVER':
                self._draw_overlay(canvas, "DEFEATED", (220,50,50), "You were struck down...")
            elif self.state == 'WIN':
                self._draw_overlay(canvas, "VICTORY!", C_GOLD, f"Score: {self.score}")

        screen.blit(canvas, (ox, oy))

        if self.flash_t > 0:
            fl = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
            a  = int(100 * self.flash_t / 0.06)
            fl.fill((*self.flash_color[:3], min(100, a)))
            screen.blit(fl, (0, 0))

        pygame.display.flip()

    def _draw_title(self, surf):
        self.stars.draw(surf)
        pulse = 0.85 + 0.15 * math.sin(self.title_t * 3)
        title_c = (int(180*pulse), int(230*pulse), 255)
        draw_text_shadow(surf, "⚡ LIGHTNING DRAW DUEL ⚡", FONT_TITLE, title_c, (self.sw//2 - 520, self.sh//3 - 50), (4,4))

        sub_lines = [
            ("SLASH  ─  Draw a straight line", C_SLASH),
            ("LIGHTNING  ─  Draw a zigzag", C_LIGHTNING),
            ("SHIELD  ─  Draw a circle", C_SHIELD),
        ]
        for i, (line, c) in enumerate(sub_lines):
            draw_text_shadow(surf, line, FONT_MED, c, (self.sw//2 - 220, self.sh//2 + i*44))

        blink = int(255 * (0.5 + 0.5*math.sin(self.title_t*4)))
        draw_text_shadow(surf, "Press SPACE to begin", FONT_MED, (blink, blink, blink), (self.sw//2-130, self.sh*0.82))

    def _draw_game(self, surf):
        self.stars.draw(surf)
        mid_x = self.sw // 2
        pygame.draw.line(surf, (60, 0, 80), (mid_x, 0), (mid_x, self.sh), 2)

        for e in self.effects: e.draw(surf)
        for atk in self.attacks: atk.draw(surf)
        self.trail.draw(surf)
        self._draw_player(surf)
        self.enemy.draw(surf)

        if self.shield_active:
            # ── SHIELD VFX UPGRADE: pulsates and rotates ──
            pulse = 0.85 + 0.15 * math.sin(pygame.time.get_ticks() * 0.01)
            r = int(65 * pulse)
            gl = get_cached_glow(r + 15, C_SHIELD)
            surf.blit(gl, (int(self.player_x) - r - 15, int(self.player_y) - r - 15))
            
            # Draw rotating defensive runes around the player body
            num_segments = 8
            angle_offset = pygame.time.get_ticks() * 0.002
            for i in range(num_segments):
                angle = angle_offset + (i * (math.tau / num_segments))
                rx = int(self.player_x + math.cos(angle) * r)
                ry = int(self.player_y + math.sin(angle) * r)
                pygame.draw.circle(surf, C_WHITE, (rx, ry), 4)

            pygame.draw.circle(surf, C_SHIELD, (int(self.player_x), int(self.player_y)), r, 2)
            draw_text_shadow(surf, f"🛡 {self.shield_t:.1f}s", FONT_SM, C_SHIELD, (int(self.player_x)-25, int(self.player_y)+55))

        for f in self.float_texts: f.draw(surf)
        self._draw_hud(surf)
        self._draw_camera(surf)

    def _draw_player(self, surf):
        px, py = int(self.player_x), int(self.player_y)
        bob = math.sin(pygame.time.get_ticks()*0.004) * 5

        shadow = pygame.Surface((100, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0,0,0,50), (0,0,100,22))
        surf.blit(shadow, (px-50, py+50+int(bob)))

        body_pts = [(px-32,py+52+int(bob)), (px+32,py+52+int(bob)), (px+22,py-20+int(bob)), (px-22,py-20+int(bob))]
        pygame.draw.polygon(surf, (0,40,80), body_pts)
        pygame.draw.polygon(surf, (0,80,160), body_pts, 2)

        pygame.draw.circle(surf, (20,60,100), (px, py-30+int(bob)), 25)
        pygame.draw.circle(surf, (40,120,200), (px, py-30+int(bob)), 25, 2)

        for ex, ey in [(px-9, py-30+int(bob)), (px+9, py-30+int(bob))]:
            gl = get_cached_glow(10, C_SHIELD)
            surf.blit(gl, (ex-10, ey-10))
            pygame.draw.circle(surf, (100,220,255), (ex,ey), 4)

        if self.drawing and self.tracking_tip:
            gl = get_cached_glow(16, C_SLASH)
            tx, ty = self.tracking_tip
            surf.blit(gl, (tx-16, ty-16))
            pygame.draw.circle(surf, C_WHITE, (tx, ty), 5)

    def _draw_hud(self, surf):
        bw = 220
        bx, by = 20, self.sh - 50
        draw_text_shadow(surf, "PLAYER", FONT_SM, C_PLAYER_HP, (bx, by-22))
        pygame.draw.rect(surf, (0,40,20), (bx-2, by-2, bw+4, 20), border_radius=8)
        hw = int(bw * self.player_hp / self.player_max)
        if hw > 0:
            pygame.draw.rect(surf, C_PLAYER_HP, (bx, by, hw, 16), border_radius=7)
        draw_text_shadow(surf, f"{self.player_hp}/{self.player_max}", FONT_SM, C_WHITE, (bx+bw+8, by))
        draw_text_shadow(surf, f"⚡ SCORE: {self.score}", FONT_MED, C_GOLD, (self.sw//2 - 80, 14))

        if self.combo >= 2:
            pulse = 0.8 + 0.2*math.sin(pygame.time.get_ticks()*0.01)
            c = (int(255*pulse), int(160*pulse), 0)
            draw_text_shadow(surf, f"COMBO x{self.combo}!", FONT_BIG, c, (self.sw//2 - 90, 50))

        if self.drawing:
            draw_text_shadow(surf, "✏ DRAWING…", FONT_SM, C_WHITE, (20, 20))

    def _draw_camera(self, surf):
        bx = self.sw - self.cam_w - 14
        by = 14
        if self.cam_frame is None:
            pygame.draw.rect(surf, (20,20,40), (bx, by, self.cam_w, self.cam_h), border_radius=8)
            return
        
        cam_surf = pygame.surfarray.make_surface(np.transpose(self.cam_frame, (1,0,2)))
        pygame.draw.rect(surf, (0,80,160), (bx-3, by-3, self.cam_w+6, self.cam_h+6), border_radius=8)
        surf.blit(cam_surf, (bx, by))

        if self.tracking_tip:
            dot_x = int(bx + (self.tracking_tip[0] / self.sw) * self.cam_w)
            dot_y = int(by + (self.tracking_tip[1] / self.sh) * self.cam_h)
            color = (0, 255, 100) if self.finger_up else (255, 200, 0)
            pygame.draw.circle(surf, color, (dot_x, dot_y), 5)

    def _draw_overlay(self, surf, title, color, subtitle):
        ov = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 160))
        surf.blit(ov, (0,0))
        draw_text_shadow(surf, title, FONT_TITLE, color, (self.sw//2 - FONT_TITLE.size(title)[0]//2, self.sh//3), (5,5))
        draw_text_shadow(surf, subtitle, FONT_BIG, C_WHITE, (self.sw//2 - FONT_BIG.size(subtitle)[0]//2, self.sh//2))
        draw_text_shadow(surf, "Press R to Restart  |  ESC to Quit", FONT_MED, (180,180,180), (self.sw//2 - 230, self.sh*0.65))

    def run(self):
        running = True
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.resize(event.w, event.h)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and self.state == 'TITLE':
                        self.state = 'PLAYING'
                    elif event.key == pygame.K_r and self.state in ('GAME_OVER', 'WIN'):
                        self.reset_game()
                        self.state = 'PLAYING'
                    elif event.key == pygame.K_1 and self.state == 'PLAYING':
                        self._cast_spell('SLASH')
                    elif event.key == pygame.K_2 and self.state == 'PLAYING':
                        self._cast_spell('LIGHTNING')
                    elif event.key == pygame.K_3 and self.state == 'PLAYING':
                        self._cast_spell('SHIELD')

            self.update(dt)
            self.draw()

        self.stream.stop()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    game = Game()
    game.run()