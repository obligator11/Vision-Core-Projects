import pygame
import math
import random
import json
import os
import sys
import array as _arr

# ── init ──────────────────────────────────────────────────────────────────────
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

W, H = 900, 650
screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
pygame.display.set_caption("FINGER ORBIT")
clock = pygame.time.Clock()

FONT_BIG = pygame.font.SysFont("monospace", 42, bold=True)
FONT_MED = pygame.font.SysFont("monospace", 22, bold=True)
FONT_SM  = pygame.font.SysFont("monospace", 14)

LEADERBOARD_FILE = "finger_orbit_scores.json"

# ── colours ───────────────────────────────────────────────────────────────────
BG         = (10,  10,  18)
WHITE      = (255, 255, 255)
FINGER_COL = (200, 180, 255)
SLOMO_TINT = (30,  60,  200, 18)
ORBITER_COLORS = [
    (140, 120, 255),
    ( 80, 220, 180),
    (255, 180,  80),
]

# ── audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100

def _pcm(buf):
    data = _arr.array("h", buf * 2)
    raw  = bytearray()
    for v in data:
        raw += v.to_bytes(2, "little", signed=True)
    return pygame.mixer.Sound(buffer=bytes(raw))

def _build_sounds():
    sounds = {}
    n = int(SAMPLE_RATE * 0.25)
    buf = []
    for i in range(n):
        t   = i / SAMPLE_RATE
        s   = 2 * (t * 120 - math.floor(t * 120 + 0.5))
        hi  = math.sin(2 * math.pi * 600 * t) if i < n // 2 else 0
        env = 1 - i / n
        buf.append(int((s * 0.25 + hi * 0.08) * env * 32767))
    sounds["collision"] = _pcm(buf)
    n2  = int(SAMPLE_RATE * 0.08)
    buf2 = []
    for i in range(n2):
        t = i / SAMPLE_RATE
        s = math.sin(2 * math.pi * 540 * t)
        buf2.append(int(s * (1 - i / n2) * 0.06 * 32767))
    sounds["chime"] = _pcm(buf2)
    return sounds

def _build_ambient():
    n   = int(SAMPLE_RATE * 2.0)
    buf = []
    for i in range(n):
        t  = i / SAMPLE_RATE
        s  = math.sin(2 * math.pi * 55   * t) * 0.04
        s += math.sin(2 * math.pi * 110  * t) * 0.02
        s += math.sin(2 * math.pi * 82.5 * t) * 0.015
        buf.append(int(s * 32767))
    snd = _pcm(buf)
    snd.set_volume(0.5)
    return snd

try:
    SFX = _build_sounds()
except Exception:
    SFX = {}

try:
    AMBIENT    = _build_ambient()
    AMBIENT_CH = pygame.mixer.Channel(0)
    AMBIENT_CH.play(AMBIENT, loops=-1)
except Exception:
    AMBIENT = None

# ── leaderboard ───────────────────────────────────────────────────────────────
def load_lb():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_lb(scores):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(scores, f)

def add_score(s, wave):
    lb = load_lb()
    lb.append({"score": s, "wave": wave})
    lb.sort(key=lambda x: x["score"], reverse=True)
    lb = lb[:10]
    save_lb(lb)
    return lb

# ── particle ──────────────────────────────────────────────────────────────────
class Particle:
    __slots__ = ("x","y","vx","vy","life","decay","size","color")
    def __init__(self, x, y, color):
        self.x, self.y = float(x), float(y)
        self.vx    = random.uniform(-4, 4)
        self.vy    = random.uniform(-5, 1)
        self.life  = 1.0
        self.decay = random.uniform(0.03, 0.06)
        self.size  = random.uniform(3, 7)
        self.color = color

    def update(self):
        self.x   += self.vx
        self.y   += self.vy
        self.vy  += 0.12
        self.life -= self.decay
        self.size *= 0.97

    def draw(self, surf):
        if self.life <= 0:
            return
        s = max(1, int(self.size))
        alpha = int(self.life * 220)
        tmp = pygame.Surface((s * 2, s * 2), pygame.SRCALPHA)
        pygame.draw.circle(tmp, (*self.color, alpha), (s, s), s)
        surf.blit(tmp, (int(self.x) - s, int(self.y) - s))

# ── orbiter ───────────────────────────────────────────────────────────────────
MAX_TRAIL = 45

class Orbiter:
    def __init__(self, x, y, color, size=6):
        self.x, self.y = float(x), float(y)
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-1.5, 1.5)
        self.color      = color
        self.size       = size
        self.trail      = []
        self.alive      = True
        self.glow_phase = random.uniform(0, math.pi * 2)

    def update(self, dt, mx, my, gravity, gW, gH):
        dx   = mx - self.x
        dy   = my - self.y
        dist = math.hypot(dx, dy) or 1
        force = gravity / (dist * dist + 200)
        self.vx += dx / dist * force * dt
        self.vy += dy / dist * force * dt
        speed = math.hypot(self.vx, self.vy)
        if speed > 9:
            self.vx = self.vx / speed * 9
            self.vy = self.vy / speed * 9
        self.vx *= 0.996
        self.vy *= 0.996
        self.x  += self.vx * dt
        self.y  += self.vy * dt
        if self.x < -20:   self.x = gW + 20
        if self.x > gW+20: self.x = -20
        if self.y < -20:   self.y = gH + 20
        if self.y > gH+20: self.y = -20
        self.trail.append((self.x, self.y))
        if len(self.trail) > MAX_TRAIL:
            self.trail.pop(0)
        self.glow_phase += 0.06

    def draw(self, surf):
        tl = len(self.trail)
        r, g, b = self.color
        for i in range(1, tl):
            alpha = int((i / tl) * 180)
            lw    = max(1, int((i / tl) * self.size * 0.9))
            x0, y0 = int(self.trail[i-1][0]), int(self.trail[i-1][1])
            x1, y1 = int(self.trail[i][0]),   int(self.trail[i][1])
            bw = abs(x1-x0) + lw*2 + 2
            bh = abs(y1-y0) + lw*2 + 2
            if bw < 1: bw = 1
            if bh < 1: bh = 1
            tmp = pygame.Surface((bw, bh), pygame.SRCALPHA)
            ox  = min(x0, x1) - lw - 1
            oy  = min(y0, y1) - lw - 1
            pygame.draw.line(tmp, (r, g, b, alpha),
                             (x0-ox, y0-oy), (x1-ox, y1-oy), lw)
            surf.blit(tmp, (ox, oy))
        glow = 0.6 + math.sin(self.glow_phase) * 0.4
        hr   = int(self.size * 2.5)
        halo = pygame.Surface((hr*2, hr*2), pygame.SRCALPHA)
        pygame.draw.circle(halo, (r, g, b, int(50 * glow)), (hr, hr), hr)
        surf.blit(halo, (int(self.x)-hr, int(self.y)-hr))
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), self.size)
        pygame.draw.circle(surf, WHITE,      (int(self.x), int(self.y)), max(2, self.size//2))

# ── obstacle ──────────────────────────────────────────────────────────────────
class Obstacle:
    def __init__(self, wave, gW, gH, offset_x=0):
        self.w    = random.randint(18, 38)
        self.h    = random.randint(40, 100)
        self.x    = float(gW + 60 + offset_x)
        self.y    = float(random.randint(30, max(31, gH - self.h - 30)))
        self.speed   = 1.8 + wave * 0.28 + random.uniform(0, 0.5)
        self.pulse   = 0.0
        self.passed  = False
        self.is_gate = random.random() < 0.3
        if self.is_gate:
            gap         = random.randint(90, 140)
            self.gate_y = self.y + self.h + gap
            self.gate_h = random.randint(35, 70)

    def update(self, dt):
        self.x    -= self.speed * dt
        self.pulse += 0.07

    def rects(self):
        rects = [pygame.Rect(int(self.x), int(self.y), self.w, self.h)]
        if self.is_gate:
            rects.append(pygame.Rect(int(self.x), int(self.gate_y), self.w, self.gate_h))
        return rects

    def offscreen(self):
        return self.x < -80

    def draw(self, surf):
        glow = 0.7 + math.sin(self.pulse) * 0.3
        col  = (int(255*glow), 60, 40)
        for rect in self.rects():
            pygame.draw.rect(surf, col, rect, border_radius=3)
            hi = pygame.Rect(rect.x+1, rect.y+1, rect.w-2, 3)
            pygame.draw.rect(surf, (255, 140, 100), hi, border_radius=2)

    def hits(self, orb):
        ox, oy = int(orb.x), int(orb.y)
        for rect in self.rects():
            cx = max(rect.left, min(ox, rect.right))
            cy = max(rect.top,  min(oy, rect.bottom))
            if math.hypot(ox-cx, oy-cy) < orb.size:
                return True
        return False

    def near_dist(self, orb):
        ox, oy = int(orb.x), int(orb.y)
        best   = 9999.0
        for rect in self.rects():
            cx   = max(rect.left, min(ox, rect.right))
            cy   = max(rect.top,  min(oy, rect.bottom))
            best = min(best, math.hypot(ox-cx, oy-cy))
        return best

# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_grid(surf, gW, gH):
    gs = pygame.Surface((gW, gH), pygame.SRCALPHA)
    for x in range(0, gW, 60):
        pygame.draw.line(gs, (255, 255, 255, 8), (x, 0), (x, gH))
    for y in range(0, gH, 60):
        pygame.draw.line(gs, (255, 255, 255, 8), (0, y), (gW, y))
    surf.blit(gs, (0, 0))

def draw_finger(surf, mx, my, frame):
    r    = 28 + math.sin(frame * 0.05) * 5
    halo = pygame.Surface((int(r*4), int(r*4)), pygame.SRCALPHA)
    for i in range(3, 0, -1):
        a = int(30 * i / 3)
        pygame.draw.circle(halo, (*FINGER_COL, a),
                           (int(r*2), int(r*2)), int(r * i / 1.2))
    surf.blit(halo, (int(mx - r*2), int(my - r*2)))
    pygame.draw.circle(surf, WHITE, (int(mx), int(my)), 5)
    for i in range(3):
        a  = frame * 0.05 + i * (math.pi * 2 / 3)
        d  = 16 + i * 5
        px = int(mx + math.cos(a) * d)
        py = int(my + math.sin(a) * d)
        dot = pygame.Surface((6, 6), pygame.SRCALPHA)
        pygame.draw.circle(dot, (*FINGER_COL, 200 - i*40), (3, 3), 3)
        surf.blit(dot, (px-3, py-3))

def draw_slomo(surf, gW, gH):
    ov = pygame.Surface((gW, gH), pygame.SRCALPHA)
    ov.fill(SLOMO_TINT)
    surf.blit(ov, (0, 0))
    lbl = FONT_SM.render("SLOW MOTION", True, (100, 160, 255))
    surf.blit(lbl, (gW//2 - lbl.get_width()//2, gH - 28))

CAM_W, CAM_H = 100, 75

def draw_minimap(surf, orbiters, obstacles, mx, my, gW, gH):
    sx, sy = CAM_W / gW, CAM_H / gH
    cam    = pygame.Surface((CAM_W, CAM_H))
    cam.fill((5, 5, 15))
    for ob in obstacles:
        for rect in ob.rects():
            r2 = pygame.Rect(int(rect.x*sx), int(rect.y*sy),
                             max(2, int(rect.w*sx)), max(2, int(rect.h*sy)))
            pygame.draw.rect(cam, (200, 60, 40), r2)
    for o in orbiters:
        pygame.draw.circle(cam, o.color, (int(o.x*sx), int(o.y*sy)), 3)
    pygame.draw.circle(cam, FINGER_COL, (int(mx*sx), int(my*sy)), 3)
    pygame.draw.rect(cam, (80, 80, 120), cam.get_rect(), 1)
    x0 = gW - CAM_W - 12
    surf.blit(cam, (x0, 12))
    lbl = FONT_SM.render("LIVE", True, (100, 255, 150))
    surf.blit(lbl, (x0 + 2, 12 + CAM_H - 16))

def draw_title(surf, gW, gH):
    surf.fill(BG)
    draw_grid(surf, gW, gH)
    t = FONT_BIG.render("FINGER ORBIT", True, WHITE)
    surf.blit(t, (gW//2 - t.get_width()//2, gH//2 - 110))
    lines = [
        ("Move your mouse — objects orbit your cursor.", FINGER_COL),
        ("Guide them through obstacles. Don't crash.",   FINGER_COL),
        ("",                                             WHITE),
        ("Press  SPACE  or  ENTER  to start",            WHITE),
    ]
    for i, (text, col) in enumerate(lines):
        r = FONT_SM.render(text, True, col)
        surf.blit(r, (gW//2 - r.get_width()//2, gH//2 - 30 + i*24))

def draw_gameover(surf, score, wave, lb, gW, gH):
    surf.fill(BG)
    draw_grid(surf, gW, gH)
    go = FONT_BIG.render("GAME OVER", True, (255, 80, 60))
    surf.blit(go, (gW//2 - go.get_width()//2, 55))
    sc = FONT_MED.render(f"Score: {score}   Wave: {wave}", True, WHITE)
    surf.blit(sc, (gW//2 - sc.get_width()//2, 120))
    lb_t = FONT_MED.render("─── LEADERBOARD ───", True, FINGER_COL)
    surf.blit(lb_t, (gW//2 - lb_t.get_width()//2, 175))
    for i, entry in enumerate(lb[:8]):
        col = (255, 215, 0) if i == 0 else WHITE
        row = FONT_SM.render(
            f"#{i+1:02d}   {entry['score']:>6} pts   wave {entry['wave']}", True, col)
        surf.blit(row, (gW//2 - row.get_width()//2, 215 + i*22))
    restart = FONT_MED.render("Press  SPACE  to play again", True, (160, 140, 255))
    surf.blit(restart, (gW//2 - restart.get_width()//2, gH - 55))

# ── helpers ───────────────────────────────────────────────────────────────────
def spawn_orbiters(wave, gW, gH):
    count = min(1 + wave // 2, 3)
    orbs  = []
    for i in range(count):
        a = random.uniform(0, math.pi * 2)
        d = random.uniform(80, 130)
        orbs.append(Orbiter(
            gW//2 + math.cos(a)*d,
            gH//2 + math.sin(a)*d,
            ORBITER_COLORS[i % len(ORBITER_COLORS)],
            size=5 + i,
        ))
    return orbs

# ── main loop ─────────────────────────────────────────────────────────────────
def run():
    global screen, W, H   # allow reassignment on resize

    state       = "title"
    orbiters    = []
    obstacles   = []
    particles   = []
    score       = 0
    wave        = 1
    lives       = 3
    slomo       = False
    slomo_timer = 0
    frame       = 0
    obs_timer   = 0.0
    chime_timer = 0
    last_lb     = []
    mx, my      = W // 2, H // 2

    pygame.mouse.set_visible(False)

    while True:
        dt = clock.tick(60) / 16.67   # 1.0 == one frame at 60 fps

        # ── events ────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ev.type == pygame.VIDEORESIZE:
                W, H   = ev.w, ev.h
                screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

            if ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos

            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if state in ("title", "gameover"):
                        state       = "playing"
                        score       = 0
                        wave        = 1
                        lives       = 3
                        slomo       = False
                        obs_timer   = 0
                        orbiters    = spawn_orbiters(wave, W, H)
                        obstacles   = []
                        particles   = []

        # ── title ─────────────────────────────────────────────────────────────
        if state == "title":
            draw_title(screen, W, H)
            draw_finger(screen, mx, my, frame)
            frame += 1
            pygame.display.flip()
            continue

        # ── game over ─────────────────────────────────────────────────────────
        if state == "gameover":
            draw_gameover(screen, score, wave, last_lb, W, H)
            draw_finger(screen, mx, my, frame)
            frame += 1
            pygame.display.flip()
            continue

        # ── playing ───────────────────────────────────────────────────────────

        # slow-motion: trigger when any orbiter is within 32 px of an obstacle
        near = 9999.0
        if orbiters and obstacles:
            thresh = max(o.size for o in orbiters)
            for o in orbiters:
                for ob in obstacles:
                    near = min(near, ob.near_dist(o))
            if near < 32 and near > thresh:
                slomo       = True
                slomo_timer = 80
        if slomo:
            slomo_timer -= 1
            if slomo_timer <= 0:
                slomo = False

        eff_dt = dt * (0.22 if slomo else 1.0)

        # spawn obstacles
        obs_timer += eff_dt
        interval   = max(100 - wave * 7, 42)
        if obs_timer >= interval:
            obs_timer = 0
            obstacles.append(Obstacle(wave, W, H))

        # update orbiters
        gravity = 5200 + wave * 500
        for o in orbiters:
            o.update(eff_dt, mx, my, gravity, W, H)

        # chime near cursor
        chime_timer += 1
        if chime_timer > 35:
            for o in orbiters:
                if math.hypot(mx - o.x, my - o.y) < 110:
                    if "chime" in SFX:
                        SFX["chime"].set_volume(0.3)
                        SFX["chime"].play()
                    chime_timer = 0
                    break

        # update / score obstacles
        for ob in obstacles:
            ob.update(eff_dt)
            if ob.offscreen() and not ob.passed:
                ob.passed = True
                score    += 10
        obstacles = [ob for ob in obstacles if not ob.offscreen()]

        # collision
        dead = set()
        for i, o in enumerate(orbiters):
            for ob in obstacles:
                if ob.hits(o):
                    dead.add(i)
                    for _ in range(22):
                        particles.append(Particle(o.x, o.y, o.color))
                    if "collision" in SFX:
                        SFX["collision"].set_volume(0.6)
                        SFX["collision"].play()
        orbiters = [o for i, o in enumerate(orbiters) if i not in dead]
        if dead:
            lives -= len(dead)
            if lives <= 0:
                last_lb = add_score(score, wave)
                state   = "gameover"
                continue
        if not orbiters:
            orbiters = spawn_orbiters(wave, W, H)

        # wave progression
        if score > wave * 160:
            wave += 1

        # particles
        for p in particles:
            p.update()
        particles = [p for p in particles if p.life > 0]

        # ── draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_grid(screen, W, H)

        if slomo:
            draw_slomo(screen, W, H)

        for ob in obstacles:
            ob.draw(screen)
        for p in particles:
            p.draw(screen)
        for o in orbiters:
            o.draw(screen)

        draw_finger(screen, mx, my, frame)

        # HUD
        lbl = FONT_SM.render("SCORE", True, (120, 120, 140))
        screen.blit(lbl, (16, 14))
        sc_txt = FONT_BIG.render(str(score), True, WHITE)
        screen.blit(sc_txt, (16, 30))
        wv_txt = FONT_SM.render(f"WAVE {wave}", True, FINGER_COL)
        screen.blit(wv_txt, (16, 76))
        for i in range(lives):
            pygame.draw.circle(screen, (255, 100, 80), (16 + i*16, 98), 5)

        draw_minimap(screen, orbiters, obstacles, mx, my, W, H)

        frame += 1
        pygame.display.flip()


if __name__ == "__main__":
    run()


