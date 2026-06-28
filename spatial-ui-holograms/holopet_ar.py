import cv2
import pygame
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import threading
import time
import heapq
import random
import math
import os

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CAM_W, CAM_H   = 640, 480
GRID_W, GRID_H = 64, 48
CELL_W = CAM_W // GRID_W   # 10 px
CELL_H = CAM_H // GRID_H   # 10 px

# YOLO model path — same folder as this script, then spatial-ui-holograms folder
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_YOLO_PATHS  = [
    os.path.join(_SCRIPT_DIR, "spatial-ui-holograms", "yolov8n.pt"),
    os.path.join(_SCRIPT_DIR, "yolov8n.pt"),
    "yolov8n.pt",   # fallback: auto-download
]
YOLO_MODEL_PATH = next((p for p in _YOLO_PATHS if os.path.exists(p)), "yolov8n.pt")

# YOLO class IDs to treat as OBSTACLES (exclude hands/arms – COCO has no "hand" class anyway)
# We block only large static objects: person(0), chair(56), couch(57), dining table(60),
# tv(62), laptop(63), book(73), vase(75), bottle(39), cup(41)
OBSTACLE_CLASSES = {0, 39, 41, 56, 57, 60, 62, 63, 73, 75}
# But person body bounding boxes include the arm — so we SHRINK person boxes to avoid
# falsely blocking the hand region above the wrist. The shrink factor below cuts the
# person box to its lower 60 % (torso + legs only).
PERSON_BBOX_SHRINK = 0.40   # chop top 40 % of a person box

# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO  — pure PCM, zero external files
# ─────────────────────────────────────────────────────────────────────────────
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

def _tone(freq, dur, vol=0.25, wave="sine"):
    sr  = 44100
    n   = int(sr * dur)
    t   = np.linspace(0, dur, n, endpoint=False)
    if   wave == "square":   raw = np.sign(np.sin(2*math.pi*freq*t))
    elif wave == "sawtooth": raw = 2*(t*freq - np.floor(t*freq+0.5))
    else:                    raw = np.sin(2*math.pi*freq*t)
    fade = min(int(sr*0.02), n)
    raw[-fade:] *= np.linspace(1, 0, fade)
    pcm = (raw * vol * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.ascontiguousarray(np.column_stack([pcm, pcm])))

SND_STEP  = _tone(280,  0.04, 0.10, "square")
SND_FOOD  = _tone(880,  0.14, 0.28, "sine")
SND_TOUCH = _tone(1400, 0.09, 0.22, "sine")
SND_BLOCK = _tone(140,  0.18, 0.18, "square")
SND_HAPPY = _tone(660,  0.07, 0.12, "sawtooth")

# ─────────────────────────────────────────────────────────────────────────────
#  THREAD-SAFE CAMERA
# ─────────────────────────────────────────────────────────────────────────────
class CameraThread:
    def __init__(self, src=0):
        self.cap     = None
        self.frame   = None
        self.running = True
        self.lock    = threading.Lock()
        self._open_camera(src)
        threading.Thread(target=self._loop, daemon=True).start()

    def _open_camera(self, src):
        """Try DSHOW first (Windows), verify a real frame arrives, fall back if not."""
        for backend in [cv2.CAP_DSHOW, 0]:  # 0 = default backend
            cap = cv2.VideoCapture(src, backend) if backend else cv2.VideoCapture(src)
            if not cap.isOpened():
                cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            # Verify we actually get a frame within 2 seconds
            deadline = time.time() + 2.0
            while time.time() < deadline:
                ret, f = cap.read()
                if ret and f is not None:
                    self.cap = cap
                    print(f"[Camera] Opened with backend={'DSHOW' if backend else 'default'}")
                    return
                time.sleep(0.05)
            cap.release()
        # Last resort: any backend, no verification
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        print("[Camera] WARNING: could not verify camera — trying anyway")

    def _loop(self):
        while self.running:
            if self.cap is None:
                time.sleep(0.05); continue
            ret, f = self.cap.read()
            if ret and f is not None:
                with self.lock:
                    self.frame = cv2.flip(f, 1)
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

# ─────────────────────────────────────────────────────────────────────────────
#  NAVMESH  — YOLO background thread, hand-safe obstacle filtering
# ─────────────────────────────────────────────────────────────────────────────
class NavMesh:
    def __init__(self, camera):
        self.camera  = camera
        print(f"[NavMesh] Loading YOLO from: {YOLO_MODEL_PATH}")
        self.model   = YOLO(YOLO_MODEL_PATH)
        self.grid    = np.ones((GRID_W, GRID_H), dtype=bool)
        self.running = True
        self.lock    = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.05); continue

            results  = self.model(frame, imgsz=320, verbose=False)
            new_grid = np.ones((GRID_W, GRID_H), dtype=bool)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls not in OBSTACLE_CLASSES:
                        continue                        # ← skip hands / unknown objects

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Shrink person box upward so hands above body aren't blocked
                    if cls == 0:
                        h_box = y2 - y1
                        y1 = y1 + int(h_box * PERSON_BBOX_SHRINK)

                    gx1 = max(0, x1 // CELL_W)
                    gy1 = max(0, y1 // CELL_H)
                    gx2 = min(GRID_W-1, x2 // CELL_W)
                    gy2 = min(GRID_H-1, y2 // CELL_H)
                    new_grid[gx1:gx2+1, gy1:gy2+1] = False

            with self.lock:
                self.grid = new_grid
            time.sleep(0.12)

    def is_walkable(self, x, y):
        if 0 <= x < GRID_W and 0 <= y < GRID_H:
            with self.lock:
                return bool(self.grid[x, y])
        return False

    def snapshot(self):
        with self.lock:
            return self.grid.copy()

# ─────────────────────────────────────────────────────────────────────────────
#  A*
# ─────────────────────────────────────────────────────────────────────────────
def astar(navmesh, start, goal):
    if not navmesh.is_walkable(goal[0], goal[1]):
        return []
    DIRS = [(0,1),(1,0),(0,-1),(-1,0),(-1,-1),(1,1),(-1,1),(1,-1)]
    frontier  = [(0, start)]
    came_from = {start: None}
    cost      = {start: 0.0}
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == goal: break
        for dx, dy in DIRS:
            nxt = (cur[0]+dx, cur[1]+dy)
            if not navmesh.is_walkable(nxt[0], nxt[1]): continue
            nc = cost[cur] + (1.414 if dx and dy else 1.0)
            if nc < cost.get(nxt, 1e18):
                cost[nxt] = nc
                heapq.heappush(frontier, (nc + abs(goal[0]-nxt[0])+abs(goal[1]-nxt[1]), nxt))
                came_from[nxt] = cur
    if goal not in came_from: return []
    path, cur = [], goal
    while cur != start:
        path.append(cur); cur = came_from[cur]
    path.reverse()
    return path

# ─────────────────────────────────────────────────────────────────────────────
#  MECH SPIDER PET — procedural, no external assets
# ─────────────────────────────────────────────────────────────────────────────
class MechSpiderRenderer:
    """
    Renders a glowing mechanical spider:
      • Rounded hexagonal core body with inner circuit-trace lines
      • 6 articulated legs (2 joints each), animated with a tripod gait
      • Glowing visor eye that tracks the direction of movement
      • Antenna with energy orb
      • Spark + ring shockwave on touch
      • Idle breathing pulse + step squash
    """

    C_BODY   = (0,   220, 255)   # cyan
    C_DARK   = (5,   18,  35)
    C_VISOR  = (255, 240, 80)    # yellow visor
    C_ACCENT = (160, 60,  255)   # violet accents
    C_SPARK  = (255, 220, 60)
    C_SHELL  = (20,  60,  90)

    def __init__(self):
        self.bob      = 0.0
        self.gait     = 0.0      # 0–2π phase
        self.step_sq  = 1.0
        self.touched  = False
        self.touch_t  = 0.0
        self.sparks   = []       # [rx,ry,vx,vy,life,max_life]
        self.rings    = []       # [r_cur, r_max, life, max_life]
        self.blink_t  = 0.0
        self.blink_ev = random.uniform(3, 6)
        self.blink_a  = 0.0
        self.dir_x    = 1.0     # movement direction for visor tracking
        self.dir_y    = 0.0

    def notify_step(self, dx=0, dy=0):
        self.step_sq = 0.75
        if dx or dy:
            mag = math.hypot(dx, dy) or 1
            self.dir_x, self.dir_y = dx/mag, dy/mag

    def notify_touch(self):
        self.touched = True
        self.touch_t = 0.8
        for _ in range(22):
            a   = random.uniform(0, math.tau)
            spd = random.uniform(2, 5)
            life= random.uniform(0.3, 0.75)
            self.sparks.append([0, 0, math.cos(a)*spd, math.sin(a)*spd, life, life])
        self.rings.append([0, 60, 0.5, 0.5])

    def update(self, dt):
        self.bob   = math.sin(time.time() * 2.2) * 3
        self.gait += dt * 5.0    # leg animation speed
        self.step_sq += (1.0 - self.step_sq) * 0.20

        self.blink_a += dt
        if self.blink_a >= self.blink_ev:
            self.blink_t  = 0.10
            self.blink_a  = 0
            self.blink_ev = random.uniform(3, 6)
        self.blink_t = max(0, self.blink_t - dt)

        if self.touch_t > 0:
            self.touch_t -= dt
            if self.touch_t <= 0: self.touched = False

        dead_s = [s for s in self.sparks if s[4] <= 0]
        for s in self.sparks:
            s[0]+=s[2]; s[1]+=s[3]; s[3]+=0.35; s[4]-=dt
        for s in dead_s: self.sparks.remove(s)

        dead_r = [r for r in self.rings if r[2] <= 0]
        for r in self.rings:
            r[0] = r[1] * (1 - r[2]/r[3])
            r[2] -= dt
        for r in dead_r: self.rings.remove(r)

    # ── internal draw helpers ─────────────────
    @staticmethod
    def _hex_pts(cx, cy, rx, ry, angle_off=0):
        pts = []
        for i in range(6):
            a = math.tau * i / 6 + angle_off
            pts.append((int(cx + rx*math.cos(a)), int(cy + ry*math.sin(a))))
        return pts

    def _draw_leg(self, surf, cx, cy, r, leg_angle, phase, side):
        """Draw one leg with 2-joint procedural animation."""
        upper_len = r * 1.1
        lower_len = r * 0.9
        foot_bob  = math.sin(phase) * r * 0.22 * side

        # shoulder
        sx = cx + math.cos(leg_angle) * r * 0.55
        sy = cy + math.sin(leg_angle) * r * 0.30
        # knee
        kx = sx + math.cos(leg_angle + 0.45*side) * upper_len
        ky = sy + math.sin(leg_angle + 0.45*side) * upper_len + foot_bob
        # foot
        fx = kx + math.cos(leg_angle - 0.3*side) * lower_len
        fy = ky + math.sin(leg_angle - 0.3*side) * lower_len + foot_bob * 0.5

        pygame.draw.line(surf, self.C_BODY,   (int(sx),int(sy)), (int(kx),int(ky)), 2)
        pygame.draw.line(surf, self.C_ACCENT, (int(kx),int(ky)), (int(fx),int(fy)), 2)
        pygame.draw.circle(surf, self.C_ACCENT, (int(kx),int(ky)), 3)  # knee joint
        pygame.draw.circle(surf, self.C_BODY,   (int(fx),int(fy)), 2)  # foot tip

    # ── main draw ────────────────────────────
    def draw(self, surf, px, py, base_r):
        r   = base_r
        bob = int(self.bob)
        cy  = py + bob
        sq  = self.step_sq
        rw  = int(r * (2.1 - sq) * 0.85)
        rh  = int(r * sq * 0.70)

        # ── Glow aura ────────────────────────
        glow_col   = (160, 40, 255, 100) if self.touched else (0, 180, 255, 50)
        glow_r     = int(rw * 2.5)
        glow_surf  = pygame.Surface((glow_r*2, glow_r*2), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, glow_col, glow_surf.get_rect())
        surf.blit(glow_surf, (px - glow_r, cy - glow_r), special_flags=pygame.BLEND_RGBA_ADD)

        # ── Shock rings ──────────────────────
        for ring in self.rings:
            alpha = max(0, min(255, int(200 * ring[2] / ring[3])))
            rcur  = int(ring[0])
            if rcur < 4:                              # skip until ring has grown enough
                continue
            ring_col = tuple(self.C_BODY[:3]) + (alpha,)
            ring_s = pygame.Surface((rcur*2+4, rcur*2+4), pygame.SRCALPHA)
            pygame.draw.ellipse(ring_s, ring_col,
                                (1, 1, rcur*2+2, rcur*2+2), 2)
            surf.blit(ring_s, (px - rcur - 2, cy - rcur - 2))

        # ── 6 Legs ───────────────────────────
        # 3 legs per side; tripod gait: legs 0,2,4 move together, 1,3,5 opposite
        leg_angles_l = [-0.35, 0.0, 0.35]    # left side (radians from horizontal)
        leg_angles_r = [-0.35, 0.0, 0.35]
        for i, base_a in enumerate(leg_angles_l):
            phase = self.gait + (0 if i%2==0 else math.pi)
            self._draw_leg(surf, px, cy, rw, math.pi + base_a, phase, +1)
        for i, base_a in enumerate(leg_angles_r):
            phase = self.gait + (math.pi if i%2==0 else 0)
            self._draw_leg(surf, px, cy, rw, base_a, phase, -1)

        # ── Shell body (hexagonal) ────────────
        hex_pts = self._hex_pts(px, cy, rw, rh, angle_off=math.pi/6)
        pygame.draw.polygon(surf, self.C_SHELL,  hex_pts)
        pygame.draw.polygon(surf, self.C_BODY,   hex_pts, 2)

        # Inner circuit trace (2 diagonal lines + centre ring)
        for ang in [math.pi/6, math.pi/2, -math.pi/6]:
            x0 = int(px + math.cos(ang) * rw * 0.5)
            y0 = int(cy + math.sin(ang) * rh * 0.5)
            x1 = int(px - math.cos(ang) * rw * 0.5)
            y1 = int(cy - math.sin(ang) * rh * 0.5)
            pygame.draw.line(surf, self.C_BODY[:3], (x0,y0), (x1,y1), 1)
        pygame.draw.circle(surf, self.C_ACCENT, (px, cy), max(2, rh//4), 1)

        # ── Head / visor ─────────────────────
        hx = int(px + self.dir_x * rw * 0.45)
        hy = int(cy + self.dir_y * rh * 0.45)
        hr = max(4, int(r * 0.28))
        pygame.draw.circle(surf, self.C_DARK,   (hx, hy), hr)
        pygame.draw.circle(surf, self.C_BODY,   (hx, hy), hr, 1)

        # Visor eye (elongated, blinks)
        ew, eh = max(3, hr-2), max(2, hr//3)
        if self.blink_t > 0:
            pygame.draw.line(surf, self.C_VISOR,
                             (hx-ew, hy), (hx+ew, hy), 2)
        else:
            eye_rect = pygame.Rect(hx-ew, hy-eh, ew*2, eh*2)
            pygame.draw.ellipse(surf, self.C_VISOR, eye_rect)
            # Shine dot
            pygame.draw.circle(surf, (255,255,255), (hx-ew//3, hy-eh//3), max(1,ew//5))

        # Touch blush
        if self.touched:
            blush = pygame.Surface((hr*4, hr*2), pygame.SRCALPHA)
            pygame.draw.ellipse(blush, (255, 60, 60, 80), blush.get_rect())
            surf.blit(blush, (hx - hr*2, hy - hr))

        # ── Antenna ──────────────────────────
        ant_wave = math.sin(time.time() * 5) * 5
        ant_bx   = px
        ant_by   = cy - rh - 2
        ant_tx   = px + int(ant_wave)
        ant_ty   = ant_by - int(r * 0.6)
        pygame.draw.line(surf, self.C_BODY,   (ant_bx, ant_by), (ant_tx, ant_ty), 2)
        # Energy orb (pulsing)
        orb_r = max(3, int(r*0.12) + int(math.sin(time.time()*8)*2))
        pygame.draw.circle(surf, self.C_ACCENT, (ant_tx, ant_ty), orb_r)
        pygame.draw.circle(surf, (220,160,255), (ant_tx, ant_ty), max(1,orb_r-2))

        # ── Sparks ───────────────────────────
        for s in self.sparks:
            alpha = max(0, int(255 * s[4]/s[5]))
            rs    = max(1, int(3 * s[4]/s[5]))
            sc    = pygame.Surface((rs*2+2, rs*2+2), pygame.SRCALPHA)
            pygame.draw.circle(sc, tuple(self.C_SPARK) + (alpha,), (rs+1, rs+1), rs)
            surf.blit(sc, (px + int(s[0]) - rs, cy + int(s[1]) - rs))

# ─────────────────────────────────────────────────────────────────────────────
#  DIGITAL PET
# ─────────────────────────────────────────────────────────────────────────────
class DigitalPet:
    ROAM_INTERVAL = 3.5

    def __init__(self, renderer):
        self.renderer = renderer
        self.gx = GRID_W // 2
        self.gy = GRID_H // 2
        self.sx = float(self.gx)
        self.sy = float(self.gy)
        self.path     = []
        self.target   = None
        self.food_pos = None
        self.step_t   = 0.0
        self.move_spd = 0.11
        self.roam_t   = 0.0
        self._prev_gx = self.gx
        self._prev_gy = self.gy

    def set_target(self, gx, gy, navmesh):
        if (gx, gy) == (self.gx, self.gy): return
        path = astar(navmesh, (self.gx, self.gy), (gx, gy))
        if path:
            self.path   = path
            self.target = (gx, gy)
        else:
            SND_BLOCK.play()

    def notify_food(self, gx, gy, navmesh):
        self.food_pos = (gx, gy)
        self.set_target(gx, gy, navmesh)

    def notify_touch(self, navmesh):
        self.renderer.notify_touch()
        SND_TOUCH.play()
        for _ in range(20):
            rx = self.gx + random.randint(-10, 10)
            ry = self.gy + random.randint(-8,  8)
            rx = max(2, min(GRID_W-3, rx))
            ry = max(2, min(GRID_H-3, ry))
            if navmesh.is_walkable(rx, ry):
                self.set_target(rx, ry, navmesh)
                break

    def update(self, dt, navmesh):
        self.roam_t += dt

        if self.target and not navmesh.is_walkable(self.target[0], self.target[1]):
            self.target = None; self.path = []; self.food_pos = None
            SND_BLOCK.play()

        if not self.target and self.roam_t >= self.ROAM_INTERVAL:
            self.roam_t = 0.0
            rx = random.randint(4, GRID_W-5)
            ry = random.randint(4, GRID_H-5)
            if navmesh.is_walkable(rx, ry):
                self.set_target(rx, ry, navmesh)

        if self.path:
            self.step_t += dt
            if self.step_t >= self.move_spd:
                self.step_t = 0.0
                nxt = self.path[0]
                if navmesh.is_walkable(nxt[0], nxt[1]):
                    dx = nxt[0] - self.gx
                    dy = nxt[1] - self.gy
                    self.gx, self.gy = nxt
                    self.path.pop(0)
                    self.renderer.notify_step(dx, dy)
                    SND_STEP.play()
                else:
                    if self.target:
                        self.path = astar(navmesh, (self.gx, self.gy), self.target)
                        if not self.path:
                            self.target = None; self.food_pos = None
                            SND_BLOCK.play()
                if not self.path and self.target:
                    if self.food_pos and self.target == self.food_pos:
                        SND_FOOD.play(); self.food_pos = None
                    self.target = None

        self.sx += (float(self.gx) - self.sx) * min(1.0, dt * 14)
        self.sy += (float(self.gy) - self.sy) * min(1.0, dt * 14)

    def pixel_pos(self, ws, hs):
        return (int((self.sx+0.5)*CELL_W*ws),
                int((self.sy+0.5)*CELL_H*hs))

# ─────────────────────────────────────────────────────────────────────────────
#  GESTURE DETECTION — tuned thresholds, proper coordinate mapping
# ─────────────────────────────────────────────────────────────────────────────
class GestureDetector:
    """
    Pinch: thumb tip ↔ index tip normalized distance < 0.07  (was 0.05 — too tight)
    Touch: index fingertip pixel within TOUCH_RADIUS of pet centre
    Finger cursor: always drawn (index tip position)
    """
    PINCH_THRESH  = 0.07
    TOUCH_RADIUS  = 70   # pixels in window space (was ~35 — too small)

    def __init__(self):
        mp_h = mp.solutions.hands
        self.hands = mp_h.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )
        self.pinch_prev = False
        self.touch_prev = False
        # outputs
        self.finger_px  = None   # (x,y) in window pixels
        self.is_pinch   = False
        self.is_touch   = False
        self.pinch_rose = False   # True only on rising edge
        self.touch_rose = False

    def process(self, rgb_frame, win_w, win_h, pet_px, pet_py):
        res = self.hands.process(rgb_frame)

        self.is_pinch   = False
        self.is_touch   = False
        self.pinch_rose = False
        self.touch_rose = False
        self.finger_px  = None

        if res.multi_hand_landmarks:
            hand  = res.multi_hand_landmarks[0]
            lm    = hand.landmark

            # Pinch = thumb (4) ↔ index (8) distance in normalized space
            dx = lm[4].x - lm[8].x
            dy = lm[4].y - lm[8].y
            dist = math.hypot(dx, dy)
            self.is_pinch = dist < self.PINCH_THRESH

            # Index fingertip — MediaPipe normalizes to the input image (CAM_W×CAM_H).
            # Map to camera pixel first, then scale to window space.
            ix_cam = int(lm[8].x * CAM_W)
            iy_cam = int(lm[8].y * CAM_H)
            ix = int(ix_cam * win_w / CAM_W)
            iy = int(iy_cam * win_h / CAM_H)
            self.finger_px = (ix, iy)

            # Touch: finger near pet AND not pinching
            dist_to_pet = math.hypot(ix - pet_px, iy - pet_py)
            self.is_touch = (not self.is_pinch) and (dist_to_pet < self.TOUCH_RADIUS)

        self.pinch_rose = self.is_pinch and not self.pinch_prev
        self.touch_rose = self.is_touch and not self.touch_prev
        self.pinch_prev = self.is_pinch
        self.touch_prev = self.is_touch

    def close(self):
        self.hands.close()

# ─────────────────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(surf, font, food_active, touched, finger_pos, is_pinch, is_touch):
    w, h = surf.get_size()

    # Title + controls
    for i, (txt, col) in enumerate([
        ("HoloPet AR  |  @SayyamAILab",   (0, 255, 200)),
        ("PINCH  → drop food",              (140, 200, 255)),
        ("POINT finger at pet → touch it",  (140, 200, 255)),
    ]):
        surf.blit(font.render(txt, True, col), (12, 10 + i*17))

    # Gesture status
    g_col = (255, 220, 60)  if is_pinch else \
            (255, 80,  120) if is_touch else \
            (80,  100, 120)
    g_txt = "● PINCH" if is_pinch else "● TOUCH" if is_touch else "○ open"
    surf.blit(font.render(g_txt, True, g_col), (12, h - 24))

    if food_active:
        t = font.render("FOOD DROPPED!", True, (255,220,60))
        surf.blit(t, (w - t.get_width() - 12, 10))
    if touched:
        t = font.render("❤  PET TOUCHED!", True, (255, 80, 120))
        surf.blit(t, (w//2 - t.get_width()//2, 10))

    # Finger cursor
    if finger_pos:
        col = (255,220,60) if is_pinch else (0,255,200)
        pygame.draw.circle(surf, col, finger_pos, 10, 2)
        pygame.draw.circle(surf, (*col, 120), finger_pos, 5)
        pygame.draw.line(surf, col, (finger_pos[0]-14, finger_pos[1]),
                         (finger_pos[0]+14, finger_pos[1]), 1)
        pygame.draw.line(surf, col, (finger_pos[0], finger_pos[1]-14),
                         (finger_pos[0], finger_pos[1]+14), 1)

# ─────────────────────────────────────────────────────────────────────────────
#  GAME ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class GameEngine:
    def __init__(self):
        self.screen  = pygame.display.set_mode((CAM_W, CAM_H), pygame.RESIZABLE)
        pygame.display.set_caption("HoloPet AR  |  @SayyamAILab")
        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("consolas", 13, bold=True)

        self.camera   = CameraThread()
        self.navmesh  = NavMesh(self.camera)
        self.renderer = MechSpiderRenderer()
        self.pet      = DigitalPet(self.renderer)
        self.gesture  = GestureDetector()

        self.food_pos = None
        self.running  = True

    def _loading_screen(self):
        """Show animated loading screen until first camera frame arrives."""
        font_big = pygame.font.SysFont("consolas", 22, bold=True)
        font_sm  = pygame.font.SysFont("consolas", 13)
        dots = 0
        while self.running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False; return
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    self.running = False; return
            if self.camera.get_frame() is not None:
                return   # camera ready
            self.screen.fill((5, 12, 25))
            w, h = self.screen.get_size()
            t = time.time()
            # Pulse ring
            for i in range(3):
                r = int(40 + i*25 + 15*math.sin(t*3 + i))
                alpha = max(0, 180 - i*50)
                s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                pygame.draw.ellipse(s, (0, 220, 255, alpha), s.get_rect(), 2)
                self.screen.blit(s, (w//2 - r, h//2 - r - 40))
            # Text
            dots = int(t*2) % 4
            txt = font_big.render("HoloPet AR  " + "." * dots, True, (0, 220, 255))
            self.screen.blit(txt, (w//2 - txt.get_width()//2, h//2 + 20))
            txt2 = font_sm.render("Initialising camera & YOLO navmesh...", True, (80, 140, 180))
            self.screen.blit(txt2, (w//2 - txt2.get_width()//2, h//2 + 52))
            txt3 = font_sm.render("@SayyamAILab", True, (40, 80, 100))
            self.screen.blit(txt3, (w//2 - txt3.get_width()//2, h - 28))
            pygame.display.flip()
            self.clock.tick(30)

    def run(self):
        self._loading_screen()   # blocks until camera ready or quit
        if not self.running:
            self._shutdown(); return

        prev = time.time()
        while self.running:
            now = time.time()
            dt  = min(now - prev, 0.05)
            prev = now

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    self.running = False

            frame = self.camera.get_frame()
            if frame is None:
                self.clock.tick(60); continue

            win_w, win_h = self.screen.get_size()
            ws = win_w / CAM_W
            hs = win_h / CAM_H

            # Gesture detection
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pet_px, pet_py = self.pet.pixel_pos(ws, hs)
            self.gesture.process(rgb, win_w, win_h, pet_px, pet_py)

            # Pinch rising edge → drop food at fingertip grid cell
            if self.gesture.pinch_rose and self.gesture.finger_px:
                ix, iy = self.gesture.finger_px
                # finger_px is in window space; map back to camera then grid
                ix_cam = ix * CAM_W // win_w
                iy_cam = iy * CAM_H // win_h
                gx = max(0, min(GRID_W-1, ix_cam // CELL_W))
                gy = max(0, min(GRID_H-1, iy_cam // CELL_H))
                if self.navmesh.is_walkable(gx, gy):
                    self.food_pos = (gx, gy)
                    self.pet.notify_food(gx, gy, self.navmesh)

            # Touch rising edge → pet reacts
            if self.gesture.touch_rose:
                self.pet.notify_touch(self.navmesh)

            # Logic
            self.renderer.update(dt)
            self.pet.update(dt, self.navmesh)
            if self.food_pos and self.pet.food_pos is None:
                self.food_pos = None

            # ── Render ────────────────────────────────────────────────────
            # 1. Camera background
            frame_surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
            self.screen.blit(pygame.transform.scale(frame_surf, (win_w, win_h)), (0,0))

            # 2. NavMesh overlay (no lock held — uses snapshot)
            grid_snap = self.navmesh.snapshot()
            overlay   = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            cw = int(CELL_W * ws) + 1
            ch = int(CELL_H * hs) + 1
            for gx in range(GRID_W):
                for gy in range(GRID_H):
                    if not grid_snap[gx, gy]:
                        pygame.draw.rect(overlay, (255, 0, 0, 45),
                                         (int(gx*CELL_W*ws), int(gy*CELL_H*hs), cw, ch))
            self.screen.blit(overlay, (0, 0))

            # 3. Path line — draw on a SRCALPHA surface to support alpha colour
            if self.pet.path and len(self.pet.path) >= 1:
                path_surf = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
                pts = [(int((self.pet.sx+0.5)*CELL_W*ws),
                        int((self.pet.sy+0.5)*CELL_H*hs))]
                pts += [(int((nx+0.5)*CELL_W*ws), int((ny+0.5)*CELL_H*hs))
                        for nx, ny in self.pet.path]
                if len(pts) >= 2:
                    pygame.draw.lines(path_surf, (0, 255, 180, 100), False, pts, 1)
                self.screen.blit(path_surf, (0, 0))

            # 4. Food pellet
            if self.food_pos:
                fx, fy = self.food_pos
                fpx = int((fx+0.5)*CELL_W*ws)
                fpy = int((fy+0.5)*CELL_H*hs)
                pulse = int(5 + 3*math.sin(time.time()*7))
                pygame.draw.circle(self.screen, (255,200,0),  (fpx,fpy), pulse+5)
                pygame.draw.circle(self.screen, (255,255,160),(fpx,fpy), pulse)

            # 5. Mech spider
            pet_px, pet_py = self.pet.pixel_pos(ws, hs)
            base_r = int(max(CELL_W*ws, CELL_H*hs) * 1.8)
            self.renderer.draw(self.screen, pet_px, pet_py, base_r)

            # 6. HUD
            draw_hud(self.screen, self.font, self.food_pos is not None,
                     self.renderer.touched,
                     self.gesture.finger_px,
                     self.gesture.is_pinch,
                     self.gesture.is_touch)

            pygame.display.flip()
            self.clock.tick(60)

        self._shutdown()

    def _shutdown(self):
        self.gesture.close()
        self.camera.stop()
        pygame.quit()
        cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = GameEngine()
    engine.run()