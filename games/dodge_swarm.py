

import cv2
import numpy as np
import mediapipe as mp
import pygame
import threading
import queue
import time
import random
import math

# ------------------------------------------------------------------------
# CONFIG — tune everything here, nothing else in the file needs touching
# ------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720
POSE_MODEL_COMPLEXITY = 1          # 0=lite/fast, 1=balanced, 2=heavy/slow
POSE_MIN_DET_CONF = 0.6
POSE_MIN_TRACK_CONF = 0.6

PLAYER_MAX_HP = 5
IFRAME_SECONDS = 0.9               # invincibility window after a hit
HITBOX_PADDING = 1.15              # multiplier on torso size for hitbox
GRAZE_PADDING = 1.9                # outer ring that counts as a "graze"

ENEMY_BASE_SPEED = 180.0           # px/sec
ENEMY_SPEED_PER_WAVE = 22.0
ENEMY_BASE_SPAWN_INTERVAL = 1.15   # seconds between spawns at wave 1
ENEMY_SPAWN_INTERVAL_FLOOR = 0.28
ENEMY_TURN_RATE = 2.6              # radians/sec max turn (steering limit)
ENEMY_RADIUS = 22
WAVE_DURATION = 18.0               # seconds per wave before difficulty bump

PARTICLE_COUNT_HIT = 26
PARTICLE_COUNT_GRAZE = 10
PARTICLE_LIFETIME = 0.5

SAMPLE_RATE = 44100

# Neon color palette (BGR since we draw with OpenCV)
COL_BG = (18, 12, 8)
COL_SKELETON = (255, 220, 60)
COL_HITBOX = (60, 255, 180)
COL_HITBOX_HIT = (60, 60, 255)
COL_GRAZE_RING = (255, 180, 60)
COL_ENEMY = (80, 60, 255)
COL_TEXT = (240, 240, 240)
COL_HP = (90, 255, 120)
COL_HP_LOST = (60, 60, 200)


# ------------------------------------------------------------------------
# THREADED CAMERA GRABBER
# Runs in its own daemon thread so a slow game frame never causes the
# webcam driver to buffer up stale frames. Queue size 1 => always fresh.
# ------------------------------------------------------------------------
class FrameGrabber:
    def __init__(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # mirror so movement feels natural
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # drop stale frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.cap.release()


# ------------------------------------------------------------------------
# PROCEDURAL AUDIO — every SFX is synthesized once at startup, no files
# ------------------------------------------------------------------------
def _tone(freq, duration, volume=0.5, wave="sine", decay=True):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    if wave == "sine":
        wav = np.sin(2 * np.pi * freq * t)
    elif wave == "square":
        wav = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        wav = np.random.uniform(-1, 1, t.shape)  # noise
    if decay:
        env = np.linspace(1, 0, t.shape[0])
        wav *= env
    wav = (wav * volume * 32767).astype(np.int16)
    stereo = np.repeat(wav.reshape(-1, 1), 2, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


class Sfx:
    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        self.hit = _tone(120, 0.25, 0.7, "square")
        self.graze = _tone(880, 0.08, 0.35, "sine")
        self.spawn = _tone(500, 0.06, 0.15, "sine")
        self.wave_clear = _tone(660, 0.35, 0.5, "sine")
        self.game_over = _tone(90, 0.9, 0.6, "square")

    def play(self, sound):
        sound.play()


# ------------------------------------------------------------------------
# GAME ENTITIES
# ------------------------------------------------------------------------
class Enemy:
    def __init__(self, spawn_pos, target_pos, speed):
        self.pos = np.array(spawn_pos, dtype=float)
        direction = np.array(target_pos) - self.pos
        self.angle = math.atan2(direction[1], direction[0])
        self.speed = speed
        self.radius = ENEMY_RADIUS
        self.alive = True

    def update(self, dt, target_pos):
        # Steering: turn toward target at a limited rate instead of
        # snapping directly onto it every frame -> dodgeable.
        desired = math.atan2(target_pos[1] - self.pos[1],
                              target_pos[0] - self.pos[0])
        diff = (desired - self.angle + math.pi) % (2 * math.pi) - math.pi
        max_turn = ENEMY_TURN_RATE * dt
        self.angle += max(-max_turn, min(max_turn, diff))
        self.pos += np.array([math.cos(self.angle), math.sin(self.angle)]) * self.speed * dt

    def draw(self, frame):
        p = tuple(self.pos.astype(int))
        cv2.circle(frame, p, self.radius, COL_ENEMY, -1)
        cv2.circle(frame, p, self.radius, (255, 255, 255), 2)
        # small motion tail for readability at speed
        tail = (int(self.pos[0] - math.cos(self.angle) * 18),
                int(self.pos[1] - math.sin(self.angle) * 18))
        cv2.line(frame, p, tail, COL_ENEMY, 3)


class Particle:
    def __init__(self, pos, color):
        self.pos = np.array(pos, dtype=float)
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(60, 260)
        self.vel = np.array([math.cos(ang), math.sin(ang)]) * spd
        self.color = color
        self.life = PARTICLE_LIFETIME
        self.max_life = PARTICLE_LIFETIME

    def update(self, dt):
        self.pos += self.vel * dt
        self.vel *= 0.9
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        alpha = max(0.0, self.life / self.max_life)
        radius = max(1, int(6 * alpha))
        cv2.circle(frame, tuple(self.pos.astype(int)), radius, self.color, -1)


# ------------------------------------------------------------------------
# POSE -> PLAYER HITBOX
# Uses shoulder + hip landmarks to build an ellipse hitbox that scales
# with how close/far the player is standing from the camera.
# ------------------------------------------------------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


def get_player_geometry(landmarks, w, h):
    """Returns (centroid_xy, half_width, half_height) or None if not visible."""
    try:
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    except (IndexError, TypeError):
        return None

    pts = [ls, rs, lh, rh]
    if any(p.visibility < 0.4 for p in pts):
        return None

    sx = np.array([ls.x, rs.x]) * w
    sy = np.array([ls.y, rs.y]) * h
    hx = np.array([lh.x, rh.x]) * w
    hy = np.array([lh.y, rh.y]) * h

    shoulder_mid = np.array([sx.mean(), sy.mean()])
    hip_mid = np.array([hx.mean(), hy.mean()])
    centroid = (shoulder_mid + hip_mid) / 2.0

    shoulder_width = abs(sx[0] - sx[1])
    torso_height = np.linalg.norm(shoulder_mid - hip_mid)

    half_w = max(30.0, shoulder_width * 0.75 * HITBOX_PADDING)
    half_h = max(40.0, torso_height * 0.9 * HITBOX_PADDING)
    return centroid, half_w, half_h


def point_in_ellipse(point, center, half_w, half_h, scale=1.0):
    dx = (point[0] - center[0]) / (half_w * scale)
    dy = (point[1] - center[1]) / (half_h * scale)
    return dx * dx + dy * dy <= 1.0


# ------------------------------------------------------------------------
# MAIN GAME
# ------------------------------------------------------------------------
class DodgeSwarm:
    def __init__(self):
        self.grabber = FrameGrabber(CAM_INDEX)
        self.pose = mp_pose.Pose(
            model_complexity=POSE_MODEL_COMPLEXITY,
            min_detection_confidence=POSE_MIN_DET_CONF,
            min_tracking_confidence=POSE_MIN_TRACK_CONF,
        )
        self.sfx = Sfx()
        self.reset()

    def reset(self):
        self.hp = PLAYER_MAX_HP
        self.score = 0
        self.wave = 1
        self.wave_timer = 0.0
        self.spawn_timer = 0.0
        self.last_hit_time = -999.0
        self.enemies = []
        self.particles = []
        self.game_over = False
        self.start_time = time.time()

    def spawn_enemy(self, w, h, target_pos):
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            pos = (random.uniform(0, w), -ENEMY_RADIUS)
        elif edge == "bottom":
            pos = (random.uniform(0, w), h + ENEMY_RADIUS)
        elif edge == "left":
            pos = (-ENEMY_RADIUS, random.uniform(0, h))
        else:
            pos = (w + ENEMY_RADIUS, random.uniform(0, h))
        speed = ENEMY_BASE_SPEED + ENEMY_SPEED_PER_WAVE * (self.wave - 1)
        speed *= random.uniform(0.85, 1.2)
        self.enemies.append(Enemy(pos, target_pos, speed))
        self.sfx.play(self.sfx.spawn)

    def spawn_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(Particle(pos, color))

    def update(self, dt, geometry):
        if self.game_over:
            return

        self.wave_timer += dt
        if self.wave_timer >= WAVE_DURATION:
            self.wave_timer = 0.0
            self.wave += 1
            self.sfx.play(self.sfx.wave_clear)

        spawn_interval = max(
            ENEMY_SPAWN_INTERVAL_FLOOR,
            ENEMY_BASE_SPAWN_INTERVAL - 0.07 * (self.wave - 1),
        )
        self.spawn_timer += dt
        target_pos = geometry[0] if geometry else (FRAME_W / 2, FRAME_H / 2)
        if self.spawn_timer >= spawn_interval:
            self.spawn_timer = 0.0
            self.spawn_enemy(FRAME_W, FRAME_H, target_pos)

        now = time.time()
        invincible = (now - self.last_hit_time) < IFRAME_SECONDS

        for enemy in self.enemies:
            enemy.update(dt, target_pos)

        # collisions against player hitbox
        surviving = []
        for enemy in self.enemies:
            off_screen = (enemy.pos[0] < -80 or enemy.pos[0] > FRAME_W + 80 or
                          enemy.pos[1] < -80 or enemy.pos[1] > FRAME_H + 80)
            if off_screen:
                continue  # despawn cleanly, no penalty for missed enemies

            if geometry:
                centroid, hw, hh = geometry
                if point_in_ellipse(enemy.pos, centroid, hw, hh):
                    if not invincible:
                        self.hp -= 1
                        self.last_hit_time = now
                        self.spawn_particles(enemy.pos, COL_HITBOX_HIT, PARTICLE_COUNT_HIT)
                        self.sfx.play(self.sfx.hit)
                        if self.hp <= 0:
                            self.game_over = True
                            self.sfx.play(self.sfx.game_over)
                    continue  # enemy consumed on hit either way
                elif point_in_ellipse(enemy.pos, centroid, hw, hh, scale=GRAZE_PADDING):
                    self.score += 2
                    self.spawn_particles(enemy.pos, COL_GRAZE_RING, PARTICLE_COUNT_GRAZE)
                    self.sfx.play(self.sfx.graze)
                    surviving.append(enemy)
                    continue

            surviving.append(enemy)
            self.score += 0  # enemy still in play, no score yet
        self.enemies = surviving

        self.particles = [p for p in self.particles if p.update(dt)]
        self.score += int(dt * (5 + self.wave))  # slow passive score for survival

    def draw_hud(self, frame):
        for i in range(PLAYER_MAX_HP):
            color = COL_HP if i < self.hp else COL_HP_LOST
            cv2.circle(frame, (30 + i * 34, 30), 12, color, -1)
        cv2.putText(frame, f"SCORE {self.score}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_TEXT, 2)
        cv2.putText(frame, f"WAVE {self.wave}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_TEXT, 2)

        if self.game_over:
            cv2.putText(frame, "GAME OVER", (FRAME_W // 2 - 160, FRAME_H // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (60, 60, 255), 4)
            cv2.putText(frame, "Press R to restart", (FRAME_W // 2 - 150, FRAME_H // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_TEXT, 2)

    def draw_player(self, frame, geometry, landmarks):
        if landmarks:
            mp_draw.draw_landmarks(
                frame, landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=COL_SKELETON, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=COL_SKELETON, thickness=2),
            )
        if geometry:
            centroid, hw, hh = geometry
            now = time.time()
            invincible = (now - self.last_hit_time) < IFRAME_SECONDS
            color = COL_HITBOX_HIT if invincible else COL_HITBOX
            cv2.ellipse(frame, tuple(centroid.astype(int)), (int(hw), int(hh)),
                        0, 0, 360, color, 2)
            cv2.ellipse(frame, tuple(centroid.astype(int)),
                        (int(hw * GRAZE_PADDING), int(hh * GRAZE_PADDING)),
                        0, 0, 360, COL_GRAZE_RING, 1)

    def run(self):
        prev_time = time.time()
        calibrating = True
        calib_start = time.time()

        while True:
            frame_bgr = self.grabber.read()
            frame_bgr = cv2.resize(frame_bgr, (FRAME_W, FRAME_H))
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb)

            landmarks = result.pose_landmarks
            geometry = None
            if landmarks:
                geometry = get_player_geometry(landmarks.landmark, FRAME_W, FRAME_H)

            canvas = frame_bgr.copy()
            overlay = np.zeros_like(canvas)
            cv2.addWeighted(canvas, 0.55, overlay, 0.0, 0, canvas)  # slight darken hook (kept simple)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            dt = min(dt, 0.05)  # clamp for stability on stutter/frame drops

            if calibrating:
                remaining = 2.5 - (now - calib_start)
                cv2.putText(canvas, "STAND IN FRAME", (FRAME_W // 2 - 190, FRAME_H // 2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, COL_TEXT, 3)
                if remaining > 0:
                    cv2.putText(canvas, f"Starting in {max(0, int(remaining) + 1)}",
                                (FRAME_W // 2 - 120, FRAME_H // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_TEXT, 2)
                else:
                    calibrating = False
                    self.reset()
            else:
                self.update(dt, geometry)
                for enemy in self.enemies:
                    enemy.draw(canvas)
                for particle in self.particles:
                    particle.draw(canvas)

            self.draw_player(canvas, geometry, landmarks)
            self.draw_hud(canvas)

            cv2.imshow("Dodge Swarm", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.game_over:
                self.reset()
            elif key == ord('c'):
                calibrating = True
                calib_start = time.time()

        self.grabber.stop()
        self.pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DodgeSwarm().run()