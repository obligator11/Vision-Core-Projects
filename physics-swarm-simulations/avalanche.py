"""
Sayyam AI Lab — Spectral Snowball
──────────────────────────────────
✦ Snow drifts in from every edge of the screen
✦ Close your fist  → snowflakes magnetically gather into a snowball at your palm
✦ Open your hand   → snowball launches in the direction you were moving
Press Q to quit.
"""

import cv2
import numpy as np
import mediapipe as mp
import pymunk
import multiprocessing as mp_os
import random
import math
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_fist(hand_landmarks, w, h):
    """Return True when the hand is closed into a fist."""
    palm   = hand_landmarks.landmark[0]   # wrist
    tips   = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]  # finger tips
    dists  = [math.hypot((palm.x - t.x) * w, (palm.y - t.y) * h) for t in tips]
    return all(d < 70 for d in dists)


def _palm_center(hand_landmarks, w, h):
    """Mid-palm position (landmark 9 — middle-finger base)."""
    lm = hand_landmarks.landmark[9]
    return lm.x * w, lm.y * h


def _spawn_snowflake(space, w, h):
    """Create one snowflake body from a random screen edge."""
    radius = random.uniform(5, 12)
    mass   = radius * 0.08
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body   = pymunk.Body(mass, moment)

    edge = random.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        body.position = random.randint(0, w), -radius - 5
    elif edge == "bottom":
        body.position = random.randint(0, w), h + radius + 5
    elif edge == "left":
        body.position = -radius - 5, random.randint(0, h)
    else:
        body.position = w + radius + 5, random.randint(0, h)

    shape            = pymunk.Circle(body, radius)
    shape.elasticity = 0.75
    shape.friction   = 0.2
    space.add(body, shape)

    # Drift toward a random point in the middle third of the screen
    tx = random.randint(w // 4, 3 * w // 4)
    ty = random.randint(h // 4, 3 * h // 4)
    dx = tx - body.position.x
    dy = ty - body.position.y
    angle = math.atan2(dy, dx)
    speed = random.uniform(80, 220)
    body.velocity = math.cos(angle) * speed, math.sin(angle) * speed

    return {"body": body, "radius": radius, "gathered": False}


# ─────────────────────────────────────────────────────────────────────────────
#  WORKER  (vision + physics — runs in a separate process)
# ─────────────────────────────────────────────────────────────────────────────

def vision_and_physics_worker(data_queue):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    hands    = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_draw = mp.solutions.drawing_utils

    # ── Pymunk world ──────────────────────────────────────────────────────────
    space         = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.88          # gentle drag simulates air / snow flutter

    # Kinematic "finger-tip" colliders (one hand → 21 landmarks)
    hand_bodies = []
    for _ in range(21):
        b      = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        s      = pymunk.Circle(b, 10)
        s.elasticity = 0.4
        s.friction   = 0.2
        space.add(b, s)
        hand_bodies.append(b)

    snowflakes: list[dict] = []

    # ── State tracking ────────────────────────────────────────────────────────
    frame_count      = 0
    hand_closed      = False
    prev_hand_closed = False
    hand_detected    = False
    palm_pos         = (320, 240)
    palm_history: deque = deque(maxlen=10)   # rolling window for velocity

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # ── Hand detection ────────────────────────────────────────────────────
        prev_hand_closed = hand_closed
        hand_closed      = False
        hand_detected    = False

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Draw skeleton (subtle, icy colours)
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(180, 220, 255), thickness=1, circle_radius=2),
                    mp_draw.DrawingSpec(color=(100, 160, 255), thickness=1, circle_radius=1),
                )

                # Move kinematic colliders to landmark positions
                for i, lm in enumerate(hand_lms.landmark):
                    if i < len(hand_bodies):
                        hand_bodies[i].position = lm.x * w, lm.y * h

                # Update palm state
                palm_pos  = _palm_center(hand_lms, w, h)
                palm_history.append(palm_pos)
                hand_detected = True

                if _is_fist(hand_lms, w, h):
                    hand_closed = True
        else:
            for b in hand_bodies:
                b.position = -300, -300

        # ── Palm throw velocity (smoothed over last few frames) ───────────────
        palm_velocity = (0.0, 0.0)
        if len(palm_history) >= 4:
            vx = (palm_history[-1][0] - palm_history[-4][0]) * 12
            vy = (palm_history[-1][1] - palm_history[-4][1]) * 12
            palm_velocity = (vx, vy)

        just_opened = prev_hand_closed and not hand_closed   # ← throw moment

        # ── Spawn snowflakes ──────────────────────────────────────────────────
        if frame_count % 8 == 0 and len(snowflakes) < 70:
            snowflakes.append(_spawn_snowflake(space, w, h))

        # ── Physics update for each snowflake ─────────────────────────────────
        to_remove = []
        for flake in snowflakes:
            b    = flake["body"]
            px, py = palm_pos
            dx   = px - b.position.x
            dy   = py - b.position.y
            dist = math.hypot(dx, dy) or 1

            if hand_closed and hand_detected:
                # ── GATHER: strong magnetic pull toward palm ──────────────────
                if dist < 300:
                    strength = 4000 / (dist + 8)
                    b.apply_force_at_local_point(
                        (dx / dist * strength, dy / dist * strength), (0, 0)
                    )
                # Damp flakes that are already very close (they "stick")
                if dist < 45:
                    b.velocity = b.velocity * 0.55
                    flake["gathered"] = True
                else:
                    flake["gathered"] = False

            elif just_opened and flake["gathered"]:
                # ── THROW: launch with palm velocity ─────────────────────────
                throw_speed = math.hypot(*palm_velocity)
                if throw_speed > 60:
                    # Fan-out a little so they don't all travel identically
                    spread = random.uniform(-0.25, 0.25)
                    angle  = math.atan2(palm_velocity[1], palm_velocity[0]) + spread
                    speed  = throw_speed * random.uniform(0.8, 1.3)
                    b.velocity = math.cos(angle) * speed, math.sin(angle) * speed
                else:
                    # Gentle scatter if the hand barely moved
                    angle = random.uniform(0, 2 * math.pi)
                    b.velocity = math.cos(angle) * 350, math.sin(angle) * 350
                flake["gathered"] = False

            else:
                # ── FREE DRIFT: gentle turbulence + soft centre gravity ───────
                flake["gathered"] = False
                fx = random.uniform(-60, 60)
                fy = random.uniform(-60, 60)
                # Very weak centre pull keeps flakes in the playfield
                fx += ((w / 2) - b.position.x) * 0.05
                fy += ((h / 2) - b.position.y) * 0.05
                b.apply_force_at_local_point((fx, fy), (0, 0))

            # Cull flakes that have left the screen entirely
            mx, my = b.position
            if mx < -200 or mx > w + 200 or my < -200 or my > h + 200:
                to_remove.append(flake)

        for flake in to_remove:
            if flake in snowflakes:
                b = flake["body"]
                space.remove(b, *b.shapes)
                snowflakes.remove(flake)

        space.step(1 / 30.0)

        # ── Pack data for renderer ────────────────────────────────────────────
        ball_data = [
            (int(f["body"].position.x), int(f["body"].position.y),
             f["radius"], f["gathered"])
            for f in snowflakes
        ]

        # Drop stale frame if renderer is behind
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
            except Exception:
                pass

        data_queue.put((frame, ball_data, hand_closed, palm_pos))
        frame_count += 1


# ─────────────────────────────────────────────────────────────────────────────
#  RENDERER  (main process)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralSnowballEngine:
    def __init__(self):
        self.data_queue    = mp_os.Queue(maxsize=2)
        self.worker_process = mp_os.Process(
            target=vision_and_physics_worker,
            args=(self.data_queue,),
            daemon=True,
        )

    # ── Snowflake visual ──────────────────────────────────────────────────────
    @staticmethod
    def _draw_snowflake(canvas, x, y, r):
        """Crisp white circle with a faint blue halo."""
        cv2.circle(canvas, (x, y), max(int(r), 1), (255, 255, 255), -1)
        cv2.circle(canvas, (x, y), max(int(r) + 2, 2), (180, 210, 255), 1)

    # ── Snowball visual ───────────────────────────────────────────────────────
    @staticmethod
    def _draw_snowball(canvas, cx, cy, count):
        """A growing icy sphere at the palm."""
        base_r = 22
        r      = int(base_r + count * 1.8)
        # Outer soft glow
        cv2.circle(canvas, (cx, cy), r + 8, (140, 190, 255), 2)
        # Main body — layered for depth
        cv2.circle(canvas, (cx, cy), r,     (230, 245, 255), -1)
        cv2.circle(canvas, (cx, cy), r - 4, (255, 255, 255), -1)
        # Specular highlight
        hx, hy = cx - r // 3, cy - r // 3
        cv2.circle(canvas, (hx, hy), max(r // 4, 4), (255, 255, 255), -1)

    def run(self):
        self.worker_process.start()
        win = "Sayyam AI Lab: Spectral Snowball"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        # ── Persistent "snow trail" layer for motion blur feel ────────────────
        trail = None

        while True:
            if not self.data_queue.empty():
                frame, ball_data, hand_closed, palm_pos = self.data_queue.get()

                h, w = frame.shape[:2]
                if trail is None:
                    trail = np.zeros_like(frame, dtype=np.float32)

                # Fade the trail toward the camera feed (gives motion blur)
                frame_f   = frame.astype(np.float32)
                trail     = cv2.addWeighted(trail, 0.55, frame_f, 0.45, 0)

                overlay   = np.zeros((h, w, 3), dtype=np.uint8)

                free      = [(x, y, r) for x, y, r, g in ball_data if not g]
                gathered  = [(x, y, r) for x, y, r, g in ball_data if g]

                # ── Free snowflakes ───────────────────────────────────────────
                for x, y, r in free:
                    self._draw_snowflake(overlay, x, y, r)

                # ── Snowball (when fist closed) ────────────────────────────────
                px, py = int(palm_pos[0]), int(palm_pos[1])
                if hand_closed and gathered:
                    self._draw_snowball(overlay, px, py, len(gathered))
                    # Also draw the individual trapped flakes inside
                    for x, y, r in gathered:
                        cv2.circle(overlay, (x, y), max(int(r * 0.7), 2),
                                   (210, 235, 255), -1)

                # ── Glow pass ─────────────────────────────────────────────────
                blur  = cv2.GaussianBlur(overlay, (21, 21), 0)
                glow  = cv2.addWeighted(overlay, 1.0, blur, 0.9, 0)

                # Composite: trail → camera → glow
                base  = np.clip(trail, 0, 255).astype(np.uint8)
                out   = cv2.addWeighted(base, 0.6, frame, 0.6, 0)
                out   = cv2.addWeighted(out,  1.0, glow, 1.0, 0)

                # ── HUD ───────────────────────────────────────────────────────
                status   = "✊ GATHERING — open hand to throw!" if hand_closed \
                           else "🤚 Drift • Close fist to collect snow"
                txt_col  = (160, 220, 255)
                cv2.putText(out, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_col, 2, cv2.LINE_AA)
                cv2.putText(out, f"Snowflakes: {len(ball_data)}", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,  txt_col, 1, cv2.LINE_AA)
                if gathered:
                    cv2.putText(out, f"Snowball size: {len(gathered)}", (10, 82),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(out, "Q — quit", (w - 90, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 150, 200), 1, cv2.LINE_AA)

                cv2.imshow(win, out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.worker_process.terminate()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp_os.freeze_support()
    SpectralSnowballEngine().run()