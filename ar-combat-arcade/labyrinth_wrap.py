import sys
import cv2
import numpy as np
import mediapipe as mp
import pygame
import threading
import queue
import math
import time

IS_WINDOWS = sys.platform.startswith("win")


# ──────────────────────────────────────────────
# AUDIO
# ──────────────────────────────────────────────
class SoundSynthesizer:
    @staticmethod
    def generate_sine_wave(freq, duration, volume=0.5, sample_rate=22050):
        num_samples = int(sample_rate * duration)
        x = np.linspace(0, duration, num_samples, endpoint=False)
        wave = np.sin(2 * np.pi * freq * x) * 32767 * volume
        return np.column_stack((wave.astype(np.int16), wave.astype(np.int16)))

    @staticmethod
    def generate_noise_wave(duration, volume=0.3, sample_rate=22050):
        num_samples = int(sample_rate * duration)
        noise = (np.random.uniform(-1, 1, num_samples) * 32767 * volume).astype(np.int16)
        return np.column_stack((noise, noise))


# ──────────────────────────────────────────────
# CAMERA HELPER
# ──────────────────────────────────────────────
def open_camera(index: int = 0):
    """
    Open the webcam robustly on every platform.

    Root-cause of the Windows black-screen bug
    ──────────────────────────────────────────
    cv2.VideoCapture(0) without a backend flag on Windows tries MSMF
    (Microsoft Media Foundation) first.  MSMF frequently:
      • Returns ok=True but delivers black/blank frames for the first
        20-30 frames while the sensor warms up, OR
      • Hangs indefinitely on the first cap.read() call.

    Fix: on Windows always use CAP_DSHOW first.  Then warm the sensor
    by reading and discarding frames until we get a non-black one (or
    we exhaust retries).  On non-Windows we use the default backend.
    """
    def _try_open(backend_flag=None):
        if backend_flag is not None:
            cap = cv2.VideoCapture(index, backend_flag)
        else:
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Keep buffer small so we always get the freshest frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ── choose backend order ────────────────────────────────────────
    if IS_WINDOWS:
        # DSHOW is the most reliable webcam backend on Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    else:
        backends = [None, cv2.CAP_V4L2]   # Linux/macOS

    cap = None
    for backend in backends:
        cap = _try_open(backend)
        if cap is not None:
            print(f"[CAM] Opened with backend={backend}")
            break

    if cap is None:
        print("[CAM] ERROR: could not open any camera.")
        return None

    # ── warm-up: drain frames until the sensor is actually live ────
    # DSHOW on Windows returns black/blank frames for the first several
    # cycles while the driver initialises AGC and auto-exposure.
    # We drain quickly (no sleep) and accept the first frame whose mean
    # exceeds 1.0, or give up after 120 attempts (~2 s at 60 fps).
    print("[CAM] Warming up sensor…")
    for attempt in range(120):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 1.0:
            print(f"[CAM] Live frame after {attempt} reads.")
            return cap

    # Still here → camera opens but only delivers black frames.
    # Return it anyway; the live loop will keep retrying.
    print("[CAM] WARNING: sensor still dark after warm-up; returning cap anyway.")
    return cap


# ──────────────────────────────────────────────
# INFERENCE WORKER  (background thread)
# ──────────────────────────────────────────────
def inference_worker_thread(in_q: queue.Queue, out_q: queue.Queue,
                            stop_event: threading.Event):
    """
    Why threading instead of multiprocessing
    ─────────────────────────────────────────
    multiprocessing.Queue pickles every frame across an OS pipe.
    On Windows this causes:
      • Deadlocks when pygame/cv2 hold internal locks at fork time
      • Per-frame IPC overhead that stalls the main render loop
    A daemon thread shares memory directly → no pickling, no stalls.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    )

    while not stop_event.is_set():
        mini_frame = None
        try:
            mini_frame = in_q.get(timeout=0.02)
            # Drain stale frames; only the freshest matters
            while not in_q.empty():
                try:
                    mini_frame = in_q.get_nowait()
                except queue.Empty:
                    break
        except queue.Empty:
            continue

        if mini_frame is None:
            continue

        rgb_frame = cv2.cvtColor(mini_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        angle = 0.0
        detected = False
        if results.multi_hand_landmarks:
            wrist = results.multi_hand_landmarks[0].landmark[
                mp.solutions.hands.HandLandmark.WRIST
            ]
            angle = (wrist.x - 0.5) * -1.8
            detected = True

        # Overwrite stale result with the fresh one
        while not out_q.empty():
            try:
                out_q.get_nowait()
            except queue.Empty:
                break
        try:
            out_q.put_nowait({"detected": detected, "angle": angle})
        except queue.Full:
            pass

    hands.close()


# ──────────────────────────────────────────────
# MAZE
# ──────────────────────────────────────────────
class MazeSystem:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.walls = []
        self.goal_pos = (0, 0)
        self.radius = 200
        self.generate_maze()

    def generate_maze(self):
        self.walls = []
        self.walls += [
            ((-180, -180), ( 180, -180)),
            (( 180, -180), ( 180,  180)),
            (( 180,  180), (-180,  180)),
            ((-180,  180), (-180, -180)),
            ((-120, -120), ( 120, -120)),
            (( 120, -120), ( 120,   60)),
            ((-120,    0), (  60,    0)),
            (( -60,   60), ( -60,  180)),
            ((   0,  -60), (   0,   60)),
            ((-120,  120), ( 120,  120)),
        ]
        self.goal_pos = (150, 150)

    def rotate_point(self, pt, rad):
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        x, y = pt
        return x * cos_a - y * sin_a + self.cx, x * sin_a + y * cos_a + self.cy

    def get_transformed_walls(self, rad):
        return [(self.rotate_point(p1, rad), self.rotate_point(p2, rad))
                for p1, p2 in self.walls]

    def get_transformed_goal(self, rad):
        return self.rotate_point(self.goal_pos, rad)


# ──────────────────────────────────────────────
# MAIN ENGINE
# ──────────────────────────────────────────────
class LabyrinthWarpEngine:
    def __init__(self):
        pygame.mixer.pre_init(22050, -16, 2, 512)
        pygame.mixer.init()

        self.in_q:  queue.Queue = queue.Queue(maxsize=2)
        self.out_q: queue.Queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.width, self.height = 1280, 720
        self.game_cx, self.game_cy = 820, 360

        self.ball_pos = [0.0, 0.0]
        self.ball_vel = [0.0, 0.0]
        self.ball_radius = 12

        self.state = "STANDBY"
        self.last_cross = 0.0
        self.cross_count = 0
        self.last_sign = 0

        self.current_angle = 0.0
        self.smoothed_angle = 0.0

        self.maze = MazeSystem(self.game_cx, self.game_cy)
        self.reset_ball()
        self._compile_audio()

    # ── audio ──────────────────────────────────────────────────────
    def _compile_audio(self):
        self.snd_roll = pygame.sndarray.make_sound(
            SoundSynthesizer.generate_sine_wave(120, 0.1, 0.15))
        self.snd_hit = pygame.sndarray.make_sound(
            SoundSynthesizer.generate_noise_wave(0.05, 0.4))
        self.snd_win = pygame.sndarray.make_sound(
            SoundSynthesizer.generate_sine_wave(587.33, 0.15, 0.3))
        self.roll_channel = pygame.mixer.Channel(0)

    def reset_ball(self):
        self.ball_pos = [-150.0, -150.0]
        self.ball_vel = [0.0, 0.0]

    # ── physics ────────────────────────────────────────────────────
    def _check_line_collision(self, px, py, p1, p2):
        ax, ay = p1;  bx, by = p2
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            return None
        t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
        cx, cy = ax + t*dx, ay + t*dy
        return math.hypot(px-cx, py-cy), (cx, cy)

    def process_physics(self, walls):
        gx = 0.45 * math.sin(-self.smoothed_angle)
        gy = 0.45 * math.cos( self.smoothed_angle)
        self.ball_vel[0] = (self.ball_vel[0] + gx) * 0.98
        self.ball_vel[1] = (self.ball_vel[1] + gy) * 0.98

        speed = math.hypot(*self.ball_vel)
        if speed > 0.3 and self.state == "PLAYING":
            if not self.roll_channel.get_busy():
                self.roll_channel.play(self.snd_roll, loops=-1)
        else:
            self.roll_channel.stop()

        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]

        collision = False
        bx = self.ball_pos[0] + self.game_cx
        by = self.ball_pos[1] + self.game_cy

        for p1, p2 in walls:
            res = self._check_line_collision(bx, by, p1, p2)
            if res:
                dist, (cx, cy) = res
                if dist < self.ball_radius:
                    collision = True
                    overlap = self.ball_radius - dist
                    nx, ny = bx - cx, by - cy
                    n_len = math.hypot(nx, ny)
                    nx, ny = (nx/n_len, ny/n_len) if n_len > 0 else (0, -1)
                    bx += nx * overlap;  by += ny * overlap
                    vd = self.ball_vel[0]*nx + self.ball_vel[1]*ny
                    self.ball_vel[0] -= 1.4 * vd * nx
                    self.ball_vel[1] -= 1.4 * vd * ny

        self.ball_pos[0] = bx - self.game_cx
        self.ball_pos[1] = by - self.game_cy
        if collision and speed > 1.5:
            self.snd_hit.play()

    # ── calibration ────────────────────────────────────────────────
    def evaluate_standby_calibration(self, angle):
        thr = 0.15
        now = time.time()
        sign = 1 if angle > thr else (-1 if angle < -thr else 0)
        if sign == 0:
            return
        if self.last_sign == 0:
            self.last_sign = sign;  self.last_cross = now
        elif sign != self.last_sign:
            self.cross_count = self.cross_count + 1 if now - self.last_cross < 2.0 else 1
            self.last_sign = sign;  self.last_cross = now
        if self.cross_count >= 4:
            self.state = "PLAYING"
            self.reset_ball()
            self.snd_win.play()

    # ── HUD ────────────────────────────────────────────────────────
    def render_overlay_hud(self, canvas, cam_frame):
        canvas[:, :] = 15
        pip_w, pip_h = 320, 240

        # Use the live frame; if it looks black/corrupt, tint it so the
        # user can tell the PiP is alive even before gesture detection.
        if cam_frame is not None and cam_frame.size > 0:
            pip = cv2.resize(cam_frame, (pip_w, pip_h))
        else:
            pip = np.full((pip_h, pip_w, 3), 30, dtype=np.uint8)
            cv2.putText(pip, "NO SIGNAL", (70, pip_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2, cv2.LINE_AA)

        canvas[120:120+pip_h, 30:30+pip_w] = pip
        cv2.rectangle(canvas, (30, 120), (30+pip_w, 120+pip_h), (0, 215, 255), 2)

        cv2.putText(canvas, "LABYRINTH-WARP: KINETIC AIR-MOUSE", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "TRACKING: MEDIAPIPE HANDS (THREADED)", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        if self.state == "STANDBY":
            cv2.rectangle(canvas, (30, 390), (350, 660), (25, 25, 25), -1)
            cv2.rectangle(canvas, (30, 390), (350, 660), (0, 0, 255), 1)
            cv2.putText(canvas, "CALIBRATION REQ",           (50, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,   0, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"WAVES: {self.cross_count}/4", (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "SWIPE PALM LEFT / RIGHT",    (50, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "4 TIMES TO START",           (50, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(130, 130, 130), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (30, 390), (350, 660), (25, 25, 25), -1)
            cv2.rectangle(canvas, (30, 390), (350, 660), (0, 255, 0), 1)
            cv2.putText(canvas, "ENGINE ACTIVE", (50, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"TILT: {math.degrees(self.smoothed_angle):.1f} DEG",
                        (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # ── maze draw ──────────────────────────────────────────────────
    def draw_maze_universe(self, canvas, walls, goal):
        cv2.circle(canvas, (self.game_cx, self.game_cy), self.maze.radius + 30, (22, 22, 22), -1)
        cv2.circle(canvas, (self.game_cx, self.game_cy), self.maze.radius,      (45, 45, 45),  2)

        gx, gy = int(goal[0]), int(goal[1])
        cv2.circle(canvas, (gx, gy), 20, (0, 230, 80), -1)
        cv2.circle(canvas, (gx, gy), 30, (0, 230, 80),  2)

        for p1, p2 in walls:
            cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     (240, 240, 240), 4, cv2.LINE_AA)

        bx = int(self.ball_pos[0] + self.game_cx)
        by = int(self.ball_pos[1] + self.game_cy)

        gr = 24
        x1, y1 = max(0, bx-gr), max(0, by-gr)
        x2, y2 = min(self.width, bx+gr), min(self.height, by+gr)
        if x2 > x1 and y2 > y1:
            roi = canvas[y1:y2, x1:x2].astype(np.int32)
            canvas[y1:y2, x1:x2] = np.clip(roi + 40, 0, 255).astype(np.uint8)

        cv2.circle(canvas, (bx, by), self.ball_radius,     (255, 255, 255), -1)
        cv2.circle(canvas, (bx, by), self.ball_radius - 3, (0, 215, 255),   -1)

        if (math.hypot((self.ball_pos[0]+self.game_cx) - goal[0],
                       (self.ball_pos[1]+self.game_cy) - goal[1]) < 22
                and self.state == "PLAYING"):
            self.state = "STANDBY"
            self.cross_count = 0
            self.snd_win.play()
            self.roll_channel.stop()

    # ── main loop ──────────────────────────────────────────────────
    def run(self):
        cv2.namedWindow("Labyrinth-Warp Engine", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Labyrinth-Warp Engine", self.width, self.height)

        # Open camera with the Windows-aware helper
        cap = open_camera(0)
        cam_ok = cap is not None

        # Start inference thread (not subprocess — avoids IPC hangs)
        worker = threading.Thread(
            target=inference_worker_thread,
            args=(self.in_q, self.out_q, self.stop_event),
            daemon=True,
        )
        worker.start()

        canvas    = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Grey placeholder so the PiP is never invisibly black at startup
        cam_frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        while True:
            # ── grab frame ─────────────────────────────────────────
            if cam_ok:
                ret, raw = cap.read()
                if ret and raw is not None and raw.size > 0:
                    cam_frame = cv2.flip(raw, 1)   # mirror = intuitive tilt control
                # if read() fails just keep the previous valid frame

            # ── feed worker ────────────────────────────────────────
            if cam_ok:
                mini = cv2.resize(cam_frame, (256, 192))
                if self.in_q.full():
                    try:    self.in_q.get_nowait()
                    except: pass
                try:    self.in_q.put_nowait(mini)
                except: pass

            # ── consume inference result ───────────────────────────
            try:
                while not self.out_q.empty():
                    payload = self.out_q.get_nowait()
                    if payload["detected"]:
                        self.current_angle = payload["angle"]
                        if self.state == "STANDBY":
                            self.evaluate_standby_calibration(self.current_angle)
            except: pass

            # ── physics + render ───────────────────────────────────
            self.smoothed_angle = 0.88 * self.smoothed_angle + 0.12 * self.current_angle

            walls = self.maze.get_transformed_walls(self.smoothed_angle)
            goal  = self.maze.get_transformed_goal(self.smoothed_angle)

            self.process_physics(walls)
            self.render_overlay_hud(canvas, cam_frame)
            self.draw_maze_universe(canvas, walls, goal)

            cv2.imshow("Labyrinth-Warp Engine", canvas)

            # 16 ms ≈ 60 fps; keeps the window message-pump alive on Windows
            if cv2.waitKey(16) & 0xFF == ord('q'):
                break

        # ── cleanup ────────────────────────────────────────────────
        self.stop_event.set()
        self.roll_channel.stop()
        if cam_ok:
            cap.release()
        worker.join(timeout=2.0)
        cv2.destroyAllWindows()
        pygame.mixer.quit()


if __name__ == "__main__":
    engine = LabyrinthWarpEngine()
    engine.run()