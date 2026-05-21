import cv2
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import time
import os
import queue as stdlib_queue
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SlipstreamConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    num_particles: int = 7000
    base_flow_speed: float = 4.5
    flow_speed_variance: float = 1.2
    background_alpha: float = 0.12
    trail_decay: float = 0.87
    collision_dilation_px: int = 9
    depth_foreground_percentile: float = 55.0
    tilt_vortex_threshold_deg: float = 25.0
    vortex_intensity: float = 3.0
    vortex_spray_count: int = 60
    max_particle_age: int = 350
    midas_model_type: str = "MiDaS_small"
    yolo_weights: str = "yolo11n-seg.pt"
    yolo_confidence: float = 0.45
    inference_queue_size: int = 2
    render_queue_size: int = 2


class MiDaSEngine:
    def __init__(self, model_type: str, device: torch.device):
        self.device = device
        self.model = torch.hub.load(
            "intel-isl/MiDaS", model_type, trust_repo=True
        )
        self.model.to(device).eval()
        transform_hub = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self.transform = (
            transform_hub.small_transform
            if model_type == "MiDaS_small"
            else transform_hub.dpt_transform
        )

    @torch.no_grad()
    def predict(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        raw = self.model(inp)
        raw = torch.nn.functional.interpolate(
            raw.unsqueeze(1),
            size=bgr_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = raw.cpu().numpy().astype(np.float32)
        d_min, d_max = depth.min(), depth.max()
        return (depth - d_min) / (d_max - d_min + 1e-8)


class YOLOSegEngine:
    def __init__(self, weights: str, confidence: float):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.confidence = confidence

    def predict(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = bgr_frame.shape[:2]
        results = self.model(bgr_frame, conf=self.confidence, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        tilt = 0.0

        if results and results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy():
                resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, (resized > 0.5).astype(np.uint8))

        if mask.any():
            tilt = self._compute_tilt(mask)

        return mask, tilt

    def _compute_tilt(self, mask: np.ndarray) -> float:
        M = cv2.moments(mask)
        if M["mu20"] == 0 and M["mu02"] == 0:
            return 0.0
        return 0.5 * np.degrees(
            np.arctan2(2.0 * M["mu11"], M["mu20"] - M["mu02"])
        )


class CollisionMesh:
    def __init__(self, config: SlipstreamConfig):
        self.config = config
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.collision_dilation_px, config.collision_dilation_px),
        )

    def build(self, yolo_mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if not yolo_mask.any():
            return np.zeros_like(yolo_mask)
        masked_depth_vals = depth[yolo_mask > 0]
        threshold = np.percentile(
            masked_depth_vals, self.config.depth_foreground_percentile
        )
        foreground = (depth >= threshold).astype(np.uint8)
        combined = np.logical_and(yolo_mask > 0, foreground > 0).astype(np.uint8)
        return cv2.dilate(combined, self.kernel, iterations=1)


class ParticleField:
    def __init__(self, config: SlipstreamConfig):
        self.W = config.width
        self.H = config.height
        self.cfg = config
        n = config.num_particles
        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros((n, 2), dtype=np.float32)
        self.speed = np.zeros(n, dtype=np.float32)
        self.turbulence = np.zeros(n, dtype=np.float32)
        self.age = np.zeros(n, dtype=np.float32)
        self._init_all()

    def _reset(self, idx: np.ndarray):
        n = len(idx)
        if n == 0:
            return
        self.pos[idx, 0] = np.random.uniform(-5, 2, n)
        self.pos[idx, 1] = np.random.uniform(0, self.H, n)
        vx = self.cfg.base_flow_speed + np.random.uniform(
            -self.cfg.flow_speed_variance, self.cfg.flow_speed_variance, n
        )
        self.vel[idx, 0] = vx
        self.vel[idx, 1] = np.random.uniform(-0.4, 0.4, n)
        self.speed[idx] = vx
        self.turbulence[idx] = 0.0
        self.age[idx] = 0.0

    def _init_all(self):
        all_idx = np.arange(self.cfg.num_particles)
        self._reset(all_idx)
        self.pos[:, 0] = np.random.uniform(0, self.W, self.cfg.num_particles)

    def update(self, collision_mask: Optional[np.ndarray], tilt_angle: float):
        self.age += 1.0
        dead = (
            (self.pos[:, 0] > self.W + 5)
            | (self.age > self.cfg.max_particle_age)
            | (self.pos[:, 1] < -10)
            | (self.pos[:, 1] > self.H + 10)
        )
        self._reset(np.where(dead)[0])

        if collision_mask is not None and collision_mask.any():
            self._collide(collision_mask, tilt_angle)

        self.pos += self.vel

        laminar = self.turbulence < 0.05
        self.vel[laminar, 0] += (
            self.cfg.base_flow_speed - self.vel[laminar, 0]
        ) * 0.04
        self.vel[laminar, 1] *= 0.93
        self.turbulence = np.maximum(0.0, self.turbulence - 0.025)
        self.speed = np.linalg.norm(self.vel, axis=1)

    def _collide(self, mask: np.ndarray, tilt_angle: float):
        h, w = mask.shape
        px = np.clip(self.pos[:, 0].astype(np.int32), 0, w - 1)
        py = np.clip(self.pos[:, 1].astype(np.int32), 0, h - 1)
        hit = mask[py, px] > 0
        hi = np.where(hit)[0]

        if len(hi) == 0:
            return

        gx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)

        nx = -gx[py[hi], px[hi]]
        ny = -gy[py[hi], px[hi]]
        mag = np.sqrt(nx ** 2 + ny ** 2) + 1e-8
        nx /= mag
        ny /= mag

        dot = self.vel[hi, 0] * nx + self.vel[hi, 1] * ny
        self.vel[hi, 0] -= 1.9 * dot * nx
        self.vel[hi, 1] -= 1.9 * dot * ny
        self.vel[hi] *= 0.4
        self.turbulence[hi] = 1.0
        self.pos[hi, 0] -= 2.5

        if abs(tilt_angle) > self.cfg.tilt_vortex_threshold_deg:
            self._vortex(hi, tilt_angle)

    def _vortex(self, hit_indices: np.ndarray, tilt_angle: float):
        n = min(self.cfg.vortex_spray_count, len(hit_indices))
        sel = np.random.choice(hit_indices, n, replace=False)
        angle_rad = np.radians(tilt_angle)
        angles = np.random.uniform(0, 2 * np.pi, n)
        self.vel[sel, 0] += self.cfg.vortex_intensity * np.cos(angles + angle_rad)
        self.vel[sel, 1] += self.cfg.vortex_intensity * np.sin(angles + angle_rad)
        self.turbulence[sel] = np.minimum(self.turbulence[sel] + 1.0, 2.0)


class VelocityColorMapper:
    _CYAN = np.array([255, 255, 0], dtype=np.float32)
    _RED = np.array([0, 0, 230], dtype=np.float32)
    _WHITE = np.array([255, 255, 255], dtype=np.float32)

    def __init__(self, max_speed: float):
        self.max_speed = max_speed

    def map(self, speed: np.ndarray, turbulence: np.ndarray) -> np.ndarray:
        n = len(speed)
        colors = np.empty((n, 3), dtype=np.float32)
        norm = np.clip(speed / (self.max_speed + 1e-6), 0.0, 1.0)
        turb = np.clip(turbulence, 0.0, 1.0)
        effective = norm * (1.0 - turb * 0.75)

        high = effective >= 0.62
        low = effective <= 0.28
        mid = ~high & ~low

        colors[high] = self._CYAN

        if low.any():
            t = (effective[low] / 0.28).reshape(-1, 1)
            colors[low] = (1.0 - t) * self._WHITE + t * self._RED

        if mid.any():
            t = ((effective[mid] - 0.28) / 0.34).reshape(-1, 1)
            colors[mid] = (1.0 - t) * self._RED + t * self._CYAN

        return np.clip(colors, 0, 255).astype(np.uint8)


class AeroRenderer:
    _CYAN_HUD = (255, 220, 0)
    _GREEN_HUD = (80, 255, 130)
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, config: SlipstreamConfig):
        self.cfg = config
        self.trail = np.zeros((config.height, config.width, 3), dtype=np.uint8)
        self.win = "SLIPSTREAM | Aerodynamics Lab"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, config.width, config.height)

    def render(
        self,
        frame: np.ndarray,
        positions: np.ndarray,
        colors: np.ndarray,
        collision_mask: Optional[np.ndarray],
        tilt_angle: float,
        fps: float,
    ) -> bool:
        H, W = self.cfg.height, self.cfg.width

        self.trail = (
            self.trail.astype(np.float32) * self.cfg.trail_decay
        ).astype(np.uint8)

        valid = (
            (positions[:, 0] >= 0)
            & (positions[:, 0] < W)
            & (positions[:, 1] >= 0)
            & (positions[:, 1] < H)
        )
        px = positions[valid, 0].astype(np.int32)
        py = positions[valid, 1].astype(np.int32)
        vc = colors[valid]

        self.trail[py, px] = np.maximum(self.trail[py, px], vc)

        dark = (frame.astype(np.float32) * self.cfg.background_alpha).astype(
            np.uint8
        )
        canvas = cv2.add(dark, self.trail)

        if collision_mask is not None and collision_mask.any():
            overlay = np.zeros_like(canvas)
            overlay[collision_mask > 0] = [12, 65, 12]
            cv2.addWeighted(canvas, 1.0, overlay, 0.38, 0, canvas)

        self._draw_hud(canvas, tilt_angle, fps)
        cv2.imshow(self.win, canvas)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def _draw_hud(self, canvas: np.ndarray, tilt: float, fps: float):
        H = canvas.shape[0]
        cv2.putText(
            canvas, "SLIPSTREAM  |  CFD WIND TUNNEL ACTIVE",
            (12, 32), self._FONT, 0.68, self._CYAN_HUD, 2,
        )
        cv2.putText(
            canvas, f"RENDER FPS   {fps:5.1f}",
            (12, 62), self._FONT, 0.55, self._GREEN_HUD, 1,
        )
        cv2.putText(
            canvas, f"OBJECT TILT  {tilt:+.1f} deg",
            (12, 86), self._FONT, 0.52, (60, 200, 255), 1,
        )
        flow_status = (
            "VORTEX DETACHMENT" if abs(tilt) > self.cfg.tilt_vortex_threshold_deg
            else "LAMINAR FLOW"
        )
        flow_color = (0, 80, 255) if "VORTEX" in flow_status else (255, 255, 0)
        cv2.putText(
            canvas, flow_status,
            (12, 110), self._FONT, 0.52, flow_color, 1,
        )

        lx, ly = 12, H - 78
        cv2.rectangle(canvas, (lx, ly), (lx + 195, ly + 65), (28, 28, 28), -1)
        cv2.rectangle(canvas, (lx, ly), (lx + 195, ly + 65), (60, 60, 60), 1)
        cv2.putText(
            canvas, "LAMINAR",
            (lx + 8, ly + 22), self._FONT, 0.44, (255, 255, 0), 1,
        )
        cv2.line(
            canvas, (lx + 88, ly + 17), (lx + 183, ly + 17), (255, 255, 0), 2,
        )
        cv2.putText(
            canvas, "TURBULENT",
            (lx + 8, ly + 50), self._FONT, 0.44, (30, 30, 230), 1,
        )
        cv2.line(
            canvas, (lx + 105, ly + 45), (lx + 183, ly + 45), (0, 0, 220), 2,
        )

    def destroy(self):
        cv2.destroyAllWindows()


def _run_inference(
    config: SlipstreamConfig,
    out_q: Queue,
    stop: Event,
):
    os.environ["PYTHONWARNINGS"] = "ignore"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    midas = MiDaSEngine(config.midas_model_type, device)
    yolo = YOLOSegEngine(config.yolo_weights, config.yolo_confidence)
    mesh = CollisionMesh(config)

    cap = cv2.VideoCapture(config.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(
            cv2.resize(frame, (config.width, config.height)), 1
        )

        try:
            yolo_mask, tilt = yolo.predict(frame)
            depth = midas.predict(frame)
            collision = mesh.build(yolo_mask, depth)
        except Exception:
            collision, tilt = None, 0.0

        try:
            out_q.put_nowait((frame, collision, tilt))
        except stdlib_queue.Full:
            pass

    cap.release()


def _run_cfd(
    config: SlipstreamConfig,
    in_q: Queue,
    out_q: Queue,
    stop: Event,
):
    pf = ParticleField(config)
    cm = VelocityColorMapper(
        config.base_flow_speed + config.flow_speed_variance
    )

    collision_mask = None
    tilt_angle = 0.0
    last_frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)

    while not stop.is_set():
        try:
            frame, collision, tilt = in_q.get_nowait()
            last_frame = frame
            collision_mask = collision
            tilt_angle = tilt
        except stdlib_queue.Empty:
            pass

        pf.update(collision_mask, tilt_angle)
        colors = cm.map(pf.speed, pf.turbulence)

        try:
            out_q.put_nowait(
                (last_frame, pf.pos.copy(), colors, collision_mask, tilt_angle)
            )
        except stdlib_queue.Full:
            pass

        time.sleep(1.0 / 240.0)


class SlipstreamOrchestrator:
    def __init__(self, config: Optional[SlipstreamConfig] = None):
        self.cfg = config or SlipstreamConfig()
        self._inf_q: Queue = mp.Queue(maxsize=self.cfg.inference_queue_size)
        self._rnd_q: Queue = mp.Queue(maxsize=self.cfg.render_queue_size)
        self._stop: Event = mp.Event()
        self._renderer = AeroRenderer(self.cfg)
        self._fps_acc = 0
        self._fps_ts = time.perf_counter()
        self._fps = 0.0

    def _launch_workers(self):
        self._inf_proc = Process(
            target=_run_inference,
            args=(self.cfg, self._inf_q, self._stop),
            daemon=True,
        )
        self._cfd_proc = Process(
            target=_run_cfd,
            args=(self.cfg, self._inf_q, self._rnd_q, self._stop),
            daemon=True,
        )
        self._inf_proc.start()
        self._cfd_proc.start()

    def _tick_fps(self):
        self._fps_acc += 1
        now = time.perf_counter()
        if now - self._fps_ts >= 1.0:
            self._fps = self._fps_acc / (now - self._fps_ts)
            self._fps_acc = 0
            self._fps_ts = now

    def run(self):
        self._launch_workers()

        blank = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        last_state = (
            blank,
            np.zeros((self.cfg.num_particles, 2), dtype=np.float32),
            np.zeros((self.cfg.num_particles, 3), dtype=np.uint8),
            None,
            0.0,
        )

        try:
            while True:
                try:
                    last_state = self._rnd_q.get_nowait()
                except stdlib_queue.Empty:
                    pass

                frame, pos, colors, collision, tilt = last_state
                self._tick_fps()

                if not self._renderer.render(
                    frame, pos, colors, collision, tilt, self._fps
                ):
                    break
        finally:
            self._stop.set()
            self._inf_proc.join(timeout=4)
            self._cfd_proc.join(timeout=4)
            self._renderer.destroy()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    SlipstreamOrchestrator(
        SlipstreamConfig(
            camera_index=0,
            width=1280,
            height=720,
            num_particles=7000,
            base_flow_speed=4.5,
            background_alpha=0.12,
            trail_decay=0.87,
        )
    ).run()