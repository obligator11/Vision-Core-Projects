import cv2
import numpy as np
import math
import time
from collections import deque, defaultdict, Counter
import threading
from ultralytics import YOLO
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContextConfig:
    stream_source: str  = "video.mp4"
    yolo_model: str     = "yolo11x.pt"

    # Lateral walking threshold: body-heights/second (distance-independent).
    walk_threshold: float = 0.20

    # Depth walking: minimum fraction of bbox height that must change
    # consistently across recent frames to count as toward/away motion.
    # 0.03 = 3 % of body height per detection frame — catches walkers at
    # up to ~15 m on a typical CCTV lens.
    depth_walk_min_ratio: float = 0.03

    # Minimum fraction of consecutive height-diffs that must share the
    # same sign for depth motion to be considered consistent (not jitter).
    depth_walk_consistency: float = 0.65   # 65 % agreement across diffs

    sit_ar_max: float        = 1.40
    stand_ar_min: float      = 1.80
    sit_height_ratio: float  = 0.82
    stand_height_ratio: float = 0.90

    person_cls: int   = 0
    carry_classes: list  = field(default_factory=lambda: [24, 25, 26, 28])
    consume_classes: list= field(default_factory=lambda: [39, 41, 46, 47, 48, 54, 55])
    phone_classes: list  = field(default_factory=lambda: [67])
    bike_classes: list   = field(default_factory=lambda: [1, 3])   # bicycle, motorcycle
    confidence: float    = 0.45

    # Persons whose bbox height is below this are too far — skip entirely.
    min_person_height: int = 60   # pixels

    buf_size: int   = 20
    ar_window: int  = 8
    act_window: int = 8


# ─────────────────────────────────────────────────────────────────────────────
#  PER-TRACK STATE
# ─────────────────────────────────────────────────────────────────────────────

class TrackState:
    def __init__(self, cfg: ContextConfig):
        self._pos        = deque(maxlen=cfg.buf_size)
        self._ar         = deque(maxlen=cfg.ar_window)
        self._acts       = deque(maxlen=cfg.act_window)
        self._height_max = 0.0
        self._height_avg = deque(maxlen=cfg.ar_window)
        self._posture    = "Standing"

    def update(self, center: tuple, aspect: float, height: float, t: float):
        self._pos.append((center, height, t))
        self._ar.append(aspect)
        self._height_avg.append(height)
        if height > self._height_max:
            self._height_max = height
        self._height_max *= 0.998

    # ── Lateral velocity ──────────────────────────────────────────────────────

    def lateral_velocity(self) -> float:
        """
        2D pixel speed over a short recent window (6 positions ≈ 0.4 s).
        Short window so a person who just started moving is caught quickly,
        not dragged down by their earlier standing-still history.
        """
        if len(self._pos) < 4:
            return 0.0
        recent = list(self._pos)[-6:]
        (x0, y0), h0, t0 = recent[0]
        (x1, y1), h1, t1 = recent[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt

    # ── Depth motion (toward / away from camera) ──────────────────────────────

    def is_depth_walking(self, cfg: ContextConfig) -> bool:
        """
        True when the bounding-box height is changing consistently over recent
        frames, indicating the person is walking toward or away from the camera.

        Two conditions must BOTH be true:
          1. Total height change / starting height > depth_walk_min_ratio
             (filters out tiny jitter even if it's consistent in sign)
          2. At least depth_walk_consistency fraction of consecutive frame-diffs
             share the same sign (growing OR shrinking — not random noise)

        Using both guards together is much more reliable than any single metric.
        Linear regression was tried but is sensitive to outlier frames from
        the tracker; the consistency ratio is more robust in practice.
        """
        recent = list(self._pos)[-8:]
        if len(recent) < 5:
            return False

        heights = [h for (_, h, _) in recent]
        h_start = heights[0]
        if h_start <= 0:
            return False

        # Guard 1: meaningful total change
        total_ratio = abs(heights[-1] - h_start) / h_start
        if total_ratio < cfg.depth_walk_min_ratio:
            return False

        # Guard 2: consistent direction (deadband ±0.5 px to ignore sub-pixel noise)
        diffs = [heights[i + 1] - heights[i] for i in range(len(heights) - 1)]
        pos = sum(1 for d in diffs if d >  0.5)
        neg = sum(1 for d in diffs if d < -0.5)
        dominant = max(pos, neg)
        if dominant < len(diffs) * cfg.depth_walk_consistency:
            return False

        return True

    # ── Posture ───────────────────────────────────────────────────────────────

    def posture(self, cfg: ContextConfig) -> str:
        ar    = float(np.mean(self._ar))         if self._ar         else 2.0
        avg_h = float(np.mean(self._height_avg)) if self._height_avg else 0.0

        sit_votes = stand_votes = 0

        if ar < cfg.sit_ar_max:
            sit_votes += 1
        elif ar > cfg.stand_ar_min:
            stand_votes += 2

        if self._height_max > 40:
            ratio = avg_h / self._height_max
            if ratio < cfg.sit_height_ratio and ar < 1.65:
                sit_votes += 1
            elif ratio > cfg.stand_height_ratio:
                stand_votes += 1

        if sit_votes >= 2:
            self._posture = "Sitting"
        elif sit_votes >= 1 and stand_votes == 0:
            self._posture = "Sitting"
        elif stand_votes >= 2:
            self._posture = "Standing"
        elif stand_votes >= 1 and sit_votes == 0:
            self._posture = "Standing"
        # else: keep previous — avoids flip-flopping in ambiguous frames

        return self._posture

    # ── Smoothing ─────────────────────────────────────────────────────────────

    def smooth(self, raw: str) -> str:
        self._acts.append(raw)
        recent = list(self._acts)
        # Sticky priority: object-interaction labels win if seen recently
        for a in reversed(recent[-3:]):
            if any(kw in a for kw in ("Eating", "Drinking", "Carrying", "Phone")):
                return a
        return Counter(recent).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION THREAD
# ─────────────────────────────────────────────────────────────────────────────

def _det_worker(model_path, classes, conf, pop_frame, push_result, stop_evt):
    model = YOLO(model_path)
    while not stop_evt.is_set():
        item = pop_frame()
        if item is None:
            time.sleep(0.002)
            continue
        frame, t_frame = item
        res = model.track(frame, persist=True, classes=classes, conf=conf, verbose=False)
        t_det = time.time()
        tracks = []
        if res[0].boxes and res[0].boxes.id is not None:
            coords  = res[0].boxes.xyxy.cpu().numpy()
            ids     = res[0].boxes.id.cpu().numpy()
            cls_ids = res[0].boxes.cls.cpu().numpy()
            names   = res[0].names
            for c, i, cl in zip(coords, ids, cls_ids):
                tracks.append((c, int(i), int(cl), names[int(cl)]))
        push_result(tracks, t_det)


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────────────────────────

_COLOURS = {
    "Walking":  (0,   210, 255),
    "Standing": (0,   230,  80),
    "Sitting":  (0,   140, 255),
    "Eating":   (255,  80, 200),
    "Carrying": (180,  90, 255),
    "Phone":    (0,   255, 200),
    "Cycling":  (255, 160,   0),
}

def _colour(action: str) -> tuple:
    for k, v in _COLOURS.items():
        if k in action:
            return v
    return (0, 255, 255)

def _draw_label(frame, text, x, y, col):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 5
    cv2.rectangle(frame, (x, max(y - th - bl - pad*2, 0)), (x + tw + pad*2, y), col, cv2.FILLED)
    cv2.putText(frame, text, (x + pad, y - bl - 1), font, scale, (0,0,0), thick, cv2.LINE_AA)

def _draw_vel_bar(frame, x1, x2, y2, vel, col):
    w = x2 - x1
    fill = min(int(vel / 300.0 * w), w)
    cv2.rectangle(frame, (x1, y2+3), (x2,       y2+8), (40,40,40), cv2.FILLED)
    cv2.rectangle(frame, (x1, y2+3), (x1+fill,  y2+8), col,        cv2.FILLED)

def _draw_sitting_mark(frame, x1, x2, y1, y2, col):
    mid, seat_y = (x1+x2)//2, y2-12
    half_w = (x2-x1)//4
    cv2.line(frame, (mid-half_w, seat_y), (mid+half_w, seat_y), col, 3)
    cv2.line(frame, (mid-half_w, y2-28),  (mid-half_w, seat_y), col, 3)
    cv2.line(frame, (mid-half_w+4, seat_y), (mid-half_w+4, y2-4), col, 2)
    cv2.line(frame, (mid+half_w-4, seat_y), (mid+half_w-4, y2-4), col, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SentinelOmniEngine:
    def __init__(self, cfg: ContextConfig):
        self.cfg   = cfg
        self.states = defaultdict(lambda: TrackState(cfg))
        self._frame_slot = None
        self._track_slot = ([], 0.0)
        self._new_tracks = False
        self._f_lock = threading.Lock()
        self._t_lock = threading.Lock()
        self._stop   = threading.Event()

        all_cls = ([cfg.person_cls] + cfg.carry_classes + cfg.consume_classes
                   + cfg.phone_classes + cfg.bike_classes)
        self._thread = threading.Thread(
            target=_det_worker,
            args=(cfg.yolo_model, all_cls, cfg.confidence,
                  self._pop_frame, self._push_tracks, self._stop),
            daemon=True,
        )
        self._thread.start()

    # ── Thread plumbing ───────────────────────────────────────────────────────

    def _push_frame(self, frame, t):
        with self._f_lock:
            self._frame_slot = (frame, t)

    def _pop_frame(self):
        with self._f_lock:
            item, self._frame_slot = self._frame_slot, None
            return item

    def _push_tracks(self, tracks, t_det):
        with self._t_lock:
            self._track_slot = (tracks, t_det)
            self._new_tracks = True

    def _read_tracks(self):
        with self._t_lock:
            tracks, t = self._track_slot
            fresh, self._new_tracks = self._new_tracks, False
            return list(tracks), t, fresh

    # ── Object-interaction helpers ────────────────────────────────────────────

    def _is_cycling(self, p_box, objects) -> bool:
        x1, y1, x2, y2 = map(int, p_box)
        pw, ph = x2-x1, y2-y1
        sx1, sx2 = x1 - pw*0.4, x2 + pw*0.4
        sy1, sy2 = y1, y2 + ph*0.6
        for o_box, _, o_cls, _ in objects:
            if o_cls not in self.cfg.bike_classes:
                continue
            ox1, oy1, ox2, oy2 = map(int, o_box)
            ocx, ocy = (ox1+ox2)/2, (oy1+oy2)/2
            if sx1 < ocx < sx2 and sy1 < ocy < sy2:
                return True
        return False

    def _object_action(self, p_box, objects):
        x1, y1, x2, y2 = map(int, p_box)
        pw, ph = x2-x1, y2-y1
        ix1, ix2 = x1 - pw*0.35, x2 + pw*0.35
        iy1, iy2 = y1 - ph*0.20, y2
        head_y  = y1 + ph*0.35
        hand_y2 = y2 - ph*0.20
        pcx, pcy = x1+pw/2, y1+ph/2

        best_phone = best_eat = best_hold = None
        d_phone = d_eat = d_hold = float("inf")

        for o_box, _, o_cls, o_name in objects:
            ox1, oy1, ox2, oy2 = map(int, o_box)
            ocx, ocy = (ox1+ox2)/2, (oy1+oy2)/2
            if not (ix1 < ocx < ix2 and iy1 < ocy < iy2):
                continue
            dist = math.hypot(ocx-pcx, ocy-pcy)
            if o_cls in self.cfg.phone_classes and dist < d_phone:
                d_phone = dist; best_phone = (o_name, ocy, head_y, hand_y2)
            elif o_cls in self.cfg.consume_classes and dist < d_eat:
                d_eat = dist;   best_eat   = (o_name, ocy, head_y)
            elif o_cls in self.cfg.carry_classes and dist < d_hold:
                d_hold = dist;  best_hold  = o_name

        if best_phone:
            name, ocy, hy, hy2 = best_phone
            if ocy < hy:    return "On Phone (calling)"
            elif ocy < hy2: return "Using Phone"
            else:           return "Holding Phone"
        if best_eat:
            name, ocy, hy = best_eat
            return f"{'Eating/Drinking' if ocy < hy else 'Carrying'} ({name})"
        if best_hold:
            return f"Carrying ({best_hold})"
        return None


    def run(self):
        cap   = cv2.VideoCapture(self.cfg.stream_source)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))
        cv2.namedWindow("Sentinel: Omni-Tracker", cv2.WINDOW_NORMAL)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self._push_frame(frame.copy(), time.time())
            tracks, t_det, is_fresh = self._read_tracks()

            persons = [t for t in tracks if t[2] == self.cfg.person_cls]
            objects = [t for t in tracks if t[2] != self.cfg.person_cls]

            for p_box, p_id, _, _ in persons:
                x1, y1, x2, y2 = map(int, p_box)
                pw, ph = x2-x1, y2-y1
                if pw <= 0:
                    continue
                if ph < self.cfg.min_person_height:
                    continue   # too far — unreliable, skip

                state  = self.states[p_id]
                cx, cy = (x1+x2)//2, (y1+y2)//2

                if is_fresh:
                    state.update((cx, cy), ph/pw, ph, t_det)

                # ── Motion classification ─────────────────────────────────────
                lat_vel   = state.lateral_velocity()
                norm_vel  = lat_vel / max(ph, 1)          # body-heights / sec
                depth_walk = state.is_depth_walking(self.cfg)
                is_walking = norm_vel > self.cfg.walk_threshold or depth_walk

                # ── Action decision (single clean if/elif/else) ───────────────
                if self._is_cycling(p_box, objects):
                    raw = "Cycling"
                elif is_walking:
                    raw = "Walking"
                else:
                    obj = self._object_action(p_box, objects)
                    raw = obj if obj else state.posture(self.cfg)

                action = state.smooth(raw)
                col    = _colour(action)

                cv2.rectangle(frame, (x1,y1), (x2,y2), col, 3 if "Sitting" in action else 2)
                if "Sitting" in action:
                    _draw_sitting_mark(frame, x1, x2, y1, y2, col)
                _draw_label(frame, f"ID:{p_id}  {action}", x1, y1, col)
                _draw_vel_bar(frame, x1, x2, y2, lat_vel, col)

            # Draw detected objects
            for o_box, _, _, o_name in objects:
                ox1, oy1, ox2, oy2 = map(int, o_box)
                cv2.rectangle(frame, (ox1,oy1), (ox2,oy2), (200,80,255), 1)
                cv2.putText(frame, o_name, (ox1, oy1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,80,255), 1, cv2.LINE_AA)

            cv2.imshow("Sentinel: Omni-Tracker", frame)
            if cv2.waitKey(delay) & 0xFF == 27:
                break

        self._stop.set()
        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = SentinelOmniEngine(ContextConfig())
    engine.run()