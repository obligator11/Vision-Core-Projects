import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import math
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple


class EngineState(Enum):
    LIVE = auto()
    TEMPORAL_SCRUB = auto()
    QUANTUM_SPLIT = auto()


@dataclass
class GestureState:
    engine_state: EngineState = EngineState.LIVE
    temporal_offset: int = 0
    anchor_angle: float = 0.0
    pinch_anchor: Optional[Tuple[int, int]] = None
    rect_origin: Optional[Tuple[int, int]] = None
    rect_terminus: Optional[Tuple[int, int]] = None
    drawing_active: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class FrameBuffer:
    def __init__(self, maxlen: int = 300):
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(np.ascontiguousarray(frame, dtype=np.uint8))

    def get_offset(self, offset: int) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            clamped = max(0, min(len(self._buffer) - 1, offset))
            idx = len(self._buffer) - 1 - clamped
            return np.ascontiguousarray(self._buffer[idx], dtype=np.uint8)

    def oldest(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            return np.ascontiguousarray(self._buffer[0], dtype=np.uint8)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class GestureProcessor:
    _PINCH_THRESHOLD = 0.055
    _ANGLE_SENSITIVITY = 1.4
    _MAX_OFFSET = 299

    def __init__(self, gesture_state: GestureState):
        self._state = gesture_state
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.65,
            model_complexity=1,
        )
        self._frame_queue: deque = deque(maxlen=2)
        self._frame_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def feed_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._frame_queue.append(np.ascontiguousarray(frame, dtype=np.uint8))

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.5)
        self._hands.close()

    def _process_loop(self) -> None:
        while self._running:
            frame = None
            with self._frame_lock:
                if self._frame_queue:
                    frame = self._frame_queue[-1]

            if frame is None:
                time.sleep(0.005)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self._hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                landmarks = results.multi_hand_landmarks[0].landmark
                self._analyze_gestures(landmarks, w, h)
            else:
                with self._state.lock:
                    if self._state.engine_state == EngineState.TEMPORAL_SCRUB:
                        self._state.engine_state = EngineState.LIVE
                        self._state.temporal_offset = 0
                        self._state.pinch_anchor = None
                    elif self._state.engine_state == EngineState.QUANTUM_SPLIT:
                        self._state.drawing_active = False

            time.sleep(0.014)

    @staticmethod
    def _euclidean_2d(lm1, lm2) -> float:
        return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

    @staticmethod
    def _wrist_angle(lm0, lm9) -> float:
        return math.degrees(math.atan2(lm9.y - lm0.y, lm9.x - lm0.x))

    @staticmethod
    def _finger_extended(tip, pip, mcp) -> bool:
        return tip.y < pip.y < mcp.y

    @staticmethod
    def _finger_folded(tip, pip) -> bool:
        return tip.y > pip.y

    def _analyze_gestures(self, landmarks, w: int, h: int) -> None:
        lm = landmarks
        pinch_dist = self._euclidean_2d(lm[4], lm[8])
        is_pinching = pinch_dist < self._PINCH_THRESHOLD

        index_ext = self._finger_extended(lm[8], lm[6], lm[5])
        middle_fold = self._finger_folded(lm[12], lm[10])
        ring_fold = self._finger_folded(lm[16], lm[14])
        pinky_fold = self._finger_folded(lm[20], lm[18])

        index_point_mode = index_ext and middle_fold and ring_fold and pinky_fold and not is_pinching

        if is_pinching:
            current_angle = self._wrist_angle(lm[0], lm[9])
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            with self._state.lock:
                if self._state.engine_state != EngineState.TEMPORAL_SCRUB:
                    self._state.anchor_angle = current_angle
                    self._state.engine_state = EngineState.TEMPORAL_SCRUB
                    self._state.pinch_anchor = (ix, iy)
                    self._state.rect_origin = None
                    self._state.rect_terminus = None
                    self._state.drawing_active = False

                delta = current_angle - self._state.anchor_angle
                if delta > 180:
                    delta -= 360
                elif delta < -180:
                    delta += 360

                raw = int(delta * self._ANGLE_SENSITIVITY)
                self._state.temporal_offset = max(0, min(self._MAX_OFFSET, raw))

        elif index_point_mode:
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            with self._state.lock:
                if not self._state.drawing_active:
                    self._state.rect_origin = (ix, iy)
                    self._state.drawing_active = True
                    self._state.engine_state = EngineState.QUANTUM_SPLIT

                self._state.rect_terminus = (ix, iy)
                self._state.engine_state = EngineState.QUANTUM_SPLIT

        else:
            with self._state.lock:
                if self._state.engine_state == EngineState.TEMPORAL_SCRUB:
                    self._state.engine_state = EngineState.LIVE
                    self._state.temporal_offset = 0
                    self._state.pinch_anchor = None
                elif self._state.engine_state == EngineState.QUANTUM_SPLIT:
                    self._state.drawing_active = False


class TemporalRenderer:
    _PALETTE = {
        "live":    (0, 255, 160),
        "scrub":   (0, 200, 255),
        "quantum": (255, 90, 30),
        "border":  (255, 255, 255),
    }
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _TIMELINE_H = 12

    def __init__(self, frame_buffer: FrameBuffer, gesture_state: GestureState):
        self._buffer = frame_buffer
        self._state = gesture_state

    def render(self, live_frame: np.ndarray) -> np.ndarray:
        with self._state.lock:
            state = self._state.engine_state
            offset = self._state.temporal_offset
            origin = self._state.rect_origin
            terminus = self._state.rect_terminus
            pinch_pt = self._state.pinch_anchor

        if state == EngineState.LIVE:
            return self._render_live(live_frame)
        elif state == EngineState.TEMPORAL_SCRUB:
            return self._render_scrub(live_frame, offset, pinch_pt)
        elif state == EngineState.QUANTUM_SPLIT:
            return self._render_quantum_split(live_frame, origin, terminus)

        return live_frame

    def _ensure_size(self, src: np.ndarray, h: int, w: int) -> np.ndarray:
        if src.shape[:2] != (h, w):
            return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
        return src

    def _render_live(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        self._draw_hud(output, "LIVE", self._PALETTE["live"], 0, len(self._buffer))
        return output

    def _render_scrub(self, frame: np.ndarray, offset: int, pinch_pt) -> np.ndarray:
        buf_len = len(self._buffer)
        if buf_len == 0:
            return frame

        past = self._buffer.get_offset(offset)
        if past is None:
            return frame

        h, w = frame.shape[:2]
        output = self._ensure_size(past, h, w).copy()

        alpha = np.clip(offset / max(buf_len - 1, 1), 0.0, 1.0)
        vignette = np.ones((h, w, 1), dtype=np.float32)
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        dist /= dist.max()
        vignette[..., 0] = 1.0 - alpha * 0.45 * dist
        output = np.clip(output.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

        if pinch_pt is not None:
            cv2.circle(output, pinch_pt, 10, self._PALETTE["scrub"], 2, cv2.LINE_AA)
            cv2.circle(output, pinch_pt, 3, self._PALETTE["scrub"], -1, cv2.LINE_AA)

        self._draw_hud(output, f"T-{offset}f  [{round(offset/30.0, 1)}s ago]",
                       self._PALETTE["scrub"], offset, buf_len)
        self._draw_timeline(output, offset, buf_len)
        return output

    def _render_quantum_split(self, live_frame: np.ndarray, origin, terminus) -> np.ndarray:
        output = live_frame.copy()
        h, w = output.shape[:2]

        if origin is None or terminus is None:
            self._draw_hud(output, "QUANTUM DRAW — DEFINE WINDOW",
                           self._PALETTE["quantum"], 0, len(self._buffer))
            return output

        x1 = max(0, min(origin[0], terminus[0]))
        y1 = max(0, min(origin[1], terminus[1]))
        x2 = min(w, max(origin[0], terminus[0]))
        y2 = min(h, max(origin[1], terminus[1]))

        if x2 - x1 < 8 or y2 - y1 < 8:
            self._draw_hud(output, "QUANTUM DRAW — DEFINE WINDOW",
                           self._PALETTE["quantum"], 0, len(self._buffer))
            return output

        past = self._buffer.oldest()
        if past is not None:
            past_rs = self._ensure_size(past, h, w)
            roi_past = past_rs[y1:y2, x1:x2]
            roi_live = output[y1:y2, x1:x2].astype(np.float32)
            blended = np.clip(roi_past.astype(np.float32) * 0.92 + roi_live * 0.08,
                              0, 255).astype(np.uint8)
            output[y1:y2, x1:x2] = blended

        cv2.rectangle(output, (x1, y1), (x2, y2), self._PALETTE["quantum"], 2, cv2.LINE_AA)
        cv2.rectangle(output, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                      self._PALETTE["border"], 1, cv2.LINE_AA)

        label_y = y1 - 8 if y1 > 20 else y2 + 18
        cv2.putText(output, "T-10s", (x1 + 4, label_y),
                    self._FONT, 0.45, self._PALETTE["quantum"], 1, cv2.LINE_AA)

        self._draw_hud(output, "QUANTUM SPLIT ACTIVE",
                       self._PALETTE["quantum"], 0, len(self._buffer))
        return output

    def _draw_hud(self, frame: np.ndarray, label: str, color: tuple,
                  offset: int, buf_len: int) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        cv2.putText(frame, "CHRONOS", (10, 28),
                    self._FONT, 0.85, color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (115, 28),
                    self._FONT, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

        status = f"BUF {buf_len}/300  |  OFF {offset}  |  {round(offset / 30.0, 1)}s"
        cv2.putText(frame, status, (10, 54),
                    self._FONT, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

        ctrl = "PINCH+TWIST=SCRUB  |  INDEX=DRAW  |  R=RESET  |  Q=QUIT"
        cv2.putText(frame, ctrl, (10, h - 10),
                    self._FONT, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    def _draw_timeline(self, frame: np.ndarray, offset: int, buf_len: int) -> None:
        if buf_len == 0:
            return
        h, w = frame.shape[:2]
        bar_x0, bar_x1 = 10, w - 10
        bar_y = h - 28
        bar_w = bar_x1 - bar_x0

        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + self._TIMELINE_H),
                      (40, 40, 40), -1)

        frac = 1.0 - (offset / max(buf_len - 1, 1))
        head_x = int(bar_x0 + frac * bar_w)
        cv2.rectangle(frame, (bar_x0, bar_y), (head_x, bar_y + self._TIMELINE_H),
                      self._PALETTE["scrub"], -1)
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + self._TIMELINE_H),
                      (80, 80, 80), 1)
        cv2.circle(frame, (head_x, bar_y + self._TIMELINE_H // 2), 6,
                   (255, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(frame, "PAST", (bar_x0, bar_y - 4),
                    self._FONT, 0.33, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, "NOW", (bar_x1 - 26, bar_y - 4),
                    self._FONT, 0.33, (100, 100, 100), 1, cv2.LINE_AA)


class ChronosEngine:
    _WINDOW_NAME = "CHRONOS  |  Temporal Reality Engine"

    def __init__(self, camera_index: int = 0, display_w: int = 1280, display_h: int = 720):
        self._camera_index = camera_index
        self._display_w = display_w
        self._display_h = display_h
        self._gesture_state = GestureState()
        self._frame_buffer = FrameBuffer(maxlen=300)
        self._gesture_processor = GestureProcessor(self._gesture_state)
        self._renderer = TemporalRenderer(self._frame_buffer, self._gesture_state)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps_timer = time.perf_counter()
        self._fps_counter = 0
        self._fps_display = 0.0

    def _init_capture(self) -> bool:
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_ANY)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._display_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._display_h)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return self._cap.isOpened()

    def _reset_quantum_state(self) -> None:
        with self._gesture_state.lock:
            self._gesture_state.engine_state = EngineState.LIVE
            self._gesture_state.rect_origin = None
            self._gesture_state.rect_terminus = None
            self._gesture_state.drawing_active = False
            self._gesture_state.temporal_offset = 0
            self._gesture_state.pinch_anchor = None

    def _update_fps(self) -> None:
        self._fps_counter += 1
        now = time.perf_counter()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps_display = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = now

    def _draw_fps(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{self._fps_display:.1f} FPS",
                    (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (80, 200, 80), 1, cv2.LINE_AA)

    def run(self) -> None:
        if not self._init_capture():
            raise RuntimeError(
                f"[CHRONOS] Fatal: cannot open camera index {self._camera_index}."
            )

        cv2.namedWindow(self._WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._WINDOW_NAME, self._display_w, self._display_h)

        self._gesture_processor.start()

        try:
            self._main_loop()
        finally:
            self._cleanup()

    def _main_loop(self) -> None:
        while True:
            ret, raw = self._cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            frame = cv2.flip(raw, 1)
            self._frame_buffer.push(frame)
            self._gesture_processor.feed_frame(frame)

            output = self._renderer.render(frame)
            self._update_fps()
            self._draw_fps(output)

            cv2.imshow(self._WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                self._reset_quantum_state()

    def _cleanup(self) -> None:
        self._gesture_processor.stop()
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    engine = ChronosEngine(camera_index=0, display_w=1280, display_h=720)
    engine.run()