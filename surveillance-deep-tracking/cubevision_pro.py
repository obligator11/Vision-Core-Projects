"""
CubeVision Pro
==============
Author  : Senior CV / Python Architect
Purpose : Webcam-based Rubik's Cube scanner and solver.
          Layer stack
          -----------
          1. OpenCV       – static 3x3 ROI grid, HSV colour extraction
          2. MediaPipe    – landmark-level hand / occlusion detection
          3. Stability    – frame-buffer deduplication before face capture
          4. Pygame mixer – async audio state machine + on-screen guidance
          5. Kociemba     – two-phase solver; validates & parses solution moves

Dependencies (install order matters):
    pip install opencv-python mediapipe==0.10.13 pygame kociemba numpy

Run:
    python cubevision_pro.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import sys
import time
import threading
import queue
import math
import textwrap
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import pygame
import kociemba

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (tweak without touching logic)
# ──────────────────────────────────────────────────────────────────────────────

# --- Camera ---
CAM_INDEX          = 0          # change to 1, 2 … if built-in cam isn't the cube cam
CAM_WIDTH          = 1280
CAM_HEIGHT         = 720
CAM_BACKEND        = cv2.CAP_DSHOW   # Windows; swap to cv2.CAP_V4L2 on Linux / 0 on macOS

# --- ROI grid (pixel coords are recalculated dynamically at start) ---
GRID_COLS          = 3
GRID_ROWS          = 3
GRID_CELL_SIZE     = 80          # px per cell
GRID_SAMPLE_RADIUS = 18          # px radius for the inner HSV sample circle
GRID_LINE_THICK    = 3

# --- Stability buffer ---
STABILITY_FRAMES   = 15          # consecutive frames the colours must match

# --- HSV colour mapping thresholds ---
# Each entry: (colour_label, kociemba_face_char, H_lo, H_hi, S_lo, V_lo)
# Hue is 0-179 in OpenCV, Sat/Val 0-255.
HSV_COLOUR_RANGES = [
    ("White",  "U", (  0, 179), ( 0,  70), (180, 255)),
    ("Yellow", "D", ( 20,  35), (100, 255), (100, 255)),
    ("Orange", "L", (  8,  20), (150, 255), (100, 255)),
    ("Red",    "F", (  0,   8), (150, 255), (100, 255)),   # low hue red
    ("Red",    "F", (165, 179), (150, 255), (100, 255)),   # wrap-around red
    ("Blue",   "B", (100, 130), (100, 255), ( 50, 255)),
    ("Green",  "R", ( 40,  80), (100, 255), ( 50, 255)),
]

# Kociemba face scan ORDER  (index → face char)
# Standard: U R F D L B
FACE_SCAN_ORDER    = ["U", "R", "F", "D", "L", "B"]
FACE_PROMPT_TEXT   = {
    "U": "Hold the WHITE centre face UP toward the camera",
    "R": "Rotate: GREEN centre face UP toward the camera",
    "F": "Rotate: RED centre face UP toward the camera",
    "D": "Rotate: YELLOW centre face UP toward the camera",
    "L": "Rotate: BLUE centre face UP toward the camera",
    "B": "Rotate: ORANGE centre face UP toward the camera",
}

# --- Pygame audio ---
AUDIO_SAMPLE_RATE  = 44_100
AUDIO_CHANNELS     = 2          # stereo – avoids the stereo crash on Windows
AUDIO_BUFFER       = 512

# Beep parameters (synthesised – no WAV file needed)
BEEP_SUCCESS_FREQ  = 880        # Hz
BEEP_SUCCESS_DUR   = 0.18       # seconds
BEEP_WARN_FREQ     = 300
BEEP_WARN_DUR      = 0.10
BEEP_SOLVE_FREQ    = 1200
BEEP_SOLVE_DUR     = 0.40

# --- UI colours (BGR for OpenCV) ---
COL_GRID_IDLE      = (200, 200, 200)
COL_GRID_LOCKED    = (  0, 220,   0)
COL_GRID_HAND      = (  0,   0, 220)
COL_WARN_BG        = (  0,   0, 180)
COL_TEXT_WHITE     = (255, 255, 255)
COL_TEXT_BLACK     = (  0,   0,   0)
COL_OVERLAY_BG     = ( 30,  30,  30)

# MediaPipe landmark indices we care about (fingertips + thumbs)
HAND_LANDMARK_CHECK = [
    mp.solutions.hands.HandLandmark.THUMB_TIP,
    mp.solutions.hands.HandLandmark.THUMB_IP,
    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
    mp.solutions.hands.HandLandmark.PINKY_TIP,
]

# ──────────────────────────────────────────────────────────────────────────────
# AUDIO ENGINE  (runs in a daemon thread so it never blocks OpenCV)
# ──────────────────────────────────────────────────────────────────────────────

class AudioEngine:
    """
    Thin wrapper around pygame.mixer.
    A background thread drains a queue of beep requests, synthesising
    pure-tone PCM on the fly – no WAV files required.
    """

    def __init__(self):
        pygame.mixer.pre_init(
            frequency=AUDIO_SAMPLE_RATE,
            size=-16,
            channels=AUDIO_CHANNELS,
            buffer=AUDIO_BUFFER,
        )
        pygame.mixer.init()
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---- public API ----------------------------------------------------------

    def beep_success(self):
        self._queue.put(("beep", BEEP_SUCCESS_FREQ, BEEP_SUCCESS_DUR, 0.6))

    def beep_warn(self):
        self._queue.put(("beep", BEEP_WARN_FREQ, BEEP_WARN_DUR, 0.4))

    def beep_solve(self):
        self._queue.put(("beep", BEEP_SOLVE_FREQ, BEEP_SOLVE_DUR, 0.8))

    def stop(self):
        self._queue.put(("quit",))

    # ---- internals -----------------------------------------------------------

    def _worker(self):
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item[0] == "quit":
                break
            if item[0] == "beep":
                _, freq, dur, vol = item
                self._play_tone(freq, dur, vol)

    @staticmethod
    def _play_tone(freq: float, dur: float, vol: float):
        """Synthesise a sine-wave tone and play it through pygame.mixer."""
        n_samples = int(AUDIO_SAMPLE_RATE * dur)
        t = np.linspace(0, dur, n_samples, endpoint=False)
        wave = (np.sin(2 * math.pi * freq * t) * 32767 * vol).astype(np.int16)
        # pygame needs a stereo array → duplicate the mono channel
        stereo = np.column_stack([wave, wave])
        sound = pygame.sndarray.make_sound(stereo)
        sound.play()
        # block only the worker thread until the sound finishes
        time.sleep(dur + 0.02)


# ──────────────────────────────────────────────────────────────────────────────
# HSV COLOUR CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def classify_hsv(h: int, s: int, v: int) -> tuple[str, str] | tuple[None, None]:
    """
    Return (colour_label, kociemba_char) for a pixel in HSV space,
    or (None, None) if no range matches.
    """
    for entry in HSV_COLOUR_RANGES:
        label, face, (h_lo, h_hi), (s_lo, _), (v_lo, _) = entry
        if h_lo <= h <= h_hi and s >= s_lo and v >= v_lo:
            return label, face
    return None, None


def sample_cell_color(hsv_frame: np.ndarray, cx: int, cy: int, radius: int) -> tuple[int, int, int]:
    """
    Return the mean HSV of a circular region centred at (cx, cy).
    """
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    mean = cv2.mean(hsv_frame, mask=mask)
    return int(mean[0]), int(mean[1]), int(mean[2])


# ──────────────────────────────────────────────────────────────────────────────
# GRID GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

def build_grid(frame_w: int, frame_h: int, cell: int = GRID_CELL_SIZE):
    """
    Return a list of 9 dicts, each describing one ROI cell.
    Cells are numbered 0-8 in reading order (top-left → bottom-right).
    """
    total_w = GRID_COLS * cell
    total_h = GRID_ROWS * cell
    ox = (frame_w - total_w) // 2   # origin x
    oy = (frame_h - total_h) // 2   # origin y

    cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x1 = ox + col * cell
            y1 = oy + row * cell
            x2 = x1 + cell
            y2 = y1 + cell
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cells.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy})

    # bounding box of the entire grid (for hand-overlap test)
    bbox = (ox, oy, ox + total_w, oy + total_h)
    return cells, bbox


# ──────────────────────────────────────────────────────────────────────────────
# HAND OCCLUSION DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

def hand_overlaps_grid(
    results,
    frame_w: int,
    frame_h: int,
    grid_bbox: tuple,
    margin: int = 10,
) -> bool:
    """
    Returns True if any fingertip/thumb landmark pixel falls inside
    the expanded grid bounding box (grid_bbox + margin).
    """
    if not results.multi_hand_landmarks:
        return False

    gx1, gy1, gx2, gy2 = grid_bbox
    # Expand slightly so even a partially-occluding finger triggers the warning
    gx1 -= margin; gy1 -= margin
    gx2 += margin; gy2 += margin

    for hand_lms in results.multi_hand_landmarks:
        for lm_idx in HAND_LANDMARK_CHECK:
            lm = hand_lms.landmark[lm_idx]
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            if gx1 <= px <= gx2 and gy1 <= py <= gy2:
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# STABILITY BUFFER
# ──────────────────────────────────────────────────────────────────────────────

class StabilityBuffer:
    """
    Accumulates per-frame colour readings and declares a 'lock' only when
    the last N frames all produced the same 9-colour sequence.
    """

    def __init__(self, required_frames: int = STABILITY_FRAMES):
        self.required = required_frames
        self._history: list[tuple] = []   # each entry is a tuple of 9 face-chars

    def push(self, colours: list[str]) -> bool:
        """
        Push a new reading (list of 9 kociemba chars, may contain None).
        Returns True the first time stability is achieved.
        """
        if None in colours:
            self.reset()
            return False

        reading = tuple(colours)
        self._history.append(reading)
        if len(self._history) > self.required:
            self._history.pop(0)

        if len(self._history) == self.required and len(set(self._history)) == 1:
            return True   # stable lock
        return False

    def reset(self):
        self._history.clear()

    @property
    def progress(self) -> float:
        """Fraction of required frames filled (0.0 – 1.0)."""
        if not self._history:
            return 0.0
        unique = len(set(self._history))
        if unique != 1:
            return 0.0
        return min(len(self._history) / self.required, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# KOCIEMBA INTERFACE
# ──────────────────────────────────────────────────────────────────────────────

def build_cube_string(face_data: dict[str, list[str]]) -> str | None:
    """
    Assemble the 54-char Kociemba string from the scanned face data.
    Format: U(9) R(9) F(9) D(9) L(9) B(9)

    Returns None and prints a diagnostic if the colour counts are wrong.
    """
    order = ["U", "R", "F", "D", "L", "B"]
    cube_str = "".join("".join(face_data[f]) for f in order)

    # Validate: exactly 9 of each face char
    counts = Counter(cube_str)
    errors = []
    for face in order:
        if counts[face] != 9:
            errors.append(f"  Face '{face}': found {counts[face]} stickers (need 9)")
    if errors:
        print("[Kociemba] Colour-count validation failed:")
        for e in errors:
            print(e)
        return None
    return cube_str


def solve_cube(cube_str: str) -> str | None:
    """
    Call kociemba.solve() and return the move sequence, or None on error.
    """
    try:
        solution = kociemba.solve(cube_str)
        return solution
    except Exception as exc:
        print(f"[Kociemba] Solver error: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# UI DRAWING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# BGR lookup for each face char (used to tint the cell colour patches)
FACE_BGR = {
    "U": (255, 255, 255),   # White
    "R": (  0, 200,   0),   # Green
    "F": (  0,   0, 220),   # Red
    "D": (  0, 230, 230),   # Yellow
    "L": (220,  80,   0),   # Blue  (display orange for L feels wrong – map correctly)
    "B": (  0, 100, 220),   # Orange
}
# Remap to RGB display colours that look right
FACE_DISPLAY_BGR = {
    "U": (230, 230, 230),   # White
    "R": (  0, 200,   0),   # Green
    "F": (  0,   0, 200),   # Red
    "D": (  0, 220, 220),   # Yellow
    "L": (210,  60,   0),   # Orange
    "B": (200,  30,   0),   # Blue (shown as blue-ish red, adjust if needed)
}


def draw_grid(
    frame: np.ndarray,
    cells: list,
    cell_colours: list[str | None],
    locked: bool,
    hand_detected: bool,
    stability_progress: float,
):
    """Render the 3x3 grid overlay with colour patches and a progress bar."""
    for idx, cell in enumerate(cells):
        face_char = cell_colours[idx] if cell_colours else None

        if hand_detected:
            border_col = COL_GRID_HAND
        elif locked:
            border_col = COL_GRID_LOCKED
        else:
            border_col = COL_GRID_IDLE

        # Draw cell rectangle
        cv2.rectangle(frame, (cell["x1"], cell["y1"]), (cell["x2"], cell["y2"]), border_col, GRID_LINE_THICK)

        # Fill a colour swatch in the inner quarter
        if face_char:
            swatch_x1 = cell["cx"] - GRID_SAMPLE_RADIUS
            swatch_y1 = cell["cy"] - GRID_SAMPLE_RADIUS
            swatch_x2 = cell["cx"] + GRID_SAMPLE_RADIUS
            swatch_y2 = cell["cy"] + GRID_SAMPLE_RADIUS
            bgr = FACE_DISPLAY_BGR.get(face_char, (128, 128, 128))
            cv2.rectangle(frame, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2), bgr, -1)
            cv2.rectangle(frame, (swatch_x1, swatch_y1), (swatch_x2, swatch_y2), (0, 0, 0), 1)

    # Stability progress bar below the grid
    if cells:
        bar_x1 = cells[0]["x1"]
        bar_x2 = cells[-1]["x2"]
        bar_y   = cells[-1]["y2"] + 10
        bar_h   = 8
        bar_fill = int((bar_x2 - bar_x1) * stability_progress)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + bar_h), COL_GRID_LOCKED, -1)


def put_text_shadow(frame, text, pos, font_scale=0.65, thickness=2, fg=COL_TEXT_WHITE, shadow=(0, 0, 0)):
    """Draw text with a 1px shadow for legibility over any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_DUPLEX, font_scale, shadow, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, fg, thickness, cv2.LINE_AA)


def draw_status_panel(
    frame: np.ndarray,
    face_index: int,
    scanned_faces: list[str],
    hand_detected: bool,
    solution: str | None,
    solver_error: str | None,
):
    """Render the left-side HUD panel."""
    h, w = frame.shape[:2]
    panel_w = 280

    # Semi-transparent dark strip on the left
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), COL_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y_cursor = 30

    put_text_shadow(frame, "CubeVision Pro", (10, y_cursor), 0.75, 2, (80, 200, 255))
    y_cursor += 35

    cv2.line(frame, (10, y_cursor), (panel_w - 10, y_cursor), (80, 80, 80), 1)
    y_cursor += 20

    # Face scan progress
    put_text_shadow(frame, "Scan progress:", (10, y_cursor), 0.55, 1)
    y_cursor += 22
    for i, fc in enumerate(FACE_SCAN_ORDER):
        done = fc in scanned_faces
        col = COL_GRID_LOCKED if done else (100, 100, 100)
        marker = "[x]" if done else "[ ]"
        label = f"  {marker} Face {fc}"
        put_text_shadow(frame, label, (10, y_cursor), 0.50, 1, col)
        y_cursor += 20

    y_cursor += 10
    cv2.line(frame, (10, y_cursor), (panel_w - 10, y_cursor), (80, 80, 80), 1)
    y_cursor += 15

    # Current instruction
    if solution is None and solver_error is None and face_index < len(FACE_SCAN_ORDER):
        current_face = FACE_SCAN_ORDER[face_index]
        prompt = FACE_PROMPT_TEXT[current_face]
        # Word-wrap to ~32 chars
        wrapped = textwrap.wrap(prompt, 30)
        put_text_shadow(frame, "Instruction:", (10, y_cursor), 0.52, 1, (200, 200, 100))
        y_cursor += 20
        for line in wrapped:
            put_text_shadow(frame, line, (10, y_cursor), 0.48, 1)
            y_cursor += 18

    # Hand-in-grid warning banner
    if hand_detected:
        cv2.rectangle(frame, (0, h - 50), (w, h), COL_WARN_BG, -1)
        put_text_shadow(
            frame,
            "  Fingers detected in grid! — Move hand away",
            (10, h - 18),
            0.60,
            2,
            COL_TEXT_WHITE,
        )

    # Solution display
    if solution:
        put_text_shadow(frame, "SOLUTION:", (10, y_cursor), 0.60, 2, (80, 255, 80))
        y_cursor += 24
        # Wrap solution moves to panel width
        moves = solution.split()
        lines, line = [], []
        for m in moves:
            line.append(m)
            if len(" ".join(line)) > 22:
                lines.append(" ".join(line[:-1]))
                line = [m]
        if line:
            lines.append(" ".join(line))
        for ln in lines:
            put_text_shadow(frame, ln, (10, y_cursor), 0.55, 1, (180, 255, 180))
            y_cursor += 18

    if solver_error:
        put_text_shadow(frame, "SOLVER ERROR:", (10, y_cursor), 0.55, 2, (0, 80, 220))
        y_cursor += 22
        for ln in textwrap.wrap(solver_error, 28):
            put_text_shadow(frame, ln, (10, y_cursor), 0.46, 1, (100, 140, 255))
            y_cursor += 17


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Camera ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # reduce latency

    if not cap.isOpened():
        sys.exit("[CubeVision] ERROR: Cannot open webcam. Check CAM_INDEX / CAM_BACKEND.")

    # Read one frame to get actual resolution (driver may not honour the set)
    ret, probe = cap.read()
    if not ret:
        sys.exit("[CubeVision] ERROR: Cannot read from webcam.")
    frame_h, frame_w = probe.shape[:2]
    print(f"[CubeVision] Camera opened at {frame_w}x{frame_h}")

    # ── 2. Grid geometry ──────────────────────────────────────────────────────
    cells, grid_bbox = build_grid(frame_w, frame_h)

    # ── 3. MediaPipe Hands ────────────────────────────────────────────────────
    mp_hands    = mp.solutions.hands
    mp_draw     = mp.solutions.drawing_utils
    hands_model = mp_hands.Hands(
        static_image_mode        = False,
        max_num_hands            = 2,
        min_detection_confidence = 0.6,
        min_tracking_confidence  = 0.5,
    )

    # ── 4. Audio engine ───────────────────────────────────────────────────────
    audio = AudioEngine()

    # ── 5. Application state ──────────────────────────────────────────────────
    face_index     = 0                # which face we are currently scanning
    scanned_faces  = []               # list of already-captured face chars
    face_data: dict[str, list[str]] = {}  # face_char → [9 kociemba chars]
    stability_buf  = StabilityBuffer()
    solution: str | None = None
    solver_error: str | None = None

    # Small cooldown so we don't immediately re-capture the same face
    cooldown_until = 0.0
    COOLDOWN_SEC   = 1.8

    print("[CubeVision] Starting — press Q to quit, R to reset.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CubeVision] WARNING: Dropped frame — skipping.")
            continue

        frame = cv2.flip(frame, 1)   # mirror so user sees a natural reflection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── MediaPipe hand detection ─────────────────────────────────────────
        rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands_model.process(rgb_for_mp)
        hand_detected = hand_overlaps_grid(hand_results, frame_w, frame_h, grid_bbox)

        # Draw faint hand skeleton for UX (doesn't affect logic)
        if hand_results.multi_hand_landmarks:
            for lms in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 180), thickness=1, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 180, 120), thickness=1),
                )

        # ── ROI colour extraction ────────────────────────────────────────────
        cell_colours: list[str | None] = []
        for cell in cells:
            h_val, s_val, v_val = sample_cell_color(hsv, cell["cx"], cell["cy"], GRID_SAMPLE_RADIUS)
            _, face_char = classify_hsv(h_val, s_val, v_val)
            cell_colours.append(face_char)

        # ── Stability buffer logic ───────────────────────────────────────────
        now = time.time()
        locked = False

        if (
            solution is None          # not solved yet
            and not hand_detected     # hand out of grid
            and face_index < len(FACE_SCAN_ORDER)  # still faces to scan
            and now > cooldown_until  # past the post-capture cooldown
        ):
            locked = stability_buf.push(cell_colours)

            if locked:
                current_face = FACE_SCAN_ORDER[face_index]
                # Reject if this face was already captured (shouldn't happen, but defensive)
                if current_face not in scanned_faces:
                    face_data[current_face] = list(stability_buf._history[-1])
                    scanned_faces.append(current_face)
                    print(f"[CubeVision] Captured face {current_face}: {face_data[current_face]}")
                    audio.beep_success()
                    face_index += 1
                stability_buf.reset()
                cooldown_until = now + COOLDOWN_SEC

                # ── All 6 faces captured → solve ───────────────────────────
                if face_index == len(FACE_SCAN_ORDER) and solution is None:
                    cube_str = build_cube_string(face_data)
                    if cube_str:
                        print(f"[CubeVision] Kociemba input: {cube_str}")
                        solution = solve_cube(cube_str)
                        if solution:
                            print(f"[CubeVision] Solution: {solution}")
                            audio.beep_solve()
                        else:
                            solver_error = "Invalid cube state — re-scan?"
                            audio.beep_warn()
                    else:
                        solver_error = "Colour validation failed — re-scan?"
                        audio.beep_warn()
        else:
            if hand_detected:
                stability_buf.reset()

        # ── Draw the grid overlay ────────────────────────────────────────────
        draw_grid(
            frame,
            cells,
            cell_colours,
            locked,
            hand_detected,
            stability_buf.progress,
        )

        # ── Draw the status panel ────────────────────────────────────────────
        draw_status_panel(
            frame,
            face_index,
            scanned_faces,
            hand_detected,
            solution,
            solver_error,
        )

        # ── Top-centre instruction overlay ───────────────────────────────────
        if solution:
            put_text_shadow(
                frame,
                "SOLVED! Follow the moves on the left panel.",
                (frame_w // 2 - 230, 35),
                0.70,
                2,
                (80, 255, 80),
            )
        elif solver_error:
            put_text_shadow(
                frame,
                "ERROR — Press R to reset and re-scan.",
                (frame_w // 2 - 220, 35),
                0.68,
                2,
                (0, 80, 220),
            )
        elif face_index < len(FACE_SCAN_ORDER):
            remaining = len(FACE_SCAN_ORDER) - face_index
            put_text_shadow(
                frame,
                f"Face {face_index + 1}/6  —  {FACE_SCAN_ORDER[face_index]}",
                (frame_w // 2 - 80, 35),
                0.72,
                2,
                (255, 200, 80),
            )

        # ── Show frame ───────────────────────────────────────────────────────
        cv2.imshow("CubeVision Pro", frame)

        # ── Keyboard handling ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[CubeVision] Quitting.")
            break
        elif key == ord("r"):
            # Full reset
            face_index    = 0
            scanned_faces.clear()
            face_data.clear()
            stability_buf.reset()
            solution      = None
            solver_error  = None
            cooldown_until = 0.0
            print("[CubeVision] State reset.")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()
    audio.stop()
    pygame.mixer.quit()
    print("[CubeVision] Shutdown complete.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()