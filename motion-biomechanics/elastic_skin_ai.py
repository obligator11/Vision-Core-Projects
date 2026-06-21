"""
=====================================================================
 ELASTIC SKIN AI
=====================================================================
Real-time webcam app that lets you "pinch and drag" your face on
screen and have the skin actually stretch like a rubber tendon
reaching out to your fingers -- not just slide around as a blob --
with a spring-driven elastic snap-back when you let go.

Pipeline:
    OpenCV (webcam I/O + image warping)
    MediaPipe Face Mesh (468 landmarks, largest face is tracked)
    MediaPipe Hands (thumb+index pinch detection)
    NumPy (tapered tendon warp + spring physics)
    Pygame (procedurally generated stretch / snap sound effects --
             no external audio assets are used anywhere)

------------------------------------------------------------------
HOW THE STRETCH WORKS (read this if you want to tune the feel):

Instead of nudging a circular gaussian "bump" toward the finger
(which just looks like a blob being dragged), the warp builds a
tapered capsule/tendon shape that runs from the anchor point "A"
(the spot on your face you pinched) to a spring-driven virtual tip
"VT" that chases your pinching fingers. Pixels inside that capsule
are re-sampled from a *small* patch of skin near A -- so the same
bit of original skin texture gets stretched/smeared across the full
length of the tendon, getting visibly thinner near the tip, exactly
like pulling taffy or rubber. When you release the pinch, VT is
sprung back to A with a damped spring (slight overshoot = jiggle),
so the tendon visibly retracts and snaps back into the face.

INSTALL (Windows / conda, matches a known-good combo):
    conda create -n cv_env python=3.11 -y
    conda activate cv_env
    pip install opencv-python numpy pygame
    pip install mediapipe==0.10.13
        (IMPORTANT: mediapipe>=0.10.33 removed the legacy
         `mp.solutions` API that this script relies on. 0.10.13 is
         a safe, stable pin.)

RUN:
    python elastic_skin_ai.py

CONTROLS:
    Pinch (thumb tip + index tip) near your forehead / cheeks /
    jawline and drag your hand -> a tapered elastic tendon of skin
    stretches out and follows your fingers.
    Release the pinch -> it springs back and snaps into place.

    E       toggle elastic warping on/off
    R       reset face instantly (kill all current deformation)
    + / =   increase elasticity strength (longer max stretch)
    -       decrease elasticity strength
    C       toggle cartoon outline overlay
    Q / ESC quit
=====================================================================
"""

import sys
import time
import math

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe is required. Install with: pip install mediapipe==0.10.13")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("Pygame is required. Install with: pip install pygame")
    sys.exit(1)


# =====================================================================
# CONFIG
# =====================================================================

WINDOW_NAME = "Elastic Skin AI"
PROC_MAX_WIDTH = 960          # frames are downscaled to this width for speed

R_BASE_FRAC = 0.16             # tendon base radius, fraction of face size
R_TIP_FRAC = 0.34              # tendon tip radius, fraction of base radius
AXIAL_SOURCE_FRAC = 0.82       # how much of the base radius worth of source
                                # texture gets smeared along the tendon length
                                # (lower = more "stretched thin" look)

MAX_DRAG_FRAC = 0.95           # drag distance limiter, fraction of face size
                                # (multiplied by elasticity strength)

SPRING_K = 190.0               # spring stiffness pulling the tip back to anchor
SPRING_DAMPING = 15.0          # damping (controls overshoot / jiggle on release)
DRAG_LERP = 14.0               # how fast the tip chases the finger while held

PINCH_RATIO_ON = 0.42           # thumb-index distance / hand-scale -> pinch START
PINCH_RATIO_OFF = 0.55          # thumb-index distance / hand-scale -> pinch END (hysteresis)

MOTION_BLUR_VEL_SCALE = 900.0   # px/sec that maps to full blur strength
MOTION_BLUR_MAX = 0.5

ELASTICITY_MIN = 0.2
ELASTICITY_MAX = 3.0
ELASTICITY_STEP = 0.1

SETTLE_DIST = 1.5               # px: below this + low velocity -> fully snapped home
SETTLE_VEL = 2.0

# Facial landmark groups (MediaPipe Face Mesh, 468-point topology)
FOREHEAD_IDX = [10, 109, 67, 103, 54, 21, 251, 284, 332, 297, 338, 9, 151, 8]
LEFT_CHEEK_IDX = [50, 101, 119, 100, 36, 205, 187, 207, 216, 192, 123]
RIGHT_CHEEK_IDX = [280, 330, 348, 329, 266, 425, 411, 427, 436, 416, 352]
JAWLINE_IDX = [152, 148, 176, 149, 150, 136, 172, 58, 132,
               377, 400, 378, 379, 365, 397, 288, 361]

GRABBABLE_IDX = sorted(set(FOREHEAD_IDX + LEFT_CHEEK_IDX + RIGHT_CHEEK_IDX + JAWLINE_IDX))


# =====================================================================
# SOUND GENERATION  (synthesized in NumPy -- zero external assets)
# =====================================================================

SAMPLE_RATE = 44100


def _to_pygame_stereo(wave: np.ndarray) -> np.ndarray:
    """Convert a mono float wave in [-1, 1] into an int16 stereo buffer."""
    wave = np.clip(wave, -1.0, 1.0)
    stereo = np.column_stack([wave, wave])
    return np.ascontiguousarray((stereo * 32767.0).astype(np.int16))


def generate_stretch_loop(duration=1.0, sr=SAMPLE_RATE) -> np.ndarray:
    """A sustained, slightly squeaky 'rubber friction' loop. Loop-safe (fades
    in/out at identical levels so the seam is inaudible)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base = 0.22 * np.sin(2 * np.pi * 165 * t)
    squeak = 0.13 * np.sin(2 * np.pi * (650 + 180 * np.sin(2 * np.pi * 3.2 * t)) * t)
    rng = np.random.default_rng(42)
    noise = 0.05 * rng.uniform(-1, 1, len(t))
    noise = np.convolve(noise, np.ones(7) / 7.0, mode="same")  # soften (cheap low-pass)
    wave = base + squeak + noise

    fade = max(1, int(sr * 0.06))
    env = np.ones_like(wave)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    wave *= env
    return _to_pygame_stereo(wave)


def generate_snap_sound(duration=0.18, sr=SAMPLE_RATE) -> np.ndarray:
    """A short percussive 'snap' for when the pinch is released."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    inst_freq = 720 * np.exp(-14 * t) + 90
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    wave = np.sin(phase)
    env = np.exp(-16 * t)
    wave *= env * 1.15
    return _to_pygame_stereo(wave)


# =====================================================================
# SMALL MATH / GEOMETRY HELPERS
# =====================================================================

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =====================================================================
# FACE / HAND TRACKING HELPERS
# =====================================================================

def get_largest_face(face_results, w, h):
    """Return (landmarks_px (Nx2 float32), bbox) for the largest detected
    face, or (None, None) if no face is present."""
    if not face_results.multi_face_landmarks:
        return None, None

    best_pts, best_area, best_bbox = None, -1, None
    for face_landmarks in face_results.multi_face_landmarks:
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark],
            dtype=np.float32,
        )
        xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
        xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
        area = (xmax - xmin) * (ymax - ymin)
        if area > best_area:
            best_area = area
            best_pts = pts
            best_bbox = (xmin, ymin, xmax, ymax)

    return best_pts, best_bbox


def get_pinch_state(hand_results, w, h):
    """Returns (is_hand_found, pinch_point, pinch_ratio) for the first
    detected hand, or (False, None, None) if no hand is present."""
    if not hand_results.multi_hand_landmarks:
        return False, None, None

    hand = hand_results.multi_hand_landmarks[0].landmark
    thumb_tip = np.array([hand[4].x * w, hand[4].y * h])
    index_tip = np.array([hand[8].x * w, hand[8].y * h])
    wrist = np.array([hand[0].x * w, hand[0].y * h])
    middle_mcp = np.array([hand[9].x * w, hand[9].y * h])

    hand_scale = max(np.linalg.norm(wrist - middle_mcp), 1e-3)
    pinch_dist = np.linalg.norm(thumb_tip - index_tip)
    ratio = pinch_dist / hand_scale
    pinch_point = tuple((thumb_tip + index_tip) / 2.0)

    return True, pinch_point, ratio


# =====================================================================
# ELASTIC DEFORMATION ENGINE -- tapered "tendon" stretch
# =====================================================================

class ElasticSkin:
    """Owns a single anchor->tip spring and renders it each frame as a
    tapered, smeared-texture tendon connecting the grabbed point on the
    face to the (spring-damped) position of the pinching fingers."""

    def __init__(self):
        self.anchor_idx = None      # grabbed landmark index (None == idle)
        self.is_dragging = False
        self.active = False         # True while a tendon is visible/animating
        self.A = None                # anchor pixel pos (np.float32[2], live-tracked)
        self.VT = np.zeros(2, dtype=np.float32)      # virtual tip position
        self.VT_vel = np.zeros(2, dtype=np.float32)  # virtual tip velocity

    def reset(self):
        self.anchor_idx = None
        self.is_dragging = False
        self.active = False
        self.VT_vel[:] = 0.0
        if self.A is not None:
            self.VT[:] = self.A

    # -----------------------------------------------------------------
    def start_grab(self, pinch_point, face_pts, bbox):
        """On a fresh pinch-down: find the nearest grabbable landmark and
        lock onto it as the tendon's anchor for this drag."""
        candidates = face_pts[GRABBABLE_IDX]
        d2 = np.sum((candidates - np.array(pinch_point, dtype=np.float32)) ** 2, axis=1)
        local_best = int(np.argmin(d2))
        best_idx = GRABBABLE_IDX[local_best]

        xmin, ymin, xmax, ymax = bbox
        grab_radius = max(xmax - xmin, ymax - ymin) * 1.1
        if math.sqrt(float(d2[local_best])) > grab_radius:
            return  # pinch happened too far from any tracked region, ignore

        self.anchor_idx = best_idx
        self.A = face_pts[best_idx].copy()
        self.VT[:] = self.A          # tendon starts collapsed, grows as you drag
        self.VT_vel[:] = 0.0
        self.is_dragging = True
        self.active = True

    def end_grab(self):
        self.is_dragging = False
        # anchor_idx / active stay set so the tendon can spring back home
        # smoothly across the following frames instead of vanishing instantly

    # -----------------------------------------------------------------
    def update_and_warp(self, frame, face_pts, bbox, pinch_point, dt, elasticity):
        """Advances the tip spring and, if a tendon is currently visible,
        returns a new frame with that local capsule region warped. Only the
        small bounding box around the tendon is touched -- the rest of the
        frame is passed through untouched."""
        h, w = frame.shape[:2]
        dt = clamp(dt, 1.0 / 90.0, 1.0 / 15.0)

        if self.anchor_idx is not None:
            self.A = face_pts[self.anchor_idx].copy()  # live-track the anchor

        if not self.active or self.A is None:
            return frame, None

        xmin, ymin, xmax, ymax = bbox
        face_size = max(xmax - xmin, ymax - ymin)
        max_drag = MAX_DRAG_FRAC * face_size * elasticity

        # ---- spring physics for the virtual tip ------------------------
        if self.is_dragging and pinch_point is not None:
            raw = np.array(pinch_point, dtype=np.float32) - self.A
            n = float(np.linalg.norm(raw))
            if n > max_drag and n > 1e-6:
                raw = raw * (max_drag / n)
            target = self.A + raw
            # fast, near-critically-damped chase while actively held
            self.VT += (target - self.VT) * min(DRAG_LERP * dt, 1.0)
            self.VT_vel *= 0.3
        else:
            target = self.A
            force = (target - self.VT) * SPRING_K - self.VT_vel * SPRING_DAMPING
            self.VT_vel += force * dt
            self.VT += self.VT_vel * dt

        L = float(np.linalg.norm(self.VT - self.A))

        if not self.is_dragging and L < SETTLE_DIST and float(np.linalg.norm(self.VT_vel)) < SETTLE_VEL:
            # fully snapped home -- stop animating until the next grab
            self.active = False
            self.anchor_idx = None
            self.VT[:] = self.A
            self.VT_vel[:] = 0.0
            return frame, None

        if L < 2.0:
            return frame, None  # negligible stretch, nothing worth warping yet

        # ---- tendon geometry --------------------------------------------
        R_base = max(R_BASE_FRAC * face_size, 12.0)
        R_tip = max(R_base * R_TIP_FRAC, 5.0)

        axis = (self.VT - self.A) / L
        perp = np.array([-axis[1], axis[0]], dtype=np.float32)

        pad = R_base + 12
        x0 = int(clamp(min(self.A[0], self.VT[0]) - pad, 0, w - 2))
        y0 = int(clamp(min(self.A[1], self.VT[1]) - pad, 0, h - 2))
        x1 = int(clamp(max(self.A[0], self.VT[0]) + pad, x0 + 2, w))
        y1 = int(clamp(max(self.A[1], self.VT[1]) + pad, y0 + 2, h))

        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        dx = X - self.A[0]
        dy = Y - self.A[1]
        t = dx * axis[0] + dy * axis[1]            # signed distance along tendon axis
        v = dx * perp[0] + dy * perp[1]             # signed perpendicular distance

        tc = np.clip(t, 0.0, L)
        R_t = R_base + (R_tip - R_base) * (tc / L)   # local tendon radius, tapering A->VT

        dist_A = np.sqrt(dx * dx + dy * dy)
        dist_VT = np.sqrt((X - self.VT[0]) ** 2 + (Y - self.VT[1]) ** 2)

        inside_body = (t >= 0) & (t <= L) & (np.abs(v) <= R_t)
        inside_base_cap = (t < 0) & (dist_A <= R_base)
        inside_tip_cap = (t > L) & (dist_VT <= R_tip)
        inside = inside_body | inside_base_cap | inside_tip_cap

        # ---- the key "real elastic" trick: smear a small source patch ---
        # near the anchor across the whole tendon length, so the same skin
        # texture gets visibly thinner/stretched the further out it goes.
        axial_scale = (R_base * AXIAL_SOURCE_FRAC) / L
        t_src = np.where(t < 0, t, tc * axial_scale)
        v_src = v * (R_base / np.maximum(R_t, 1e-3))

        src_x = self.A[0] + axis[0] * t_src + perp[0] * v_src
        src_y = self.A[1] + axis[1] * t_src + perp[1] * v_src
        src_x = np.clip(src_x, 0, w - 1).astype(np.float32)
        src_y = np.clip(src_y, 0, h - 1).astype(np.float32)

        map_x = np.where(inside, src_x, X).astype(np.float32)
        map_y = np.where(inside, src_y, Y).astype(np.float32)

        warped_sub = cv2.remap(
            frame, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # soft feathered edge so the tendon blends into surrounding skin
        mask_u8 = (inside.astype(np.uint8)) * 255
        mask_u8 = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=3.0)
        mask_f = (mask_u8.astype(np.float32) / 255.0)[..., None]

        # subtle rubbery sheen: the thinner it gets, the brighter/glossier
        stretch_ratio = np.clip(1.0 - (R_t / R_base), 0.0, 1.0)
        sheen = (stretch_ratio * 28.0)[..., None] * inside_body.astype(np.float32)[..., None]

        out_sub = warped_sub.astype(np.float32) + sheen
        out_sub = np.clip(out_sub, 0, 255)

        frame_sub = frame[y0:y1, x0:x1].astype(np.float32)
        blended = frame_sub * (1.0 - mask_f) + out_sub * mask_f
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out = frame.copy()
        out[y0:y1, x0:x1] = blended

        info = {
            "A": (float(self.A[0]), float(self.A[1])),
            "VT": (float(self.VT[0]), float(self.VT[1])),
            "L": L,
            "R_base": R_base,
            "R_tip": R_tip,
            "axis": (float(axis[0]), float(axis[1])),
            "perp": (float(perp[0]), float(perp[1])),
        }
        return out, info


# =====================================================================
# VISUAL FX HELPERS
# =====================================================================

def draw_tendon_outline(img, info, color, t_time):
    """Draws a soft glowing rim around the tapered tendon silhouette plus a
    pulsing ring at the tip, so the grabbed/stretched region clearly reads
    on camera (and on social media)."""
    A = np.array(info["A"], dtype=np.float32)
    VT = np.array(info["VT"], dtype=np.float32)
    axis = np.array(info["axis"], dtype=np.float32)
    perp = np.array(info["perp"], dtype=np.float32)
    L = info["L"]
    R_base, R_tip = info["R_base"], info["R_tip"]

    steps = 14
    top_pts, bot_pts = [], []
    for i in range(steps + 1):
        tt = L * (i / steps)
        r = R_base + (R_tip - R_base) * (tt / max(L, 1e-3))
        p = A + axis * tt
        top_pts.append(p + perp * r)
        bot_pts.append(p - perp * r)

    poly = np.array(top_pts + bot_pts[::-1], dtype=np.int32)

    overlay = img.copy()
    cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

    pulse = 1.0 + 0.18 * math.sin(t_time * 9.0)
    r_ring = max(4, int(R_tip * 1.4 * pulse))
    overlay2 = img.copy()
    cv2.circle(overlay2, (int(VT[0]), int(VT[1])), r_ring, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay2, 0.6, img, 0.4, 0, dst=img)


def apply_cartoon_outline(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img, edges_bgr)


def draw_ui(img, fps, elasticity, elastic_on, cartoon_on, face_found, hand_found):
    h, w = img.shape[:2]
    panel_h = 92
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

    cyan = (255, 230, 0)
    dim = (180, 180, 180)

    cv2.putText(img, "ELASTIC SKIN AI", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cyan, 2, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:4.1f}", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(img, f"Elasticity: {elasticity:.1f}x", (16, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, dim, 1, cv2.LINE_AA)

    status = f"Warp:{'ON' if elastic_on else 'OFF'}  Cartoon:{'ON' if cartoon_on else 'OFF'}"
    cv2.putText(img, status, (230, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dim, 1, cv2.LINE_AA)

    instr = "Pinch (thumb+index) near your face and drag to stretch"
    if not face_found:
        instr = "No face detected..."
    elif not hand_found:
        instr = "Show your hand: Pinch to stretch"
    cv2.putText(img, instr, (230, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 1, cv2.LINE_AA)

    keys = "E:warp  R:reset  +/-:strength  C:cartoon  Q:quit"
    cv2.putText(img, keys, (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dim, 1, cv2.LINE_AA)


# =====================================================================
# MAIN APPLICATION
# =====================================================================

def main():
    # ---- sound setup ---------------------------------------------------
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    pygame.init()

    stretch_sound = pygame.sndarray.make_sound(generate_stretch_loop())
    snap_sound = pygame.sndarray.make_sound(generate_snap_sound())
    stretch_channel = pygame.mixer.Channel(0)
    snap_channel = pygame.mixer.Channel(1)

    # ---- camera setup ----------------------------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)

    # ---- mediapipe setup ---------------------------------------------------
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    skin = ElasticSkin()

    elasticity = 1.0
    elastic_on = True
    cartoon_on = False

    was_pinching = False
    prev_pinch_point = None
    prev_out_frame = None

    fps_avg = 24.0
    prev_t = time.time()

    print("Elastic Skin AI running. Press Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: failed to read frame from webcam, stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror for natural interaction

            h0, w0 = frame.shape[:2]
            if w0 > PROC_MAX_WIDTH:
                scale = PROC_MAX_WIDTH / float(w0)
                frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - prev_t
            prev_t = now
            inst_fps = 1.0 / dt if dt > 0 else fps_avg
            fps_avg = fps_avg * 0.9 + inst_fps * 0.1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)
            rgb.flags.writeable = True

            face_pts, bbox = get_largest_face(face_results, w, h)
            face_found = face_pts is not None

            hand_found, pinch_point, pinch_ratio = get_pinch_state(hand_results, w, h)

            # ---- pinch state machine (with hysteresis to avoid flicker) --
            is_pinching = False
            if hand_found and pinch_ratio is not None:
                if was_pinching:
                    is_pinching = pinch_ratio < PINCH_RATIO_OFF
                else:
                    is_pinching = pinch_ratio < PINCH_RATIO_ON

            if face_found and elastic_on:
                if is_pinching and not was_pinching:
                    skin.start_grab(pinch_point, face_pts, bbox)
                    if not stretch_channel.get_busy():
                        stretch_channel.play(stretch_sound, loops=-1)
                elif not is_pinching and was_pinching:
                    skin.end_grab()
                    stretch_channel.stop()
                    snap_channel.play(snap_sound)
            elif was_pinching and not is_pinching:
                # hand-only release with no/disabled face tracking
                skin.end_grab()
                stretch_channel.stop()
                snap_channel.play(snap_sound)

            was_pinching = is_pinching

            # ---- deformation + warp ---------------------------------------
            info = None
            if face_found and elastic_on:
                out_frame, info = skin.update_and_warp(
                    frame, face_pts, bbox, pinch_point if is_pinching else None, dt, elasticity
                )
            else:
                out_frame = frame
                if not elastic_on:
                    skin.reset()  # keep things consistent while warp is disabled

            # ---- motion blur on fast drags ---------------------------------
            if (is_pinching and prev_pinch_point is not None and pinch_point is not None
                    and prev_out_frame is not None and prev_out_frame.shape == out_frame.shape):
                vel = dist(pinch_point, prev_pinch_point) / max(dt, 1e-3)
                blur_amount = clamp(vel / MOTION_BLUR_VEL_SCALE, 0.0, MOTION_BLUR_MAX)
                if blur_amount > 0.02:
                    out_frame = cv2.addWeighted(
                        out_frame, 1.0 - blur_amount, prev_out_frame, blur_amount, 0
                    )

            # ---- glow / tendon outline on the grabbed+stretched region -----
            if info is not None:
                draw_tendon_outline(out_frame, info, (255, 220, 0), now)

            # ---- cartoon overlay --------------------------------------------
            if cartoon_on:
                out_frame = apply_cartoon_outline(out_frame)

            # ---- UI -----------------------------------------------------------
            draw_ui(out_frame, fps_avg, elasticity, elastic_on, cartoon_on, face_found, hand_found)

            cv2.imshow(WINDOW_NAME, out_frame)

            prev_out_frame = out_frame.copy()
            prev_pinch_point = pinch_point if pinch_point is not None else prev_pinch_point

            # ---- keyboard handling -------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                break
            elif key in (ord('e'), ord('E')):
                elastic_on = not elastic_on
                if not elastic_on:
                    skin.reset()
                    stretch_channel.stop()
            elif key in (ord('r'), ord('R')):
                skin.reset()
                stretch_channel.stop()
            elif key in (ord('+'), ord('=')):
                elasticity = clamp(elasticity + ELASTICITY_STEP, ELASTICITY_MIN, ELASTICITY_MAX)
            elif key == ord('-'):
                elasticity = clamp(elasticity - ELASTICITY_STEP, ELASTICITY_MIN, ELASTICITY_MAX)
            elif key in (ord('c'), ord('C')):
                cartoon_on = not cartoon_on

            # window closed via the 'X' button
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        hands.close()
        pygame.mixer.quit()
        pygame.quit()


if __name__ == "__main__":
    main()