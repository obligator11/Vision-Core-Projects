import sys
import time

import cv2
import numpy as np
import pygame
import mediapipe as mp

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1000, 700
FPS = 30
CAM_INDEX = 0
SAMPLE_RATE = 22050

HEATMAP_DECAY = 0.997          # slow decay so the map stays "live" over a session
HEATMAP_BLOB_SIGMA = 28
HEATMAP_DOWNSCALE = 4          # compute heatmap at reduced res for performance

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_VERT = (159, 145)
RIGHT_EYE_VERT = (386, 374)

CALIBRATION_POINTS_FRAC = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)]

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Gaze Heatmap Reader")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 40, bold=True)
font_med = pygame.font.SysFont("arial", 26, bold=True)
font_small = pygame.font.SysFont("arial", 18)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# SOUND
# ----------------------------------------------------------------------------
def to_sound(samples, volume=0.6):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_chime(freq=700, ms=140):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t * 8)
    return to_sound(tone)


snd_calib = make_chime(700)
snd_done = make_chime(1100, 250)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def get_gaze_ratio(landmarks):
    """
    Returns (hx, hy) - the iris position as a 0..1 ratio within the eye
    socket, averaged across both eyes. ~0.5,0.5 = looking straight ahead.
    """
    def eye_ratio(iris_ids, corners, vert):
        iris_x = np.mean([landmarks[i].x for i in iris_ids])
        iris_y = np.mean([landmarks[i].y for i in iris_ids])
        left_x, right_x = landmarks[corners[0]].x, landmarks[corners[1]].x
        top_y, bottom_y = landmarks[vert[0]].y, landmarks[vert[1]].y
        hx = (iris_x - left_x) / max(1e-4, (right_x - left_x))
        hy = (iris_y - top_y) / max(1e-4, (bottom_y - top_y))
        return hx, hy

    lx, ly = eye_ratio(LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_VERT)
    rx, ry = eye_ratio(RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_VERT)
    return (lx + rx) / 2, (ly + ry) / 2


# ----------------------------------------------------------------------------
# CALIBRATION MAPPING (simple bilinear-ish regression)
# ----------------------------------------------------------------------------
class GazeMapper:
    def __init__(self):
        self.samples = []   # list of (gaze_ratio_xy, screen_frac_xy)
        self.calibrated = False
        self.coeffs_x = None
        self.coeffs_y = None

    def add_sample(self, gaze_xy, screen_frac_xy):
        self.samples.append((gaze_xy, screen_frac_xy))

    def fit(self):
        if len(self.samples) < 4:
            return False
        A = []
        bx, by = [], []
        for (gx, gy), (sx, sy) in self.samples:
            A.append([gx, gy, gx * gy, 1.0])
            bx.append(sx)
            by.append(sy)
        A = np.array(A)
        self.coeffs_x, *_ = np.linalg.lstsq(A, np.array(bx), rcond=None)
        self.coeffs_y, *_ = np.linalg.lstsq(A, np.array(by), rcond=None)
        self.calibrated = True
        return True

    def map(self, gaze_xy):
        gx, gy = gaze_xy
        if not self.calibrated:
            return gx, gy   # naive fallback: raw ratio as screen fraction
        feat = np.array([gx, gy, gx * gy, 1.0])
        sx = float(np.dot(feat, self.coeffs_x))
        sy = float(np.dot(feat, self.coeffs_y))
        return max(0.0, min(1.0, sx)), max(0.0, min(1.0, sy))


mapper = GazeMapper()


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class GameState:
    def __init__(self):
        self.calibrating = True
        self.calib_index = 0
        self.heatmap = None   # low-res float accumulator, created on first frame
        self.show_camera = True
        self.last_gaze_screen = None

    def clear_heatmap(self):
        if self.heatmap is not None:
            self.heatmap[:] = 0


state = GameState()


def ensure_heatmap():
    if state.heatmap is None:
        hw = max(1, WIDTH // HEATMAP_DOWNSCALE)
        hh = max(1, HEIGHT // HEATMAP_DOWNSCALE)
        state.heatmap = np.zeros((hh, hw), dtype=np.float32)


def add_gaze_point(sx_frac, sy_frac):
    ensure_heatmap()
    hh, hw = state.heatmap.shape
    cx, cy = int(sx_frac * hw), int(sy_frac * hh)
    if 0 <= cx < hw and 0 <= cy < hh:
        yy, xx = np.ogrid[:hh, :hw]
        sigma = HEATMAP_BLOB_SIGMA / HEATMAP_DOWNSCALE
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        state.heatmap += blob.astype(np.float32) * 0.6


def render_heatmap_surface():
    ensure_heatmap()
    hm = state.heatmap
    if hm.max() > 0:
        norm = np.clip(hm / max(hm.max(), 1e-6), 0, 1)
    else:
        norm = hm

    # simple "jet-like" colormap via numpy (no matplotlib dependency)
    r = np.clip(1.5 - np.abs(4 * norm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * norm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * norm - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    alpha = (norm > 0.03).astype(np.float32)[..., None]
    rgba = np.concatenate([rgb, alpha], axis=-1)
    rgba_u8 = (rgba * 255).astype(np.uint8)

    surf = pygame.image.frombuffer(
        np.transpose(rgba_u8, (1, 0, 2)).copy(), (rgba_u8.shape[1], rgba_u8.shape[0]), "RGBA")
    return pygame.transform.smoothscale(surf, (WIDTH, HEIGHT))


def draw_calibration(now):
    idx = state.calib_index
    fx, fy = CALIBRATION_POINTS_FRAC[idx]
    px, py = int(fx * WIDTH), int(fy * HEIGHT)
    pulse = 18 + int(6 * abs(np.sin(now * 4)))
    pygame.draw.circle(screen, (255, 60, 60), (px, py), pulse)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), pulse, 3)

    msg = font_med.render(f"Look at the dot and press SPACE  ({idx + 1}/{len(CALIBRATION_POINTS_FRAC)})",
                           True, (255, 255, 255))
    screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, 30))


def draw_hud():
    mode = "Heatmap only" if not state.show_camera else "Camera + heatmap"
    label = font_small.render(f"[{mode}]  Press H to toggle, C to clear heatmap",
                               True, (255, 255, 255))
    screen.blit(label, (20, 20))

    title = font_small.render("GAZE HEATMAP READER - shows where you've actually been looking",
                               True, (255, 255, 255))
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 26))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    running = True

    while running:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = max(400, event.w), max(300, event.h)
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                state.heatmap = None
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_c:
                    state.clear_heatmap()
                elif event.key == pygame.K_h:
                    state.show_camera = not state.show_camera
                elif event.key == pygame.K_SPACE and state.calibrating:
                    if state.last_gaze_screen is not None:
                        fx, fy = CALIBRATION_POINTS_FRAC[state.calib_index]
                        mapper.add_sample(state.last_gaze_screen, (fx, fy))
                        snd_calib.play()
                        state.calib_index += 1
                        if state.calib_index >= len(CALIBRATION_POINTS_FRAC):
                            mapper.fit()
                            state.calibrating = False
                            snd_done.play()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_seen = False
        if results.multi_face_landmarks:
            face_seen = True
            landmarks = results.multi_face_landmarks[0].landmark
            gaze_ratio = get_gaze_ratio(landmarks)
            state.last_gaze_screen = gaze_ratio

            if not state.calibrating:
                sx, sy = mapper.map(gaze_ratio)
                add_gaze_point(sx, sy)

        state.heatmap = state.heatmap * HEATMAP_DECAY if state.heatmap is not None else None

        if state.show_camera:
            surf = frame_to_surface(frame)
            screen.blit(surf, (0, 0))
            dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 90))
            screen.blit(dim, (0, 0))
        else:
            screen.fill((15, 15, 20))

        if not state.calibrating:
            heat_surf = render_heatmap_surface()
            screen.blit(heat_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        if not face_seen:
            warn = font_med.render("Face not detected", True, (255, 90, 90))
            screen.blit(warn, (WIDTH / 2 - warn.get_width() / 2, HEIGHT / 2))

        if state.calibrating:
            draw_calibration(now)
        draw_hud()

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    face_mesh.close()
    pygame.quit()


if __name__ == "__main__":
    main()