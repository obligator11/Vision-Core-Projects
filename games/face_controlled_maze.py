import math
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

TILT_SENSITIVITY = 2400.0     # how much ball acceleration per radian of head tilt
BALL_RADIUS = 12
BALL_FRICTION = 0.90
WALL_THICKNESS = 14

# MediaPipe Face Mesh landmark indices used for head-pose solvePnP
POSE_LANDMARK_IDS = {
    "nose_tip": 1, "chin": 152, "left_eye_outer": 33,
    "right_eye_outer": 263, "left_mouth": 61, "right_mouth": 291,
}
# Generic 3D face model (mm), matched to the landmarks above
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye outer corner
    (43.3, 32.7, -26.0),    # right eye outer corner
    (-28.9, -28.9, -24.1),  # left mouth corner
    (28.9, -28.9, -24.1),   # right mouth corner
], dtype=np.float64)

# Maze layout: grid of walls as fractional rects (x, y, w, h) in 0..1 space
MAZE_WALLS_FRAC = [
    (0.0, 0.0, 1.0, 0.03), (0.0, 0.97, 1.0, 0.03),
    (0.0, 0.0, 0.03, 1.0), (0.97, 0.0, 0.03, 1.0),
    (0.15, 0.0, 0.03, 0.55), (0.30, 0.45, 0.03, 0.55),
    (0.45, 0.0, 0.03, 0.6), (0.60, 0.35, 0.03, 0.65),
    (0.75, 0.0, 0.03, 0.55),
]
START_FRAC = (0.06, 0.06)
GOAL_FRAC = (0.90, 0.90)
GOAL_RADIUS_FRAC = 0.035

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
pygame.init()
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Face-Controlled Maze")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont("arial", 48, bold=True)
font_med = pygame.font.SysFont("arial", 28, bold=True)
font_small = pygame.font.SysFont("arial", 18)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=False,
                              min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)

CAM_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
CAM_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
FOCAL_LENGTH = CAM_W
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, CAM_W / 2],
    [0, FOCAL_LENGTH, CAM_H / 2],
    [0, 0, 1]
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1))


# ----------------------------------------------------------------------------
# SOUND
# ----------------------------------------------------------------------------
def to_sound(samples, volume=0.6):
    samples = np.clip(samples * volume, -1, 1)
    audio = (samples * 32767).astype(np.int16)
    stereo = np.column_stack([audio, audio])
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))


def make_thud(ms=90):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    tone = np.sin(2 * np.pi * 110 * t) * np.exp(-t * 25)
    return to_sound(tone)


def make_victory(ms=700):
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.linspace(0, ms / 1000, n, False)
    notes = [523, 659, 784, 1047]
    tone = np.zeros(n)
    seg = n // len(notes)
    for i, f in enumerate(notes):
        s, e = i * seg, min(n, (i + 1) * seg)
        seg_t = t[s:e] - t[s]
        tone[s:e] = np.sin(2 * np.pi * f * seg_t) * np.exp(-seg_t * 3)
    return to_sound(tone, volume=0.5)


snd_wall = make_thud()
snd_victory = make_victory()


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def frame_to_surface(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def estimate_head_pose(landmarks, frame_w, frame_h):
    image_points = np.array([
        (landmarks[POSE_LANDMARK_IDS["nose_tip"]].x * frame_w,
         landmarks[POSE_LANDMARK_IDS["nose_tip"]].y * frame_h),
        (landmarks[POSE_LANDMARK_IDS["chin"]].x * frame_w,
         landmarks[POSE_LANDMARK_IDS["chin"]].y * frame_h),
        (landmarks[POSE_LANDMARK_IDS["left_eye_outer"]].x * frame_w,
         landmarks[POSE_LANDMARK_IDS["left_eye_outer"]].y * frame_h),
        (landmarks[POSE_LANDMARK_IDS["right_eye_outer"]].x * frame_w,
         landmarks[POSE_LANDMARK_IDS["right_eye_outer"]].y * frame_h),
        (landmarks[POSE_LANDMARK_IDS["left_mouth"]].x * frame_w,
         landmarks[POSE_LANDMARK_IDS["left_mouth"]].y * frame_h),
        (landmarks[POSE_LANDMARK_IDS["right_mouth"]].x * frame_w,
         landmarks[POSE_LANDMARK_IDS["right_mouth"]].y * frame_h),
    ], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, CAMERA_MATRIX, DIST_COEFFS,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = math.atan2(-rmat[2, 0], sy)
    yaw = math.atan2(rmat[1, 0], rmat[0, 0])
    return yaw, pitch


# ----------------------------------------------------------------------------
# MAZE / PHYSICS
# ----------------------------------------------------------------------------
def get_walls():
    return [pygame.Rect(x * WIDTH, y * HEIGHT, w * WIDTH, h * HEIGHT)
            for (x, y, w, h) in MAZE_WALLS_FRAC]


def get_start_pos():
    return START_FRAC[0] * WIDTH, START_FRAC[1] * HEIGHT


def get_goal():
    return (GOAL_FRAC[0] * WIDTH, GOAL_FRAC[1] * HEIGHT, GOAL_RADIUS_FRAC * WIDTH)


class GameState:
    def __init__(self):
        pos = get_start_pos()
        self.ball_x, self.ball_y = pos
        self.vx, self.vy = 0.0, 0.0
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.calibrated = False
        self.start_time = time.time()
        self.won = False
        self.finish_time = None
        self.last_wall_sound = 0

    def reset(self):
        pos = get_start_pos()
        self.ball_x, self.ball_y = pos
        self.vx, self.vy = 0.0, 0.0
        self.start_time = time.time()
        self.won = False
        self.finish_time = None


state = GameState()


def resolve_wall_collisions(walls, now):
    ball_rect = pygame.Rect(state.ball_x - BALL_RADIUS, state.ball_y - BALL_RADIUS,
                             BALL_RADIUS * 2, BALL_RADIUS * 2)
    for wall in walls:
        if ball_rect.colliderect(wall):
            overlap_left = ball_rect.right - wall.left
            overlap_right = wall.right - ball_rect.left
            overlap_top = ball_rect.bottom - wall.top
            overlap_bottom = wall.bottom - ball_rect.top
            min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

            if min_overlap == overlap_left:
                state.ball_x -= overlap_left
                state.vx = -state.vx * 0.3
            elif min_overlap == overlap_right:
                state.ball_x += overlap_right
                state.vx = -state.vx * 0.3
            elif min_overlap == overlap_top:
                state.ball_y -= overlap_top
                state.vy = -state.vy * 0.3
            else:
                state.ball_y += overlap_bottom
                state.vy = -state.vy * 0.3

            if now - state.last_wall_sound > 0.15:
                snd_wall.play()
                state.last_wall_sound = now
            ball_rect = pygame.Rect(state.ball_x - BALL_RADIUS, state.ball_y - BALL_RADIUS,
                                     BALL_RADIUS * 2, BALL_RADIUS * 2)


def update_physics(yaw, pitch, dt, now):
    if state.won:
        return
    walls = get_walls()

    if state.calibrated and yaw is not None:
        rel_yaw = yaw - state.neutral_yaw
        rel_pitch = pitch - state.neutral_pitch
        state.vx += rel_yaw * TILT_SENSITIVITY * dt
        state.vy += -rel_pitch * TILT_SENSITIVITY * dt

    state.vx *= BALL_FRICTION
    state.vy *= BALL_FRICTION
    state.ball_x += state.vx * dt
    state.ball_y += state.vy * dt

    state.ball_x = max(BALL_RADIUS, min(WIDTH - BALL_RADIUS, state.ball_x))
    state.ball_y = max(BALL_RADIUS, min(HEIGHT - BALL_RADIUS, state.ball_y))

    resolve_wall_collisions(walls, now)

    gx, gy, gr = get_goal()
    if math.hypot(state.ball_x - gx, state.ball_y - gy) < gr:
        state.won = True
        state.finish_time = time.time() - state.start_time
        snd_victory.play()


def draw_maze():
    for wall in get_walls():
        pygame.draw.rect(screen, (90, 90, 110), wall, border_radius=4)
        pygame.draw.rect(screen, (140, 140, 170), wall, 2, border_radius=4)

    gx, gy, gr = get_goal()
    pygame.draw.circle(screen, (60, 220, 90), (int(gx), int(gy)), int(gr))
    pygame.draw.circle(screen, (255, 255, 255), (int(gx), int(gy)), int(gr), 2)
    goal_label = font_small.render("GOAL", True, (255, 255, 255))
    screen.blit(goal_label, (gx - goal_label.get_width() / 2, gy - gr - 22))

    pygame.draw.circle(screen, (255, 210, 60), (int(state.ball_x), int(state.ball_y)), BALL_RADIUS)
    pygame.draw.circle(screen, (255, 255, 255), (int(state.ball_x), int(state.ball_y)), BALL_RADIUS, 2)


def draw_hud(yaw, pitch, face_seen):
    if not state.calibrated:
        msg = font_med.render("Look straight at the camera, then press C to calibrate",
                               True, (255, 210, 60))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, 20))
    elif not face_seen:
        msg = font_med.render("Face not detected - come into frame", True, (255, 90, 90))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, 20))
    else:
        elapsed = time.time() - state.start_time
        timer_label = font_med.render(f"Time: {elapsed:0.1f}s", True, (255, 255, 255))
        screen.blit(timer_label, (20, 20))

    if state.won:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))
        msg = font_big.render("MAZE COMPLETE!", True, (80, 255, 120))
        screen.blit(msg, (WIDTH / 2 - msg.get_width() / 2, HEIGHT / 2 - 60))
        t_label = font_med.render(f"Time: {state.finish_time:0.1f}s", True, (255, 255, 255))
        screen.blit(t_label, (WIDTH / 2 - t_label.get_width() / 2, HEIGHT / 2))
        hint = font_small.render("Press R to try again, Q to quit", True, (220, 220, 220))
        screen.blit(hint, (WIDTH / 2 - hint.get_width() / 2, HEIGHT / 2 + 50))

    title = font_small.render("FACE-CONTROLLED MAZE - tilt your head to steer the ball",
                               True, (255, 255, 255))
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, HEIGHT - 26))


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    global WIDTH, HEIGHT, screen
    running = True
    last_time = time.time()

    while running:
        now = time.time()
        dt = min(0.05, now - last_time)
        last_time = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = max(400, event.w), max(300, event.h)
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                state.reset()
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    state.reset()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_seen = False
        yaw = pitch = None
        if results.multi_face_landmarks:
            face_seen = True
            landmarks = results.multi_face_landmarks[0].landmark
            pose_result = estimate_head_pose(landmarks, frame.shape[1], frame.shape[0])
            if pose_result:
                yaw, pitch = pose_result

        keys_now = pygame.key.get_pressed()
        if keys_now[pygame.K_c] and face_seen and yaw is not None:
            state.neutral_yaw = yaw
            state.neutral_pitch = pitch
            state.calibrated = True

        if state.calibrated:
            update_physics(yaw, pitch, dt, now)

        surf = frame_to_surface(frame)
        screen.blit(surf, (0, 0))
        dim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 130))
        screen.blit(dim, (0, 0))

        draw_maze()
        draw_hud(yaw, pitch, face_seen)

        pygame.display.flip()
        clock.tick(FPS)

    cap.release()
    face_mesh.close()
    pygame.quit()


if __name__ == "__main__":
    main()