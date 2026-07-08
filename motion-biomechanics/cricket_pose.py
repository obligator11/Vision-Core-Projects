

import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math

# ---------------- MediaPipe setup ----------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6,
                     min_tracking_confidence=0.6)

# ---------------- Game constants ----------------
HIT_ZONE_Y_RATIO = 0.55        # where the "hit zone" line sits vertically
SWING_SPEED_SIX = 1400         # px/sec threshold for a six
SWING_SPEED_FOUR = 850         # px/sec threshold for a four
SWING_SPEED_SINGLE = 350       # px/sec threshold for 1-2 runs
HIT_WINDOW_FRAMES = 6          # frames around hit-zone crossing counted as "the swing moment"
BALLS_PER_OVER = 6
MAX_WICKETS = 6

WRIST_LANDMARK = mp_pose.PoseLandmark.RIGHT_WRIST

# ---------------- Game state ----------------
class GameState:
    def __init__(self):
        self.score = 0
        self.wickets = 0
        self.balls_bowled = 0
        self.ball_active = False
        self.ball_y = 0
        self.ball_x_ratio = 0.5
        self.ball_speed = 6            # px per frame, increases with overs
        self.result_text = ""
        self.result_timer = 0
        self.wrist_history = []        # (x, y, timestamp)
        self.hit_registered = False
        self.game_over = False
        self.spawn_delay_until = time.time() + 2

    def reset_ball(self):
        self.ball_active = False
        self.hit_registered = False
        self.result_text = ""
        self.ball_x_ratio = random.uniform(0.4, 0.6)
        self.spawn_delay_until = time.time() + 1.5

    def restart_game(self):
        self.__init__()


def get_over_speed_bonus(balls_bowled):
    over_number = balls_bowled // BALLS_PER_OVER
    return over_number * 1.2   # ball gets faster each completed over


def wrist_swing_speed(history):
    """Compute max px/sec speed from recent wrist position history."""
    if len(history) < 2:
        return 0
    max_speed = 0
    for i in range(1, len(history)):
        x1, y1, t1 = history[i - 1]
        x2, y2, t2 = history[i]
        dt = t2 - t1
        if dt <= 0:
            continue
        dist = math.hypot(x2 - x1, y2 - y1)
        speed = dist / dt
        max_speed = max(max_speed, speed)
    return max_speed


def classify_hit(speed):
    if speed >= SWING_SPEED_SIX:
        return 6, "SIX!! 🚀"
    elif speed >= SWING_SPEED_FOUR:
        return 4, "FOUR! 🏏"
    elif speed >= SWING_SPEED_SINGLE:
        return random.choice([1, 2]), "Good run!"
    else:
        return 0, "OUT! Bowled 🎯"


def draw_ui(frame, state, w, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Score: {state.score}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
    cv2.putText(frame, f"Wickets: {state.wickets}/{MAX_WICKETS}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

    overs = state.balls_bowled // BALLS_PER_OVER
    balls_this_over = state.balls_bowled % BALLS_PER_OVER
    cv2.putText(frame, f"Overs: {overs}.{balls_this_over}", (w - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Hit zone line
    hit_y = int(h * HIT_ZONE_Y_RATIO)
    cv2.line(frame, (0, hit_y), (w, hit_y), (255, 255, 255), 2)
    cv2.putText(frame, "HIT ZONE", (w // 2 - 60, hit_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if state.result_text and time.time() < state.result_timer:
        text_size = cv2.getTextSize(state.result_text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
        tx = w // 2 - text_size[0] // 2
        cv2.putText(frame, state.result_text, (tx, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

    if state.game_over:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "GAME OVER", (w // 2 - 160, h // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        cv2.putText(frame, f"Final Score: {state.score}", (w // 2 - 140, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(frame, "Press 'r' to restart or 'q' to quit", (w // 2 - 220, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    state = GameState()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if not state.game_over:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            wrist_px = None
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark[WRIST_LANDMARK]
                wrist_px = (int(lm.x * w), int(lm.y * h))
                cv2.circle(frame, wrist_px, 14, (0, 255, 255), -1)

                now = time.time()
                state.wrist_history.append((wrist_px[0], wrist_px[1], now))
                state.wrist_history = [p for p in state.wrist_history if now - p[2] < 0.4]

            # Spawn a new ball after delay
            if not state.ball_active and time.time() > state.spawn_delay_until:
                state.ball_active = True
                state.ball_y = 60
                state.ball_speed = 6 + get_over_speed_bonus(state.balls_bowled)

            # Animate ball
            if state.ball_active:
                state.ball_y += state.ball_speed
                bx = int(state.ball_x_ratio * w)
                by = int(state.ball_y)
                cv2.circle(frame, (bx, by), 12, (0, 0, 255), -1)
                cv2.circle(frame, (bx, by), 12, (255, 255, 255), 2)

                hit_y = int(h * HIT_ZONE_Y_RATIO)

                # Ball is inside the hit window
                if abs(by - hit_y) < 25 and not state.hit_registered:
                    speed = wrist_swing_speed(state.wrist_history)
                    runs, text = classify_hit(speed)
                    state.hit_registered = True
                    state.result_text = text
                    state.result_timer = time.time() + 1.2
                    state.score += runs
                    if runs == 0:
                        state.wickets += 1
                    state.balls_bowled += 1
                    if state.wickets >= MAX_WICKETS:
                        state.game_over = True

                # Ball passed the zone without a swing check (safety net)
                if by > h - 40:
                    state.reset_ball()
                elif state.hit_registered and by > hit_y + 60:
                    state.reset_ball()

        draw_ui(frame, state, w, h)
        cv2.imshow("AI Cricket - Pose Controlled", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and state.game_over:
            state.restart_game()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()