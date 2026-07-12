

import math
import random
import string
import sys
import time

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720

TRAIL_DECAY = 0.93          # closer to 1.0 = trail lingers longer
GLOW_BLUR_SIGMA = 9
STROKE_THICKNESS = 6

PINCH_THRESHOLD = 0.055     # normalized distance between thumb & index tips

TRACE_TIME_LIMIT = 9.0      # seconds to trace each letter
TRACE_BAND_PX = 26          # tolerance width around the letter outline
AUTO_ADVANCE_SCORE = 88.0   # % accuracy that instantly advances to next letter

# ----------------------------------------------------------------------------
# INIT
# ----------------------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
if not cap.isOpened():
    print("ERROR: could not open webcam.")
    sys.exit(1)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def hue_color(t):
    """Rainbow color that slowly cycles over time. Returns BGR tuple."""
    hue = int((t * 40) % 180)
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def make_letter_band(letter, w, h):
    """
    Renders a big letter, extracts its outline, and dilates it into a
    tolerance "band" mask the user's trail must fall within.
    Returns (band_mask, display_mask) both single-channel uint8.
    """
    solid = np.zeros((h, w), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 9.0
    thickness = 18
    text_size = cv2.getTextSize(letter, font, scale, thickness)[0]
    org = ((w - text_size[0]) // 2, (h + text_size[1]) // 2)
    cv2.putText(solid, letter, org, font, scale, 255, thickness, cv2.LINE_AA)

    edges = cv2.Canny(solid, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    outline = cv2.dilate(edges, kernel, iterations=1)

    band = cv2.dilate(outline, np.ones((TRACE_BAND_PX, TRACE_BAND_PX), np.uint8))
    return band, outline


def get_fingertip_and_pinch(landmarks, w, h):
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
    px, py = int(index_tip.x * w), int(index_tip.y * h)
    return (px, py), pinch_dist < PINCH_THRESHOLD


# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
class TraceRound:
    def __init__(self, w, h):
        self.letter = random.choice(string.ascii_uppercase)
        self.band, self.display_outline = make_letter_band(self.letter, w, h)
        self.round_trail_mask = np.zeros((h, w), dtype=np.uint8)
        self.start_time = time.time()
        self.finished = False
        self.last_score = 0.0

    def score(self):
        covered = cv2.bitwise_and(self.round_trail_mask, self.band)
        covered_px = int(np.count_nonzero(covered))
        band_px = max(1, int(np.count_nonzero(self.band)))
        drawn_px = max(1, int(np.count_nonzero(self.round_trail_mask)))

        completeness = covered_px / band_px          # how much of the letter you traced
        precision = covered_px / drawn_px             # how much of your drawing stayed on-target
        pct = (completeness * 0.6 + precision * 0.4) * 100
        self.last_score = min(100.0, pct)
        return self.last_score


class GameState:
    def __init__(self, w, h):
        self.mode = "free"   # "free" or "trace"
        self.total_score = 0
        self.letters_matched = 0
        self.trace_round = TraceRound(w, h)

    def new_letter(self, w, h):
        self.trace_round = TraceRound(w, h)


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: could not read from webcam.")
        return
    h, w = first_frame.shape[:2]

    trail_layer = np.zeros((h, w, 3), dtype=np.float32)
    prev_point = None
    state = GameState(w, h)

    print("GHOST TRACE running. Press F=free draw, T=trace mode, C=clear, N=new letter, Q=quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pen_down = False
        cur_point = None
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            cur_point, is_pinching = get_fingertip_and_pinch(landmarks, w, h)
            pen_down = not is_pinching

        # fade the glowing visual trail
        trail_layer *= TRAIL_DECAY

        if pen_down and cur_point:
            color = hue_color(now)
            if prev_point:
                cv2.line(trail_layer, prev_point, cur_point, color, STROKE_THICKNESS, cv2.LINE_AA)
                if state.mode == "trace" and not state.trace_round.finished:
                    cv2.line(state.trace_round.round_trail_mask, prev_point, cur_point,
                              255, STROKE_THICKNESS, cv2.LINE_AA)
            else:
                cv2.circle(trail_layer, cur_point, STROKE_THICKNESS // 2, color, -1, cv2.LINE_AA)
            prev_point = cur_point
        else:
            prev_point = None

        # glow bloom
        glow = cv2.GaussianBlur(trail_layer, (0, 0), GLOW_BLUR_SIGMA)
        composite = cv2.add(trail_layer, glow * 0.6)
        composite_u8 = np.clip(composite, 0, 255).astype(np.uint8)

        out = cv2.add(frame, composite_u8)

        # fingertip marker
        if cur_point:
            marker_color = (0, 255, 255) if pen_down else (0, 0, 255)
            cv2.circle(out, cur_point, 10, marker_color, 2, cv2.LINE_AA)

        # ---- TRACE MODE overlay ----
        if state.mode == "trace":
            tr = state.trace_round
            overlay = out.copy()
            overlay[tr.display_outline > 0] = (255, 180, 60)
            out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)

            elapsed = now - tr.start_time
            remaining = max(0.0, TRACE_TIME_LIMIT - elapsed)
            live_score = tr.score() if not tr.finished else tr.last_score

            cv2.putText(out, f"Trace the letter: {tr.letter}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, f"Time: {remaining:0.1f}s", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, f"Match: {live_score:0.0f}%", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 255, 120), 2, cv2.LINE_AA)

            if not tr.finished and (remaining <= 0 or live_score >= AUTO_ADVANCE_SCORE):
                tr.finished = True
                gained = int(live_score * 10)
                state.total_score += gained
                state.letters_matched += 1
                cv2.putText(out, f"+{gained} pts!", (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 255, 120), 3, cv2.LINE_AA)
                cv2.imshow("Ghost Trace", out)
                cv2.waitKey(700)
                state.new_letter(w, h)

            cv2.putText(out, f"Total Score: {state.total_score}  Letters: {state.letters_matched}",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(out, "FREE DRAW - point to draw, pinch to lift pen", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(out, "[F] Free  [T] Trace  [C] Clear  [N] New letter  [Q] Quit",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Ghost Trace", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            state.mode = "free"
        elif key == ord('t'):
            state.mode = "trace"
            state.new_letter(w, h)
        elif key == ord('c'):
            trail_layer[:] = 0
            if state.mode == "trace":
                state.trace_round.round_trail_mask[:] = 0
        elif key == ord('n') and state.mode == "trace":
            state.new_letter(w, h)

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()