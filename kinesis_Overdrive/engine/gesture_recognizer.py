"""
Hand-gesture recognition built on MediaPipe Hands.

Classifies each detected hand as OPEN_PALM, FIST, or UNKNOWN by counting
extended fingers from landmark geometry. This drives all menu / pause
navigation so the player never has to touch a keyboard mid-workout.
"""

import cv2
import mediapipe as mp

OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
UNKNOWN = "UNKNOWN"

# Landmark indices (MediaPipe Hands topology)
_TIPS = [4, 8, 12, 16, 20]     # thumb, index, middle, ring, pinky tips
_PIPS = [3, 6, 10, 14, 18]     # corresponding lower joints
_WRIST = 0


class GestureRecognizer:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.6):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """Returns a list of dicts: {'gesture': str, 'center': (x, y)}."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        detections = []
        if not results.multi_hand_landmarks:
            return detections

        for hand_landmarks in results.multi_hand_landmarks:
            pts = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
            gesture = self._classify(pts)
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            detections.append({"gesture": gesture, "center": (cx, cy)})
        return detections

    def _classify(self, pts):
        wrist = pts[_WRIST]
        extended = 0
        for tip_idx, pip_idx in zip(_TIPS, _PIPS):
            tip, pip = pts[tip_idx], pts[pip_idx]
            # A finger counts as "extended" if its tip is farther from the
            # wrist than its pip joint is — robust to hand rotation, unlike
            # a naive y-coordinate comparison.
            d_tip = self._dist(tip, wrist)
            d_pip = self._dist(pip, wrist)
            if d_tip > d_pip * 1.15:
                extended += 1

        if extended >= 4:
            return OPEN_PALM
        if extended <= 1:
            return FIST
        return UNKNOWN

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def close(self):
        self._hands.close()


class GestureHoldTracker:
    """
    Debounces a target gesture: tracks how long it has been held
    continuously and reports progress toward a confirmation threshold.
    Use one instance per gesture you want to watch (e.g. one for
    OPEN_PALM-to-start, one for FIST-to-pause).
    """

    def __init__(self, target_gesture, hold_seconds):
        self.target_gesture = target_gesture
        self.hold_seconds = hold_seconds
        self._hold_start = None

    def update(self, detections, now):
        """
        detections: list from GestureRecognizer.process().
        Returns (progress_0_to_1, confirmed: bool).
        """
        present = any(d["gesture"] == self.target_gesture for d in detections)
        if present:
            if self._hold_start is None:
                self._hold_start = now
            elapsed = now - self._hold_start
            progress = min(1.0, elapsed / self.hold_seconds)
            if progress >= 1.0:
                self._hold_start = None  # reset so it must be re-triggered
                return 1.0, True
            return progress, False
        else:
            self._hold_start = None
            return 0.0, False
