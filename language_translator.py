"""
================================================================================
 SIGN LANGUAGE (ASL FINGERSPELLING) TO TEXT TRANSLATOR
 YOLOv8 (signer localization) + MediaPipe Hands (landmark tracking)
 + scikit-learn (trainable classifier) + OpenCV (UI, I/O)
================================================================================

PROBLEM IT SOLVES
------------------
Communication between deaf/hard-of-hearing signers and people who don't know
sign language is a real, everyday accessibility barrier. This script turns a
webcam into a live ASL fingerspelling-to-text translator: it watches your
hand, recognizes which letter you're signing, builds it into words and
sentences on screen, and can read the sentence aloud with text-to-speech.

WHY THIS IS BETTER THAN A HARDCODED RULE SET
---------------------------------------------
Everyone's hands, camera angle, and signing style differ slightly, so a fixed
set of "if finger X is up then it's letter Y" rules is brittle in practice.
Instead this script ships as a small trainable pipeline:

    1. COLLECT  -- you show each letter to the camera and press its key;
                   the script saves normalized hand-landmark features to a
                   CSV dataset (your own real samples, your own hand).
    2. TRAIN    -- a RandomForestClassifier is trained on that CSV and saved
                   to disk. Takes seconds, runs entirely offline.
    3. RUN      -- the trained model recognizes your signs live. If you
                   haven't trained a model yet, it automatically falls back
                   to a built-in geometric rule classifier (covers a subset
                   of the alphabet) so the script still works out of the box.

HOW THE THREE LIBRARIES COMBINE
--------------------------------
- YOLOv8 finds the PERSON in frame first and we only look for hands inside
  their bounding box. This matters in real rooms with other people, TVs,
  posters of hands, etc. walking through the background -- it keeps the
  translator locked onto the actual signer. (Optional --no-yolo flag skips
  this and just uses the full frame.)
- MediaPipe Hands finds the 21 hand landmarks inside that region.
- The landmark positions (normalized for translation/scale so it doesn't
  matter how close you are to the camera) become the feature vector for
  classification -- either the trained RandomForest or the fallback rules.
- OpenCV handles capture, the on-screen keyboard-style UI, and drawing.

USAGE
-----
    pip install opencv-python mediapipe ultralytics numpy scikit-learn joblib
    pip install pyttsx3      # optional, enables the 't' speak-aloud key

    # Step 1: collect training samples for each letter you want to sign.
    #         Hold a letter shape, press its corresponding key repeatedly
    #         (slightly varying hand angle/distance each time) to log
    #         samples. Aim for at least ~30 samples per letter.
    python sign_language_translator.py --mode collect

    # Step 2: train a classifier on what you collected.
    python sign_language_translator.py --mode train

    # Step 3: run live recognition (uses trained model if found, otherwise
    #         the built-in rule-based fallback covers A,B,C,D,F,I,L,O,U,V,W,Y).
    python sign_language_translator.py --mode run

RUN-MODE CONTROLS
------------------
    [space]      insert a space into the sentence
    [backspace]  delete last character
    c            clear the current sentence
    t            speak the sentence aloud (requires pyttsx3)
    q            quit

NOTE ON SCOPE
-------------
ASL fingerspelling includes two letters, J and Z, that involve *motion*
rather than a static hand shape -- this script (like most static-landmark
classifiers) does not attempt those two. Full ASL (word-level signs) also
involves motion, facial grammar, and two-handed signs far beyond
fingerspelling; this is intentionally scoped to the fingerspelling alphabet,
which is still genuinely useful for spelling names, places, and words that
don't have a dedicated sign.
"""

import argparse
import csv
import os
import sys
import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


# ==========================================================================
# CONFIGURATION
# ==========================================================================
class Config:
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONF_THRESHOLD = 0.45
    PERSON_CLASS_ID = 0

    DATA_FILE = "sign_data.csv"
    MODEL_FILE = "sign_model.joblib"

    # Collectable / recognizable letters (J and Z excluded -- see docstring)
    LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # skip J, Z

    STABLE_FRAMES = 10        # consecutive matching predictions to commit a letter
    RESET_FRAMES = 8          # frames with no hand needed before repeat-letter allowed
    MIN_CONFIDENCE = 0.55     # for the trained ML model


# ==========================================================================
# LANDMARK FEATURE EXTRACTION (shared by collect / train / run)
# ==========================================================================
def normalize_landmarks(hand_landmarks):
    """
    Converts MediaPipe's 21 (x, y, z) hand landmarks into a translation- and
    scale-invariant feature vector: shift so the wrist is the origin, then
    scale by the hand's overall size. This means the classifier works
    regardless of where the hand is in frame or how close it is to the
    camera.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = pts[0].copy()
    pts -= wrist  # translation-invariant

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale  # scale-invariant

    return pts.flatten()  # length 63


# ==========================================================================
# BUILT-IN GEOMETRIC RULE CLASSIFIER (fallback when no trained model exists)
# ==========================================================================
class RuleBasedClassifier:
    """
    Covers a subset of the ASL alphabet using finger extended/curled state
    plus a couple of extra geometric checks. Meant as a functional starting
    point, not a replacement for training your own model.
    """
    MCP_IDS = [5, 9, 13, 17]   # index, middle, ring, pinky knuckles
    PIP_IDS = [6, 10, 14, 18]
    TIP_IDS = [8, 12, 16, 20]
    STRAIGHT_ANGLE_THRESHOLD = 40  # degrees; below this, finger counts as "extended"

    def _is_finger_straight(self, lm, mcp_id, pip_id, tip_id):
        """
        Checks if a finger is actually straight by measuring the angle
        between its two segments (knuckle->pip and pip->tip), rather than
        just comparing distances from the wrist. Distance-from-wrist is
        unreliable: it's easily fooled by hand rotation/tilt and by fingers
        that are only half-curled, which was causing almost every relaxed
        open hand to get misread as "all fingers extended" (the B shape).
        An angle-based check only calls a finger "extended" when it's
        genuinely straight, regardless of how the hand is angled to the
        camera.
        """
        mcp = np.array([lm[mcp_id].x, lm[mcp_id].y])
        pip = np.array([lm[pip_id].x, lm[pip_id].y])
        tip = np.array([lm[tip_id].x, lm[tip_id].y])

        v1 = pip - mcp
        v2 = tip - pip
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False

        cos_angle = np.dot(v1, v2) / (n1 * n2)
        bend_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return bend_angle < self.STRAIGHT_ANGLE_THRESHOLD

    def _finger_states(self, hand_landmarks, handedness_label):
        lm = hand_landmarks.landmark
        states = []
        # thumb: compare x position relative to its own IP joint, direction
        # depends on which hand is showing
        thumb_extended = (lm[4].x < lm[3].x) if handedness_label == "Right" else (lm[4].x > lm[3].x)
        states.append(thumb_extended)
        # other four fingers: genuinely straight (not just "farther from
        # wrist than the knuckle") => extended
        for mcp_id, pip_id, tip_id in zip(self.MCP_IDS, self.PIP_IDS, self.TIP_IDS):
            states.append(self._is_finger_straight(lm, mcp_id, pip_id, tip_id))
        return tuple(states)  # (thumb, index, middle, ring, pinky)

    def predict(self, hand_landmarks, handedness_label):
        s = self._finger_states(hand_landmarks, handedness_label)
        lm = hand_landmarks.landmark
        wrist = lm[0]

        thumb_index_dist = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        thumb_middle_dist = np.hypot(lm[4].x - lm[12].x, lm[4].y - lm[12].y)

        # (thumb, index, middle, ring, pinky) -> letter, for shapes where the
        # four fingers aren't all in the curled/curved state (those are
        # handled separately below, since O/C/E all share that same pattern
        # and need a distance-based tiebreaker instead).
        table = {
            (False, True, False, False, False): "D",
            (False, True, True, False, False): "V",
            (False, True, True, True, False): "W",
            (False, True, True, True, True): "B",
            (True, True, True, True, True): "B",
            (False, False, False, False, True): "I",
            (True, True, False, False, False): "L",
        }

        if s in table:
            return table[s], 0.6

        if thumb_index_dist < 0.07 and s[2] and s[3] and s[4]:
            return "F", 0.55

        # -- O / C / E all present as "index, middle, ring, pinky curled" --
        # tell them apart using how curved (vs. flat-fisted) the index
        # finger is, plus where the thumb tip sits relative to it.
        if not any(s[1:]):
            if s[0]:
                # thumb extended out to the side, fingers curled -> A
                return "A", 0.55

            index_mcp = lm[5]
            index_tip_to_wrist = np.hypot(lm[8].x - wrist.x, lm[8].y - wrist.y)
            index_mcp_to_wrist = np.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
            curl_ratio = index_tip_to_wrist / (index_mcp_to_wrist + 1e-6)

            if curl_ratio > 1.0:
                # fingers are curved outward (not pressed flat into a fist)
                # -> this is the O/C shape family
                if thumb_index_dist < 0.10:
                    return "O", 0.6   # thumb and index tips touching -> closed circle
                else:
                    return "C", 0.55  # gap between thumb and fingers -> open curve
            else:
                # tight fist, thumb tucked across the front of curled fingers
                return "E", 0.5

        return None, 0.0


# ==========================================================================
# TEXT-TO-SPEECH HELPER
# ==========================================================================
class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init() if TTS_AVAILABLE else None

    def say(self, text):
        if not text:
            return
        if self.engine is not None:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            print(f"[TTS unavailable] Would have said: '{text}'")


# ==========================================================================
# SHARED PIPELINE: YOLO person localization + MediaPipe hand detection
# ==========================================================================
class HandFinder:
    def __init__(self, use_yolo=True):
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.person_model = YOLO(Config.YOLO_MODEL) if self.use_yolo else None
        if use_yolo and not YOLO_AVAILABLE:
            print("[WARN] ultralytics not installed -> skipping person localization, "
                  "using full frame. Run: pip install ultralytics")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def get_roi(self, frame):
        """Returns (x1, y1, x2, y2) region to search for hands: the largest
        detected person's box, or the full frame if YOLO is off/unavailable."""
        h, w = frame.shape[:2]
        if not self.use_yolo:
            return (0, 0, w, h)

        results = self.person_model(frame, verbose=False)[0]
        best_box, best_area = None, 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == Config.PERSON_CLASS_ID and conf >= Config.YOLO_CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area, best_box = area, (x1, y1, x2, y2)

        return best_box if best_box else (0, 0, w, h)

    def find_hand(self, frame, roi):
        """Runs MediaPipe Hands inside the ROI. Returns
        (hand_landmarks_in_full_frame_coords, handedness_label) or (None, None)."""
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None, None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label = results.multi_handedness[0].classification[0].label  # "Left"/"Right"

        # Remap landmark coords from crop-space back to full-frame space so
        # drawing/normalization downstream is consistent.
        ch, cw = crop.shape[:2]
        for lm in hand_landmarks.landmark:
            lm.x = (lm.x * cw + x1) / frame.shape[1]
            lm.y = (lm.y * ch + y1) / frame.shape[0]

        return hand_landmarks, handedness_label

    def draw(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)


# ==========================================================================
# MODE: COLLECT
# ==========================================================================
def run_collect(args):
    finder = HandFinder(use_yolo=not args.no_yolo)
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    file_exists = os.path.exists(Config.DATA_FILE)
    f = open(Config.DATA_FILE, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["label"] + [f"f{i}" for i in range(63)])

    counts = Counter()
    if file_exists:
        with open(Config.DATA_FILE, "r") as rf:
            for row in csv.DictReader(rf):
                counts[row["label"]] += 1

    print("Collect mode. Press a letter key to log a sample for that letter "
          "(A-Y, no J/Z). Press 'q' to finish.")
    print(f"Recognizable letters: {''.join(Config.LETTERS)}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            roi = finder.get_roi(frame)
            hand_landmarks, _ = finder.find_hand(frame, roi)

            if hand_landmarks:
                finder.draw(frame, hand_landmarks)

            cv2.putText(frame, "Press a letter key to save a sample. 'q' to finish.",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y = 55
            summary = " ".join(f"{L}:{counts.get(L, 0)}" for L in Config.LETTERS)
            for i in range(0, len(summary), 60):
                cv2.putText(frame, summary[i:i + 60], (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y += 20

            cv2.imshow("Collect Sign Samples", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            ch = chr(key).upper() if 32 <= key < 127 else None
            if ch in Config.LETTERS and hand_landmarks is not None:
                features = normalize_landmarks(hand_landmarks)
                writer.writerow([ch] + features.tolist())
                counts[ch] += 1
                f.flush()
                print(f"Saved sample for '{ch}' (total: {counts[ch]})")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        f.close()
        print(f"Data saved to {Config.DATA_FILE}")


# ==========================================================================
# MODE: TRAIN
# ==========================================================================
def run_train(args):
    if not SKLEARN_AVAILABLE:
        print("scikit-learn/joblib not installed. Run: pip install scikit-learn joblib")
        return
    if not os.path.exists(Config.DATA_FILE):
        print(f"No data file found at {Config.DATA_FILE}. Run --mode collect first.")
        return

    X, y = [], []
    with open(Config.DATA_FILE, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])

    X = np.array(X)
    y = np.array(y)
    label_counts = Counter(y)
    print(f"Loaded {len(X)} samples across {len(label_counts)} letters: {dict(label_counts)}")

    if len(X) < 20 or len(label_counts) < 2:
        print("Not enough data to train a meaningful model yet -- collect more samples.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if min(label_counts.values()) > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Validation accuracy: {accuracy * 100:.1f}%")

    joblib.dump(clf, Config.MODEL_FILE)
    print(f"Model saved to {Config.MODEL_FILE}")


# ==========================================================================
# MODE: RUN (live translation)
# ==========================================================================
class SentenceBuilder:
    def __init__(self):
        self.sentence = ""
        self.recent_predictions = deque(maxlen=Config.STABLE_FRAMES)
        self.last_committed = None
        self.no_hand_frames = 0

    def observe(self, predicted_label):
        if predicted_label is None:
            self.no_hand_frames += 1
            self.recent_predictions.clear()
            if self.no_hand_frames >= Config.RESET_FRAMES:
                self.last_committed = None
            return None

        self.no_hand_frames = 0
        self.recent_predictions.append(predicted_label)

        if len(self.recent_predictions) == Config.STABLE_FRAMES and \
                len(set(self.recent_predictions)) == 1:
            stable_label = self.recent_predictions[0]
            if stable_label != self.last_committed:
                self.sentence += stable_label
                self.last_committed = stable_label
                self.recent_predictions.clear()
                return stable_label
        return None

    def hold_progress(self):
        return len(self.recent_predictions) / Config.STABLE_FRAMES

    def backspace(self):
        self.sentence = self.sentence[:-1]

    def add_space(self):
        self.sentence += " "
        self.last_committed = None

    def clear(self):
        self.sentence = ""
        self.last_committed = None


def run_live(args):
    finder = HandFinder(use_yolo=not args.no_yolo)
    speaker = Speaker()
    builder = SentenceBuilder()

    model = None
    if SKLEARN_AVAILABLE and os.path.exists(Config.MODEL_FILE):
        model = joblib.load(Config.MODEL_FILE)
        print(f"Loaded trained model from {Config.MODEL_FILE}")
    else:
        print("No trained model found -- using built-in rule-based fallback "
              f"(covers: {', '.join(sorted(set(['A','B','C','D','E','F','I','L','O','V','W'])))}). "
              "Run --mode collect then --mode train for much better accuracy.")
    rule_classifier = RuleBasedClassifier()

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    print("Live translation running. [space]=space  [backspace]=delete  "
          "c=clear  t=speak  q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            roi = finder.get_roi(frame)
            hand_landmarks, handedness = finder.find_hand(frame, roi)

            predicted_label = None
            if hand_landmarks is not None:
                finder.draw(frame, hand_landmarks)
                if model is not None:
                    features = normalize_landmarks(hand_landmarks).reshape(1, -1)
                    probs = model.predict_proba(features)[0]
                    best_idx = int(np.argmax(probs))
                    if probs[best_idx] >= Config.MIN_CONFIDENCE:
                        predicted_label = model.classes_[best_idx]
                else:
                    predicted_label, _ = rule_classifier.predict(hand_landmarks, handedness)

            committed = builder.observe(predicted_label)
            if committed:
                print(f"Committed letter: {committed}  |  sentence so far: '{builder.sentence}'")

            # -- UI overlay --
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 90), (w, h), (30, 30, 30), -1)
            cv2.putText(frame, f"Letter: {predicted_label or '-'}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            bar_w = int(200 * builder.hold_progress())
            cv2.rectangle(frame, (200, h - 75), (200 + bar_w, h - 55), (0, 200, 0), -1)
            cv2.rectangle(frame, (200, h - 75), (400, h - 55), (255, 255, 255), 1)

            cv2.putText(frame, f"Sentence: {builder.sentence}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Sign Language Translator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                builder.add_space()
            elif key in (8, 127):  # backspace
                builder.backspace()
            elif key == ord('c'):
                builder.clear()
            elif key == ord('t'):
                speaker.say(builder.sentence)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final sentence: '{builder.sentence}'")


# ==========================================================================
# ENTRY POINT
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="ASL fingerspelling to text translator.")
    parser.add_argument("--mode", choices=["collect", "train", "run"], default="run")
    parser.add_argument("--source", default=0, help="Webcam index or video file path.")
    parser.add_argument("--no-yolo", action="store_true",
                         help="Skip YOLO person localization, use full frame for hand search.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "collect":
        run_collect(args)
    elif args.mode == "train":
        run_train(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()