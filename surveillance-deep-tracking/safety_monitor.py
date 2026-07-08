

import argparse
import csv
import os
import sys
import time
import math
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==========================================================================
# CONFIGURATION
# ==========================================================================
class Config:
    # -- Person detection --
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONF_THRESHOLD = 0.45
    PERSON_CLASS_ID = 0

    # -- PPE detection (custom-model mode) --
    PPE_CONF_THRESHOLD = 0.40
    # Class-name substrings (case-insensitive) that indicate a VIOLATION when
    # produced by a custom PPE model. Matches the common public
    # "Hard Hat Workers" dataset naming convention out of the box.
    VIOLATION_KEYWORDS = ["no-hardhat", "no-helmet", "no-vest", "no-safety"]

    # -- Tracking / temporal smoothing --
    MAX_MATCH_DISTANCE = 120
    TRACK_TIMEOUT = 1.5
    CONSEC_FRAMES_TO_CONFIRM = 5
    ALERT_COOLDOWN_SEC = 10.0

    # -- Output --
    OUTPUT_DIR = "safety_events"
    LOG_FILE = "safety_log.csv"

    # -- Display --
    SHOW_SKELETON = False
    SHOW_FPS = True
    ZONE_COLOR = (0, 0, 255)
    ZONE_ALPHA = 0.25


# ==========================================================================
# UTILITIES
# ==========================================================================
def beep():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1500, 250)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    log_path = os.path.join(path, Config.LOG_FILE)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "violation_type",
                              "detail", "snapshot_file"])
    return log_path


def parse_zone_string(zone_str):
    """'x1,y1;x2,y2;...' -> [(x1,y1), (x2,y2), ...]"""
    points = []
    for pair in zone_str.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        points.append((int(x_str), int(y_str)))
    return points


def point_in_polygon(point, polygon):
    if polygon is None or len(polygon) < 3:
        return False
    poly_np = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(poly_np, (float(point[0]), float(point[1])), False)
    return result >= 0


# ==========================================================================
# PPE HEURISTIC DETECTOR (color-based fallback when no trained model given)
# ==========================================================================
class PPEHeuristicDetector:
    """
    Looks for hi-vis vest colors on the torso region and hard-hat colors on
    the head region using HSV thresholding. Not as accurate as a trained
    model, but requires zero extra downloads and works reasonably well
    against plain-clothes backgrounds.
    """

    # Hi-vis vest: safety orange, safety yellow/lime
    VEST_HSV_RANGES = [
        ((10, 120, 120), (25, 255, 255)),    # orange
        ((25, 100, 120), (35, 255, 255)),    # yellow/lime
    ]
    # Hard hats: commonly white, yellow, orange, red, blue -- we check for
    # any strongly saturated "hat-like" color blob in the head region.
    HELMET_HSV_RANGES = [
        ((25, 100, 150), (35, 255, 255)),    # yellow
        ((10, 120, 150), (25, 255, 255)),    # orange
        ((0, 0, 180), (180, 40, 255)),       # white
    ]

    def _color_fraction(self, hsv_region, ranges):
        if hsv_region.size == 0:
            return 0.0
        mask_total = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_region, np.array(lower), np.array(upper))
            mask_total = cv2.bitwise_or(mask_total, mask)
        return float(np.count_nonzero(mask_total)) / mask_total.size

    def check(self, person_crop_bgr):
        """Returns (has_vest: bool, has_helmet: bool)."""
        h, w = person_crop_bgr.shape[:2]
        if h < 20 or w < 20:
            return True, True  # too small to judge reliably -> don't flag

        head_region = person_crop_bgr[0:int(h * 0.2), :]
        torso_region = person_crop_bgr[int(h * 0.2):int(h * 0.6), :]

        head_hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV) if head_region.size else None
        torso_hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV) if torso_region.size else None

        vest_frac = self._color_fraction(torso_hsv, self.VEST_HSV_RANGES) if torso_hsv is not None else 0.0
        helmet_frac = self._color_fraction(head_hsv, self.HELMET_HSV_RANGES) if head_hsv is not None else 0.0

        has_vest = vest_frac > 0.12
        has_helmet = helmet_frac > 0.10
        return has_vest, has_helmet


# ==========================================================================
# PER-PERSON TRACK
# ==========================================================================
class PersonTrack:
    _next_id = 0

    def __init__(self, centroid):
        self.id = PersonTrack._next_id
        PersonTrack._next_id += 1
        self.last_seen = time.time()
        self.last_centroid = centroid

        # violation persistence counters
        self.zone_violation_frames = 0
        self.ppe_violation_frames = 0

        # alert cooldowns (separate per violation type)
        self.last_zone_alert = 0.0
        self.last_ppe_alert = 0.0

        # flash flags for display
        self.flash_zone_until = 0.0
        self.flash_ppe_until = 0.0

    def update(self, centroid):
        self.last_centroid = centroid
        self.last_seen = time.time()


class CentroidTracker:
    def __init__(self):
        self.tracks = {}

    def update(self, centroids):
        assigned = []
        used_ids = set()

        for centroid in centroids:
            best_id, best_dist = None, Config.MAX_MATCH_DISTANCE
            for tid, track in self.tracks.items():
                if tid in used_ids:
                    continue
                dist = math.hypot(centroid[0] - track.last_centroid[0],
                                   centroid[1] - track.last_centroid[1])
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None:
                track = self.tracks[best_id]
                track.update(centroid)
                used_ids.add(best_id)
            else:
                track = PersonTrack(centroid)
                self.tracks[track.id] = track
                used_ids.add(track.id)

            assigned.append(track)

        now = time.time()
        stale = [tid for tid, t in self.tracks.items()
                 if now - t.last_seen > Config.TRACK_TIMEOUT]
        for tid in stale:
            del self.tracks[tid]

        return assigned


# ==========================================================================
# ZONE DEFINITION (interactive mouse-click setup)
# ==========================================================================
class ZoneDefiner:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.points = []
        self.done = False

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

    def run(self):
        window = "Define restricted zone: click points, 'c'=confirm, 'r'=reset, 'q'=skip"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_mouse)

        while True:
            display = self.frame.copy()
            for p in self.points:
                cv2.circle(display, p, 5, (0, 0, 255), -1)
            if len(self.points) > 1:
                cv2.polylines(display, [np.array(self.points, dtype=np.int32)],
                               isClosed=False, color=(0, 0, 255), thickness=2)
            cv2.imshow(window, display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('c') and len(self.points) >= 3:
                self.done = True
                break
            elif key == ord('r'):
                self.points = []
            elif key == ord('q'):
                self.points = []
                break

        cv2.destroyWindow(window)
        return self.points if self.done else None


# ==========================================================================
# MAIN SYSTEM
# ==========================================================================
class WorkplaceSafetyMonitor:
    def __init__(self, source=0, display=True, zone=None, ppe_model_path=None):
        self.display = display

        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics is required for person detection. "
                                "Install with: pip install ultralytics")

        self.person_model = YOLO(Config.YOLO_MODEL)
        self.ppe_model = YOLO(ppe_model_path) if ppe_model_path else None
        self.heuristic_ppe = PPEHeuristicDetector() if self.ppe_model is None else None

        self.tracker = CentroidTracker()
        self.zone = zone  # list of (x, y) or None
        self.log_path = ensure_output_dir(Config.OUTPUT_DIR)

        self.prev_time = time.time()
        self.fps = 0.0

    # ----------------------------------------------------------------
    def detect_people(self, frame):
        results = self.person_model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == Config.PERSON_CLASS_ID and conf >= Config.YOLO_CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        return boxes

    # ----------------------------------------------------------------
    def get_foot_point(self, frame, bbox):
        """Runs MediaPipe Pose on the person crop and returns the averaged
        ankle position in full-frame coordinates. Falls back to the bbox
        bottom-center if pose estimation fails."""
        x1, y1, x2, y2 = bbox
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
        fallback = ((x1 + x2) / 2, y2)

        if x2c <= x1c or y2c <= y1c:
            return fallback, None

        crop = frame[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return fallback, None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            return fallback, None

        lm = results.pose_landmarks.landmark
        ch, cw = crop.shape[:2]
        l_ankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        fx = x1c + ((l_ankle.x + r_ankle.x) / 2) * cw
        fy = y1c + ((l_ankle.y + r_ankle.y) / 2) * ch
        return (fx, fy), results.pose_landmarks

    # ----------------------------------------------------------------
    def check_ppe_heuristic(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        has_vest, has_helmet = self.heuristic_ppe.check(crop)
        missing = []
        if not has_helmet:
            missing.append("no-hardhat")
        if not has_vest:
            missing.append("no-vest")
        return missing

    # ----------------------------------------------------------------
    def check_ppe_model(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        results = self.ppe_model(crop, verbose=False)[0]
        missing = []
        names = results.names
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < Config.PPE_CONF_THRESHOLD:
                continue
            cls_name = names[int(box.cls[0])].lower()
            if any(kw in cls_name for kw in Config.VIOLATION_KEYWORDS):
                missing.append(cls_name)
        return missing

    # ----------------------------------------------------------------
    def raise_alert(self, frame, track, violation_type, detail):
        now = time.time()
        last_alert = track.last_zone_alert if violation_type == "ZONE" else track.last_ppe_alert
        if now - last_alert < Config.ALERT_COOLDOWN_SEC:
            return
        if violation_type == "ZONE":
            track.last_zone_alert = now
            track.flash_zone_until = now + 2.0
        else:
            track.last_ppe_alert = now
            track.flash_ppe_until = now + 2.0

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snapshot_name = f"violation_{track.id}_{violation_type}_{timestamp}.jpg"
        snapshot_path = os.path.join(Config.OUTPUT_DIR, snapshot_name)
        cv2.imwrite(snapshot_path, frame)

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, track.id, violation_type, detail, snapshot_name])

        beep()
        print(f"[ALERT] {violation_type} violation -> person_id={track.id} "
              f"detail={detail} -> saved {snapshot_path}")

    # ----------------------------------------------------------------
    def draw_zone(self, frame):
        if not self.zone or len(self.zone) < 3:
            return
        overlay = frame.copy()
        pts = np.array(self.zone, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], Config.ZONE_COLOR)
        cv2.addWeighted(overlay, Config.ZONE_ALPHA, frame, 1 - Config.ZONE_ALPHA, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=Config.ZONE_COLOR, thickness=2)
        cv2.putText(frame, "RESTRICTED ZONE", (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.ZONE_COLOR, 2)

    # ----------------------------------------------------------------
    def draw_person(self, frame, track, bbox, foot_point, zone_flag, ppe_missing, landmarks):
        x1, y1, x2, y2 = bbox
        now = time.time()
        in_zone_flash = now < track.flash_zone_until
        in_ppe_flash = now < track.flash_ppe_until

        color = (0, 200, 0)
        labels = [f"ID {track.id}"]
        if in_zone_flash:
            color = (0, 0, 255)
            labels.append("ZONE VIOLATION")
        if in_ppe_flash:
            color = (0, 0, 255)
            labels.append(f"PPE MISSING: {', '.join(ppe_missing) if ppe_missing else '?'}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        y_text = max(20, y1 - 10)
        for i, text in enumerate(labels):
            cv2.putText(frame, text, (x1, y_text - i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if foot_point:
            fp_color = (0, 0, 255) if zone_flag else (255, 200, 0)
            cv2.circle(frame, (int(foot_point[0]), int(foot_point[1])), 6, fp_color, -1)

        if Config.SHOW_SKELETON and landmarks:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            sub = frame[y1c:y2c, x1c:x2c]
            if sub.size > 0:
                self.mp_drawing.draw_landmarks(sub, landmarks, self.mp_pose.POSE_CONNECTIONS)

    # ----------------------------------------------------------------
    def update_fps(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

    # ----------------------------------------------------------------
    def run(self):
        print("Workplace safety monitor running. Press 'q' to quit.")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("End of stream / camera read failed.")
                    break

                bboxes = self.detect_people(frame)
                centroids = [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1, x2, y2) in bboxes]
                tracks = self.tracker.update(centroids)

                for track, bbox in zip(tracks, bboxes):
                    foot_point, landmarks = self.get_foot_point(frame, bbox)

                    # -- zone check --
                    zone_flag = point_in_polygon(foot_point, self.zone)
                    if zone_flag:
                        track.zone_violation_frames += 1
                    else:
                        track.zone_violation_frames = max(0, track.zone_violation_frames - 1)
                    if track.zone_violation_frames >= Config.CONSEC_FRAMES_TO_CONFIRM:
                        self.raise_alert(frame, track, "ZONE", "entered restricted zone")

                    # -- PPE check --
                    if self.ppe_model is not None:
                        missing = self.check_ppe_model(frame, bbox)
                    else:
                        missing = self.check_ppe_heuristic(frame, bbox)

                    if missing:
                        track.ppe_violation_frames += 1
                    else:
                        track.ppe_violation_frames = max(0, track.ppe_violation_frames - 1)
                    if track.ppe_violation_frames >= Config.CONSEC_FRAMES_TO_CONFIRM:
                        self.raise_alert(frame, track, "PPE", ", ".join(missing) if missing else "unknown")

                    self.draw_person(frame, track, bbox, foot_point, zone_flag, missing, landmarks)

                self.draw_zone(frame)

                self.update_fps()
                if Config.SHOW_FPS:
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if self.display:
                    cv2.imshow("Workplace Safety Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print(f"Stopped. Events logged in {self.log_path}")


# ==========================================================================
# ENTRY POINT
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Workplace PPE & restricted-zone safety monitor.")
    parser.add_argument("--source", default=0,
                         help="Webcam index (e.g. 0) or path to a video file.")
    parser.add_argument("--no-display", action="store_true",
                         help="Run headless (no GUI window).")
    parser.add_argument("--zone", default=None,
                         help='Restricted zone polygon as "x1,y1;x2,y2;x3,y3;...".')
    parser.add_argument("--define-zone", action="store_true",
                         help="Interactively click a restricted zone on the first frame.")
    parser.add_argument("--no-zone", action="store_true",
                         help="Disable zone checking entirely (PPE-only monitoring).")
    parser.add_argument("--ppe-model", default=None,
                         help="Path to a custom-trained YOLO PPE model weights file. "
                              "If omitted, uses a color-based heuristic instead.")
    return parser.parse_args()


def main():
    args = parse_args()

    zone = None
    if args.zone:
        zone = parse_zone_string(args.zone)

    cap_probe = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if args.define_zone and not args.no_zone:
        ok, first_frame = cap_probe.read()
        if ok:
            zone = ZoneDefiner(first_frame).run()
        cap_probe.release()
        # reopen fresh capture for the real run (webcams especially need this)
    else:
        cap_probe.release()

    if args.no_zone:
        zone = None

    monitor = WorkplaceSafetyMonitor(
        source=args.source,
        display=not args.no_display,
        zone=zone,
        ppe_model_path=args.ppe_model,
    )
    monitor.run()


if __name__ == "__main__":
    main()