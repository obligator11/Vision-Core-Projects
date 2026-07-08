

import argparse
import time

import cv2

from pose_detector import PoseDetector
from exercise_analyzer import ExerciseAnalyzer
from report_generator import save_session_report

# ---------------------------------------------------------------------
# UI drawing helpers
# ---------------------------------------------------------------------
PANEL_COLOR = (20, 20, 20)
ACCENT = (76, 154, 255)      # BGR
GOOD_COLOR = (86, 179, 54)
WARN_COLOR = (48, 86, 255)


def draw_dashboard(frame, analyzer, result, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    exercise = result.get("exercise", "Detecting...")
    cv2.putText(frame, f"Exercise: {exercise}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2)

    if exercise == "Plank":
        cv2.putText(frame, f"Hold: {result.get('hold_seconds', 0)}s", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"Reps: {result.get('reps', analyzer.counters.get(exercise, 0))}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    total = analyzer.total_reps()
    cv2.putText(frame, f"Total reps: {total}   Form score: {analyzer.form_score()}%",
                (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (350, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    feedback = result.get("feedback", "")
    if feedback:
        is_warning = any(w in feedback.lower() for w in
                          ["don't", "push", "keep", "straighten", "avoid"])
        color = WARN_COLOR if is_warning else GOOD_COLOR
        cv2.rectangle(frame, (0, h - 55), (w, h), PANEL_COLOR, -1)
        cv2.putText(frame, feedback, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, "[q] quit  [r] reset  [space] pause", (20, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return frame


# ---------------------------------------------------------------------
def run(camera_index=0, video_path=None):
    source = video_path if video_path else camera_index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    detector = PoseDetector()
    analyzer = ExerciseAnalyzer()

    prev_time = time.time()
    paused = False
    session_start = time.time()

    print("AI Fitness Trainer running. Press 'q' in the video window to finish and save your report.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1) if video_path is None else frame

            frame = detector.find_pose(frame, draw=True)
            landmarks = detector.get_landmarks(frame)
            result = analyzer.analyze(landmarks)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            frame = draw_dashboard(frame, analyzer, result, fps)
            cv2.imshow("AI Fitness Trainer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            analyzer.counters = {k: 0 for k in analyzer.counters}
            analyzer.plank_seconds = 0.0
            analyzer.form_events = []
            print("Counters reset.")
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    duration = time.time() - session_start
    info = save_session_report(
        analyzer.counters, analyzer.plank_seconds, analyzer.form_score(), duration
    )
    print("\n===== SESSION COMPLETE =====")
    print(f"Duration        : {duration:.1f}s")
    print(f"Reps            : {analyzer.counters}")
    print(f"Plank hold      : {analyzer.plank_seconds:.1f}s")
    print(f"Form score      : {analyzer.form_score()}%")
    print(f"Estimated kcal  : {info['calories']}")
    print(f"CSV log saved   : {info['csv_path']}")
    print(f"Chart saved     : {info['chart_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fitness Trainer & Posture Corrector")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file instead of webcam")
    args = parser.parse_args()

    run(camera_index=args.camera, video_path=args.video)