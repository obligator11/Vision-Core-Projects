import cv2
import config 
from core.video_processor import VideoProcessor
from core.motion_analyzer import MotionAnalyzer
from core.anomaly_detector import AnomalyDetecter

def test_pipeline():
    print("=== Starting Live Stream Test Pipeline ===")

    processor = VideoProcessor()
    analyzer = MotionAnalyzer()
    detector = anomaly_detector

    if not processor.initialize_stream():
        print(f"[CRITICAL] Could not read video file asset: {config.VIDEO_SOURCE}")
        return

    processor.load_model()


    window_name = "Sensorless PdM System Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    print("\n[SUCCESS] Pipeline running smoothly.")
    print("Press 'ESC' to safely stop execution.")
    while True:
        success, frame = processor.get_frame()
        if not success:
            print("[INFO] Video track ended or frame dropped. Restarting playback loop...")
            processor.initialize_stream()
            continue

        vibration_score = analyzer.calculate_vibration_intensity(frame)
        is_anomaly, anomaly_score = detector.evaluate_value(vibration_score)

        vib_text = f"Simulation Vibration: {vibration_score:.2f} Hz/Units"
        anomaly_text  =f"Anomaly Flag: {is_anomaly} (Score: {anomaly_score:.2f})"

        alert_color = (0, 0, 255) if is_anomaly else (255, 255, 0)
        
        cv2.putText(frame, vib_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, anomaly_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, alert_color, 2)
        
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(20) & 0xFF == 27:


    processor.release_stream()
    print("=== Pipeline safely shutdown===")

if __name__ == "__main__":
    test_pipeline()

