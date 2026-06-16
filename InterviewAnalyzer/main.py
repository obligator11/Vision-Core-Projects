import cv2
import time
from core.video_stream import VideoStream
from core.face_analysis import FaceAnalyzer
from core.posture_analysis import PostureAnalyzer
from core.audio_nlp_engine import AudioNLPEngine
from core.confidence_engine import ConfidenceEngine
from audio.sound_manager import SoundManager
from ui.dashboard_overlay import DashboardOverlay

def run_pipeline():
    # Direct target hook link to your mock video file source path
    camera_stream = VideoStream(source="mock_interview.mp4").start()
    
    face_engine = FaceAnalyzer()
    body_engine = PostureAnalyzer()
    audio_nlp_engine = AudioNLPEngine(model_size="base").start()
    
    metrics_engine = ConfidenceEngine()
    audio_queue = SoundManager()
    hud_overlay = DashboardOverlay()

    window_title = "Smart Interview & Confidence Analyzer"
    
    # UNLOCK WINDOW RESIZING: Instantly enables drag scale updates
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    print("\n[SYSTEM RUNNING] Absolute orientation matrix active. Press 'q' to exit.\n")

    while True:
        success, current_frame = camera_stream.read()
        if not success or current_frame is None:
            time.sleep(0.005)
            continue

        # Fetch structural frame layers data
        face_payload = face_engine.analyze(current_frame)
        posture_payload = body_engine.analyze(current_frame)
        audio_payload = audio_nlp_engine.get_metrics()
        
        # Process composite matrix combinations
        performance_results = metrics_engine.process(face_payload, posture_payload, audio_payload)

        # Trigger acoustic notification notes
        current_rating = performance_results["rating"]
        if current_rating == "EXCELLENT_CONFIDENCE":
            audio_queue.trigger_feedback("HIGH_CONFIDENCE")
        elif current_rating == "AVERAGE_CONFIDENCE":
            audio_queue.trigger_feedback("MEDIUM_CONFIDENCE")
        elif current_rating == "NERVOUS_BEHAVIOR":
            audio_queue.trigger_feedback("LOW_CONFIDENCE")

        # Render integrated interface updates onto image canvas
        annotated_frame = hud_overlay.draw(
            frame=current_frame, 
            face_data=face_payload, 
            posture_data=posture_payload, 
            audio_data=audio_payload, 
            score_payload=performance_results
        )

        cv2.imshow(window_title, annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[SYSTEM CLOSING] Releasing multi-modal processor threads cleanly...")
    camera_stream.stop()
    audio_nlp_engine.stop()
    audio_queue.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()