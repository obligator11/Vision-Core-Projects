"""
SafetyAI - Dataset Video Stream Pipeline Coordinator with State Reset Controls
"""
import os
import sys
import time
import cv2
import numpy as np

from core.pose_detector import PoseDetector
from core.motion_tracker import MotionTracker
from core.safety_engine import SafetyEngine
from audio.sound_manager import SoundManager
from ui.safety_overlay import SafetyOverlay

VIDEO_DATASET_PATH = "sample_worker_feed.mp4" 

def initialize_safety_system():
    print("==========================================================")
    print("INITIALIZING: AI Lone Worker Safety Monitor (Dataset Feed)")
    print("==========================================================")
    
    if not os.path.exists(VIDEO_DATASET_PATH):
        print(f"[CRITICAL ERROR]: Target video file not found at path: '{VIDEO_DATASET_PATH}'")
        sys.exit(1)
        
    cap = cv2.VideoCapture(VIDEO_DATASET_PATH)
    if not cap.isOpened():
        print(f"[CRITICAL ERROR]: Could not parse video stream.")
        sys.exit(1)
        
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0  
    frame_delay = 1.0 / native_fps
    
    window_title = "AI Personal Safety Monitoring System (Lone Worker Detection)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL) 
    
    detector = PoseDetector()
    tracker = MotionTracker(buffer_window_size=30)
    engine = SafetyEngine()
    audio = SoundManager()
    ui = SafetyOverlay()
    
    print("[SUCCESS]: Safety-Critical Pipeline Connected. Resets enabled on loop.")
    
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            
            # --- DETECT REWIND / VIDEO RESTART LOOP ---
            if not ret or frame is None:
                print("[LOOP RESET]: Rewinding video feed. Wiping state engine registers...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # EXECUTE TRACKING RESETS HERE
                tracker.reset()
                engine.reset()
                audio.set_risk_level("SAFE")
                continue
                
            h, w, _ = frame.shape
            
            # Phase A: Inference Engine Parsing
            raw_landmarks, full_results = detector.process_frame(frame)
            absolute_points = detector.extract_key_vectors(raw_landmarks, w, h)
            
            # Phase B: Tracking Array Telemetry Updates
            motion_metrics = tracker.update(absolute_points)
            
            # Phase C: State Machine Execution Decisions
            score, risk, anomaly = engine.evaluate_states(absolute_points, motion_metrics, w, h)
            
            # Phase D: Thread-Safe Audio Alert Update Injection
            audio.set_risk_level(risk)
            
            # Phase E: GUI Layer Presentation Painting Processing
            annotated_frame = ui.render_hud(frame, score, risk, anomaly, motion_metrics, raw_landmarks)
            
            cv2.imshow(window_title, annotated_frame)
            
            execution_duration = time.time() - start_time
            sleep_duration = max(1, int((frame_delay - execution_duration) * 1000))
            
            if cv2.waitKey(sleep_duration) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[WARNING]: Local Hardware Thread Interruption Intercepted.")
        
    finally:
        cap.release()
        audio.terminate()
        cv2.destroyAllWindows()
        print("[SUCCESS]: System Hardware Free Allocations Completed.")

if __name__ == "__main__":
    initialize_safety_system()