import time
import numpy as np
import cv2

from core.pose_tracker import ThreadedPoseTracker
from core.predictor import KinematicPredictor
from core.lag_engine import LagCalculationEngine
from audio.sound_manager import AsynchronousSoundEngine
from ui.lag_visualizer import InterfaceDashboard

def execute_pipeline():
    # Instantiate core sub-systems
    tracker = ThreadedPoseTracker(src=0).start()
    predictor = KinematicPredictor(buffer_size=12, prediction_horizon_sec=0.22)
    engine = LagCalculationEngine(calculation_window=45)
    audio_node = AsynchronousSoundEngine()
    hud_panel = InterfaceDashboard()

    last_sound_time = time.monotonic()
    prev_lag = 0.0

    print("[SYSTEM BOOT] Human Lag Detector online. Press 'q' to safely terminate environment.")

    try:
        while True:
            success, frame = tracker.read_frame()
            if not success or frame is None:
                time.sleep(0.005)
                continue

            # Mirror frames for natural alignment feedback
            frame = cv2.flip(frame, 1)
            
            # Track target feature vectors
            actual_pos = tracker.extract_target_landmark(frame)
            
            predicted_pos = None
            spatial_err = 0.0
            current_lag = 0.0

            if actual_pos is not None:
                # Update predictor models with incoming landmark data coordinates
                smoothed_actual, predicted_pos = predictor.update_and_predict(landmark_id=0, current_pos=actual_pos)
                # Compute spatial-temporal metrics
                spatial_err, current_lag = engine.calculate_metrics(smoothed_actual, predicted_pos)
                
                # Non-blocking adaptive audio feedback management
                now = time.monotonic()
                if now - last_sound_time > 0.6:  # Threshold delay guard
                    if current_lag > 220.0:
                        audio_node.trigger_warning_beep()
                        last_sound_time = now
                    elif current_lag < 80.0 and prev_lag >= 80.0:
                        audio_node.trigger_success_tone()
                        last_sound_time = now
                
                prev_lag = current_lag

            # Update the interface canvas layout
            lag_history = engine.get_lag_history()
            rendered_canvas = hud_panel.draw_hud(
                canvas=frame,
                actual=actual_pos,
                predicted=predicted_pos,
                lag_ms=current_lag,
                distance_error=spatial_err,
                lag_history=lag_history
            )

            # Render frame array sequence to adjustable screen asset
            key_stroke = hud_panel.render_output(rendered_canvas)
            if key_stroke == ord('q'):
                break

    finally:
        print("[SHUTDOWN] Terminating system streams and releasing hardware devices gracefully.")
        tracker.stop()
        hud_panel.close_panels()

if __name__ == "__main__":
    execute_pipeline()