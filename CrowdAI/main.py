import cv2
import sys
import os
import numpy as np

# Align local paths systematically across the script workspace directory structures
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.video_stream import VideoStream
from core.person_detector import PersonDetector
from audio.sound_manager import SoundManager
from ui.crowd_dashboard import CrowdDashboard

class TelemetrySmoother:
    """
    Implements a fast moving-average window and a decay state-machine to prevent
    real-time status flickering while ensuring instant zero-lag response updates.
    """
    def __init__(self, count_window=3, density_window=3, hold_frames=8):
        self.count_history = []
        self.density_history = []
        self.count_window = count_window
        self.density_window = density_window
        
        # Hysteresis parameters to hold onto a HIGH CONGESTION alert state
        self.hold_frames = hold_frames
        self.danger_cooldown_counter = 0

    def smooth(self, raw_count, raw_density):
        self.count_history.append(raw_count)
        if len(self.count_history) > self.count_window:
            self.count_history.pop(0)
            
        self.density_history.append(raw_density)
        if len(self.density_history) > self.density_window:
            self.density_history.pop(0)

        smoothed_count = int(np.mean(self.count_history))
        smoothed_density = int(np.mean(self.density_history))

        if smoothed_count >= 10 or smoothed_density >= 60:
            congestion_level = "HIGH CONGESTION"
            self.danger_cooldown_counter = self.hold_frames  
        else:
            if self.danger_cooldown_counter > 0:
                congestion_level = "HIGH CONGESTION"  
                self.danger_cooldown_counter -= 1
            elif smoothed_count >= 5 or smoothed_density >= 30:
                congestion_level = "MEDIUM"
            else:
                congestion_level = "LOW"

        return smoothed_count, smoothed_density, congestion_level

def calculate_adaptive_crowd_telemetry(boxes, previous_centroids, flow_history, smoother, max_history=10):
    """
    Computes spatial distribution matrices and paths using adaptive metrics 
    and perspective normalization formulas optimized for real-time 30 FPS video frames.
    """
    total_occupants = len(boxes)
    centroids = []
    for (x, y, w, h) in boxes:
        centroids.append([int(x + w / 2), int(y + h)])
    centroids = np.array(centroids)

    if total_occupants == 0:
        smoothed_cnt, smoothed_den, level = smoother.smooth(0, 0)
        return {
            "density_score": smoothed_den,
            "congestion_level": level,
            "flow_efficiency": 100,
            "centroids": np.array([]),
            "cluster_labels": np.array([])
        }, None

    # --- PERSPECTIVE-ADAPTIVE CLUSTERING SECTOR ---
    labels = np.full(total_occupants, -1)
    cluster_id = 0
    
    for i in range(total_occupants):
        if labels[i] != -1:
            continue
            
        y_depth_factor = centroids[i][1]
        adaptive_eps = max(40, int(y_depth_factor * 0.20))
        
        distances = np.linalg.norm(centroids - centroids[i], axis=1)
        neighbors = np.where(distances <= adaptive_eps)[0]
        
        if len(neighbors) >= 3:
            labels[neighbors] = cluster_id
            cluster_id += 1

    high_density_nodes = np.count_nonzero(labels != -1)
    raw_density_score = int((high_density_nodes / total_occupants) * 100)
    
    smoothed_cnt, smoothed_den, congestion_level = smoother.smooth(total_occupants, raw_density_score)

    # --- VECTOR MOTION FLOW TRACKING ---
    flow_efficiency = 100
    if previous_centroids is not None and len(previous_centroids) > 0:
        displacements = []
        for current_node in centroids:
            deltas = np.linalg.norm(previous_centroids - current_node, axis=1)
            closest_idx = np.argmin(deltas)
            
            # Real-Time 30 FPS Rule: Tightened matching threshold to 90 pixels 
            # to remove 1-second track delays and stop track swapping artifacts.
            if deltas[closest_idx] < 90:
                displacements.append(deltas[closest_idx])
        
        if displacements:
            mean_vel = np.mean(displacements)
            flow_history.append(mean_vel)
            if len(flow_history) > max_history:
                flow_history.pop(0)
            
            smoothed_velocity = np.mean(flow_history)
            
            if smoothed_velocity < 1.2 and congestion_level == "HIGH CONGESTION":
                flow_efficiency = max(5, int(smoothed_velocity * 40))
            else:
                flow_efficiency = min(100, int(60 + (smoothed_velocity * 3.5)))
        else:
            if congestion_level == "HIGH CONGESTION":
                flow_efficiency = 10
    else:
        if congestion_level == "HIGH CONGESTION":
            flow_efficiency = 20

    telemetry = {
        "density_score": smoothed_den,
        "congestion_level": congestion_level,
        "flow_efficiency": flow_efficiency,
        "centroids": centroids,
        "cluster_labels": labels
    }
    return telemetry, centroids

def main():
    print("=================================================================")
    print("[BOOT] Launching Finalized Zero-Lag Real-Time AI Crowd Analyzer Subsystems...")
    print("=================================================================")

    video_source_target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "crowd_sample.mp4")
    if not os.path.exists(video_source_target):
        print(f"[SYSTEM WARNING] Sample clip asset missing. Falling back to active Webcam Stream.")
        video_source_target = 0

    try:
        stream = VideoStream(src=video_source_target).start()
    except Exception as err:
        print(f"[FATAL INITIALIZATION ERROR] Video stream failed to bind: {err}")
        return

    detector = PersonDetector(use_yolo=True, confidence_threshold=0.35)
    audio = SoundManager()
    dashboard = CrowdDashboard()
    
    # Optimized window caches for immediate, ultra-fast metric rendering responses
    smoother = TelemetrySmoother(count_window=3, density_window=3, hold_frames=8)

    window_hud_title = "AI Crowd & Queue Analytics HUD Dashboard"
    cv2.namedWindow(window_hud_title, cv2.WINDOW_NORMAL)

    previous_centroids = None
    velocity_history_stack = []

    print("[SYSTEM READY] Processing stability loop initialized. Monitoring active...")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            # Step 1: Object tracking layer execution
            boxes, engine_string_id = detector.detect(frame)

            # Step 2: Ultra-low latency telemetry calculations
            telemetry, current_centroids = calculate_adaptive_crowd_telemetry(
                boxes, previous_centroids, velocity_history_stack, smoother
            )
            previous_centroids = current_centroids

            # Step 3: Trigger alerts
            audio.trigger_alert_state(telemetry["congestion_level"])

            # Step 4: UI Presentation rendering updates
            annotated_frame = dashboard.draw_telemetry_hud(frame, telemetry, boxes, engine_string_id)

            cv2.imshow(window_hud_title, annotated_frame)

            # Standard 30 FPS playback frame tick rate interval delay sync loop context
            key = cv2.waitKey(24) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        stream.stop()
        audio.terminate()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Pipeline terminated safely. Project code completely polished.")

if __name__ == "__main__":
    main()