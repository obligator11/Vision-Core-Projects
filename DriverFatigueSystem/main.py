import cv2
import os
import sys
import numpy as np
from config import SystemConfig
from core.video_stream import VideoStream
from core.face_detector import FaceDetector
from core.fatigue_analyzer import FatigueAnalyzer
from audio.sound_manager import SoundManager
from ui.overlay import UIOverlay

class MasterDriverSystemOrchestrator:
    """Application wrapper handling cross-module integrations and system lifecycles execution runs."""
    
    def __init__(self) -> None:
        print("[INFO] Initializing system subcomponents pipelines...")
        
        # Verify and assemble directory structures prerequisites configs bounds paths
        if not os.path.exists(SystemConfig.AUDIO_DIRECTORY):
            os.makedirs(SystemConfig.AUDIO_DIRECTORY)
            print(f"[INFO] Constructed placeholder components path route maps structures setup: {SystemConfig.AUDIO_DIRECTORY}")
            
        # Instantiate architectural abstractions entities blocks layers configuration maps
        self.video_stream = VideoStream(
            src=SystemConfig.CAMERA_INDEX,
            width=SystemConfig.FRAME_WIDTH,
            height=SystemConfig.FRAME_HEIGHT
        )
        self.face_detector = FaceDetector()
        self.fatigue_analyzer = FatigueAnalyzer()
        self.sound_manager = SoundManager()
        self.ui_overlay = UIOverlay()
        
        self.is_running: bool = False

    def run(self) -> None:
        """Starts processing operations loops, tracking execution frameworks frames constraints."""
        self.video_stream.start()
        self.is_running = True
        
        # Define window title reference
        window_title = "Driver Fatigue & Attention Monitoring System"
        
        # --- THE SCREEN FIX ---
        # Initialize a resizable display context before entering the video frame loop
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        
        print("[INFO] Processing stream frames execution. Press 'Q' to terminate execution loops context.")

        while self.is_running:
            grabbed, frame = self.video_stream.read()
            if not grabbed or frame is None:
                continue

            h, w, _ = frame.shape
            landmarks_contract = self.face_detector.process_frame(frame)
            
            fatigue_score, attention_score, risk_status = 0.0, 100.0, "SAFE"
            
            if landmarks_contract is not None:
                fatigue_score, attention_score, risk_status = self.fatigue_analyzer.analyze(
                    landmarks_contract=landmarks_contract,
                    img_w=w,
                    img_h=h
                )
                
            self.sound_manager.update_status(risk_status)
            
            processed_frame = self.ui_overlay.render_hud(
                frame=frame,
                fatigue_score=fatigue_score,
                attention_score=attention_score,
                risk_status=risk_status,
                landmarks_contract=landmarks_contract
            )
            
            # Show on our adjustable canvas window
            cv2.imshow(window_title, processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                self.is_running = False

        self._shutdown()

    def _shutdown(self) -> None:
        """Safely stops operations pipelines and frees open system resources pools drivers contexts."""
        print("[INFO] Processing systematic closure routines structures contexts setups protocols...")
        self.video_stream.stop()
        self.face_detector.close()
        self.sound_manager.close()
        cv2.destroyAllWindows()
        print("[INFO] Structural resource instances terminated successfully. System safe shutdown achieved.")

if __name__ == "__main__":
    orchestrator = MasterDriverSystemOrchestrator()
    orchestrator.run()