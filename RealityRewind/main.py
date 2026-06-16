import cv2
import numpy as np
import sys
import time
from core.video_stream import ThreadedVideoStream
from core.motion_tracker import MotionTracker
from core.rewind_engine import RewindEngine
from audio.sound_manager import SoundManager
from ui.rewind_renderer import RewindRenderer

def main():
    print("[ Reality Rewind AI ] initializing spatial tracking matrices...")
    
    try:
        video_source = ThreadedVideoStream(src=0).start()
    except Exception as e:
        print(f"Error accessing hardware capture stream: {e}")
        sys.exit(1)

    tracker = MotionTracker(max_points_to_track=40)
    engine = RewindEngine(fps=30, buffer_seconds=4)
    audio = SoundManager()
    renderer = RewindRenderer()

    window_name = "Reality Rewind AI Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540) 

    current_speed = 1.0
    rewind_toggle_state = False

    while True:
        grabbed, frame = video_source.read()
        if not grabbed or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        
        # Query output image measurements dynamically to handle user-adjusted resizing windows
        window_rect = cv2.getWindowImageRect(window_name)
        win_w, win_h = window_rect[2], window_rect[3]

        key = cv2.waitKey(1) & 0xFF

        # Speed adjustments
        if key == ord('1'): 
            current_speed = 1.0
            if engine.is_rewinding: engine.set_rewind_mode(True, speed=current_speed)
        elif key == ord('2'): 
            current_speed = 2.0
            if engine.is_rewinding: engine.set_rewind_mode(True, speed=current_speed)
        elif key == ord('4'): 
            current_speed = 4.0
            if engine.is_rewinding: engine.set_rewind_mode(True, speed=current_speed)
        elif key == ord('q'):
            break

        # Process space bar key stroke as a single-press toggle loop trigger switch
        if key == ord(' '):
            rewind_toggle_state = not rewind_toggle_state
            if rewind_toggle_state:
                engine.set_rewind_mode(True, speed=current_speed)
                audio.play_rewind_trigger()
                audio.start_ambient_hum()
            else:
                engine.set_rewind_mode(False)
                audio.stop_all()

        if not engine.is_rewinding:
            # When working forward, actively populate history track indexes
            tracked_points = tracker.update(frame)
            engine.record_state(tracked_points)

        # Extract chronological space tracks and kinetics arrays
        tracks, velocities = engine.compute_temporal_tracks()

        # Render paths directly onto current video viewport canvas array frames
        renderer.render(frame, tracks, velocities, engine.is_rewinding, current_speed)

        # Interpolate output canvas parameters if user scales window size parameters
        if win_w > 50 and win_h > 50:
            display_frame = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame

        cv2.imshow(window_name, display_frame)

    print("\nShutting down Reality Rewind pipelines...")
    audio.stop_all()
    video_source.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()