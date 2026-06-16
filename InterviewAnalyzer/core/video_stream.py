import cv2
import threading
import time

class VideoStream:
    """Decouples frame acquisition from main loop processing. Natively monitors 
    and handles file FPS metadata to prevent fast-forwarding."""
    def __init__(self, source=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(source)
        self.is_file = isinstance(source, str)
        
        if not self.is_file:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.frame_delay = 0.005  # Minimal throttle for live camera hardware
        else:
            # Query source asset properties to track exact native playback pacing
            fps = self.stream.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120: 
                fps = 30.0  # Safe robust baseline fallback
            self.frame_delay = 1.0 / fps

        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            start_time = time.time()
            grabbed, frame = self.stream.read()
            
            if not grabbed:
                if self.is_file:
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.01)
                    continue
                    
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                
            # Strict execution pace throttle logic matching original recording speed
            elapsed = time.time() - start_time
            sleep_time = max(0.001, self.frame_delay - elapsed)
            time.sleep(sleep_time)

    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()