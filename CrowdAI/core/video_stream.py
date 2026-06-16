import cv2
import threading
import time

class VideoStream:
    """
    Thread-isolated frame capture manager capable of streaming from a hardware camera index 
    or an absolute file path path with automatic loop rewinding capabilities.
    """
    def __init__(self, src=0):
        self.src = src
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"[ERROR] Could not initialize video source context target: {src}")
            
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        # Check if compiling from an external file tracking directory path target
        self.is_file = isinstance(src, str)

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            
            # Loop handler: if file hits EOF (End of File), rewind immediately
            if not grabbed and self.is_file:
                # Reset capture position index to frame point zero
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                grabbed, frame = self.stream.read()
                
            if not grabbed:
                self.stop()
                break
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
                
            # Adaptive frame pacing selector: prevents processing the file 
            # faster than the system can display it
            if self.is_file:
                time.sleep(0.025)  # Matches standard ~40 FPS playback ceiling bounds
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()