import cv2
import threading
import time

class ThreadedVideoStream:
    """
    Handles high-performance webcam multi-threaded frame acquisition
    to decouple I/O bottlenecks from processing loops.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise RuntimeError(f"Failed to open video source camera context: {src}")
            
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update_loop, args=(), daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if not grabbed:
                continue
            
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self):
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        self.stream.release()