import cv2
import time
from threading import Thread, Lock
from config import SystemConfig

class VideoStream:
    """Manages an asynchronous background thread loop to capture frames from a webcam or video file."""
    
    def __init__(self, src: int | str = 0, width: int = 640, height: int = 480) -> None:
        self.src = src
        self.stream = cv2.VideoCapture(src)
        
        # Only apply resolution configurations to physical cameras, not static files
        if isinstance(self.src, int):
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.grabbed, self.frame = self.stream.read()
        self.running: bool = False
        self.thread: Thread | None = None
        self.lock: Lock = Lock()

    def start(self) -> 'VideoStream':
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self._update_loop, name="BackgroundVideoStreamThread", args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def _update_loop(self) -> None:
        """Asynchronously queries the source stream; resets index paths to loop if reading a file."""
        while self.running:
            grabbed, frame = self.stream.read()
            
            if not grabbed:
                # If it's a file path string, auto-rewind the video to loop it continuously
                if isinstance(self.src, str):
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.002)
                    continue
                    
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
                
            # If reading a file, add a tiny sleep constraint to simulate natural playback speed 
            if isinstance(self.src, str):
                time.sleep(1 / SystemConfig.FPS_TARGET)

    def read(self) -> tuple[bool, cv2.Mat | None]:
        with self.lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.stream.isOpened():
            self.stream.release()