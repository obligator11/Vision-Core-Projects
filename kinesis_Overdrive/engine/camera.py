"""Threaded camera capture so frame grabbing never blocks the render loop."""

import threading
import cv2


class Camera:
    def __init__(self, index=0, mirror=True):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. "
                "Check that it isn't in use by another application."
            )
        self.mirror = mirror
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._latest_frame = frame

    def read(self):
        """Returns the most recent frame (or None if nothing captured yet)."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()
