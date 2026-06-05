import cv2
import numpy as np
from ultralytics import YOLO
import config

class VideoProcessor:
    def __init__(self, source=config.CAMERA_INDEX):
        self.source = source
        self.cap = None
        self.model = None

    def initialize_stream(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        return self.cap.isOpened()

    def load_model(self):
        print(f"[INFO] Initializing YOLO26 model layer on device: {config.DEVICE}")
        self.model = YOLO(config.YOLO_MODEL_NAME)
        
        dummy_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        self.model(dummy_frame, device =config.DEVICE, verbose=False)
        print("[INFO] Neural network pipeline fully optimized and warmed up.")

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        return True, frame

    def release_stream(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()




        
        