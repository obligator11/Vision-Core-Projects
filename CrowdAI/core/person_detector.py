import cv2
import numpy as np

class PersonDetector:
    """
    Robust feature extraction layer offering native YOLOv8 deep inference tracking 
    with an optimized scale-invariant OpenCV HOG fallback wrapper.
    """
    def __init__(self, use_yolo=True, confidence_threshold=0.4):
        self.use_yolo = use_yolo
        self.confidence_threshold = confidence_threshold
        self.engine_name = "HOG CPU Fallback Engine"
        
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                # Load ultra-lightweight nano model optimized for edge deployments
                self.model = YOLO("yolov8n.pt")
                self.engine_name = "YOLOv8 Edge Inference"
            except ImportError:
                print("[SYSTEM WARNING] 'ultralytics' package missing. Rolling back to native HOG Engine.")
                self.use_yolo = False
                self._bootstrap_hog()
        else:
            self._bootstrap_hog()

    def _bootstrap_hog(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """
        Parses a structural pixel matrix and returns formal bounding boxes: [[x, y, w, h], ...]
        """
        boxes = []
        if self.use_yolo:
            results = self.model(frame, verbose=False)[0]
            for box in results.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Coerce check strictly for COCO standard person taxonomy identification (Class 0)
                if class_id == 0 and confidence >= self.confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
        else:
            # Multi-scale multi-pass aspect window search for human forms
            found, _ = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            for (x, y, w, h) in found:
                boxes.append([int(x), int(y), int(w), int(h)])
                
        return boxes, self.engine_name