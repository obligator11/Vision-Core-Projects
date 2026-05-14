import cv2
import numpy as np
import threading
import queue
from ultralytics import YOLO
import easyocr

class OverwatchEngine:
    def __init__(self, video_source, yolo_model_path="yolo11n.pt"):
        print("[SYS] Initializing Sayyam AI Lab: Project 'Overwatch'...")
        
        # 1. The Core Sensor (YOLO on CUDA)
        print(f"[SYS] Loading Vision Engine: {yolo_model_path}")
        self.model = YOLO(yolo_model_path)
        
        # Dynamically grab the AI's known classes (Fixes the Car detection bug)
        self.ai_classes = self.model.names 
        
        # 2. The Secondary Sensor (Classical Computer Vision for Plates)
        # Using a built-in OpenCV Haar Cascade to find plates since base YOLO cannot.
        print("[SYS] Initializing Classical Haar Cascade Plate Hunter...")
        self.plate_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        
        # 3. Ingestion Layer
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # 4. Omni-Engine: Asynchronous Pipeline
        self.ocr_queue = queue.Queue(maxsize=20)
        self.plate_cache = {}  # Stores track_id -> extracted_text
        self.reader = easyocr.Reader(['en'], gpu=True) 
        
        # Start OCR Worker Thread
        self.ocr_thread = threading.Thread(target=self._async_ocr_worker, daemon=True)
        self.ocr_thread.start()
        print("[SYS] Asynchronous OCR Engine Engaged.")

    def _async_ocr_worker(self):
        """Background thread to process expensive OCR tasks without dropping FPS."""
        while True:
            try:
                plate_img, track_id = self.ocr_queue.get()
                
                # Image Clarification (OpenCV Blueprints)
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Execute OCR
                result = self.reader.readtext(thresh, detail=0)
                if result:
                    # Clean up text and store in cache
                    text = "".join(e for e in result[0].upper() if e.isalnum())
                    if len(text) > 3: # Ignore random short garbage reads
                        self.plate_cache[track_id] = text
                    
                self.ocr_queue.task_done()
            except Exception:
                pass 

    def _check_containment(self, inner_box, outer_box):
        """Math utility to check if a violation (no-helmet) is inside a vehicle box."""
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box
        return (ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2)

    def execute(self):
        cv2.namedWindow("Sayyam AI Lab: Overwatch Engine", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sayyam AI Lab: Overwatch Engine", 1280, 720)
        print("[SYS] Overwatch Live. Press 'q' to terminate.")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # High-Speed Inference (Tracker enabled to assign persistent IDs)
            results = self.model.track(frame, persist=True, verbose=False)
            
            attributes = [] # Holds helmets/seatbelts
            
            # Safe check to ensure we actually detected something with an ID
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                # First pass: Separate vehicles from attributes (like helmets)
                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    cls_name = self.ai_classes[cls_id]
                    
                    if cls_name in ['Helmet', 'No-Helmet', 'Seatbelt', 'No-Seatbelt']:
                        attributes.append((box, cls_name))
                        continue
                        
                    # Target Vehicles (Cars, Trucks, Buses, Motorcycles)
                    if cls_name in ['car', 'truck', 'bus', 'motorcycle']:
                        x1, y1, x2, y2 = map(int, box)
                        is_compliant = True
                        color = (0, 255, 0) # Green (Compliant / Normal)
                        
                        # --- 1. Judgment UI & Logic ---
                        for a_box, a_name in attributes:
                            if self._check_containment(a_box, box):
                                if (cls_name == 'motorcycle' and a_name == 'No-Helmet') or \
                                   (cls_name in ['car', 'truck'] and a_name == 'No-Seatbelt'):
                                    is_compliant = False
                                    break 
                        
                        if not is_compliant:
                            color = (0, 0, 255) # Red (Violation!)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        label = f"ID:{track_id} {cls_name.upper()}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # --- 2. Identity Extraction (Plate Hunting) ---
                        # Crop the vehicle and scan ONLY the vehicle for a license plate
                        veh_crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        
                        if veh_crop.size > 0:
                            gray_veh = cv2.cvtColor(veh_crop, cv2.COLOR_BGR2GRAY)
                            # Haar Cascade scanning for plates
                            plates = self.plate_detector.detectMultiScale(gray_veh, scaleFactor=1.1, minNeighbors=4)
                            
                            for (px, py, pw, ph) in plates:
                                # Convert relative crop coordinates back to absolute frame coordinates
                                abs_px1, abs_py1 = x1 + px, y1 + py
                                abs_px2, abs_py2 = abs_px1 + pw, abs_py1 + ph
                                
                                # Draw Gold box around the plate
                                cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 215, 255), 2)
                                
                                # Push to Async OCR Queue if not cached
                                if track_id not in self.plate_cache and not self.ocr_queue.full():
                                    plate_img_crop = frame[abs_py1:abs_py2, abs_px1:abs_px2]
                                    self.ocr_queue.put((plate_img_crop, track_id))
                                
                                # Render the Extracted Text
                                plate_text = self.plate_cache.get(track_id, "SCANNING...")
                                cv2.putText(frame, plate_text, (abs_px1, abs_py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

            cv2.imshow("Sayyam AI Lab: Overwatch Engine", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        print("[SYS] Overwatch Engine safely shut down.")

if __name__ == "__main__":
    VIDEO_URL = "test_traffic.mp4" # Ensure your video file name matches here!
    
    engine = OverwatchEngine(video_source=VIDEO_URL, yolo_model_path="yolo11n.pt")
    engine.execute()