import cv2
import numpy as np
from collections import defaultdict
import time
from ultralytics import YOLO
import mediapipe as mp

class FrictionlessRetailEngine:
    def __init__(self, source, model_path='yolo11n.pt'):
        self.cap = cv2.VideoCapture(source)
        self.model = YOLO(model_path, task='segment')
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=10,
            min_detection_confidence=0.5
        )
        self.carts = defaultdict(set)
        self.interaction_start = {}
        self.running = True

    def process_frame(self, frame):
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False, 
            device='0'
        )
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.mp_hands.process(rgb_frame)

        hand_coords = []
        if hand_results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in hand_results.multi_hand_landmarks:
                cx = int(hand_lms.landmark[9].x * w)
                cy = int(hand_lms.landmark[9].y * h)
                hand_coords.append((cx, cy))

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            persons = []
            items = []

            for box, obj_id, cls_id in zip(boxes, ids, classes):
                label = self.model.names[cls_id]
                if label == 'person':
                    persons.append((obj_id, box))
                else:
                    items.append((obj_id, box, label))

            for item_id, item_box, item_label in items:
                ix1, iy1, ix2, iy2 = item_box
                touched = False
                touching_person_id = None

                for px, py in hand_coords:
                    if ix1 <= px <= ix2 and iy1 <= py <= iy2:
                        for person_id, person_box in persons:
                            px1, py1, px2, py2 = person_box
                            if px1 <= px <= px2 and py1 <= py <= py2:
                                touched = True
                                touching_person_id = person_id
                                break
                    if touched:
                        break

                pair_key = (touching_person_id, item_id)
                
                if touched and touching_person_id is not None:
                    if pair_key not in self.interaction_start:
                        self.interaction_start[pair_key] = time.time()
                    elif time.time() - self.interaction_start[pair_key] > 0.5:
                        self.carts[touching_person_id].add(item_label)
                else:
                    keys_to_remove = [k for k in self.interaction_start if k[1] == item_id]
                    for k in keys_to_remove:
                        del self.interaction_start[k]

            for obj_id, box, cls_id in zip(ids, boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                display_text = f"ID:{obj_id} {label}"
                if label == 'person' and obj_id in self.carts:
                    display_text += f" Cart:{list(self.carts[obj_id])}"
                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow("Frictionless Retail Engine", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = FrictionlessRetailEngine(source="cctv_feed.mp4", model_path="yolo11n.pt")
    engine.run()