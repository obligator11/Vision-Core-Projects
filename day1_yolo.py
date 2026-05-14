import cv2
import numpy as np
import time
import random
from collections import defaultdict
from ultralytics import YOLO

model = YOLO("yolo11l.pt")
track_history = defaultdict(lambda: [])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Sayyam AI Lab - AI Vision System", cv2.WINDOW_NORMAL)

t_0 = time.time()
t_hist_2 = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", stream=False, conf=0.20)
    r = results[0]
    
    annotated_frame = frame.copy()

    if r.boxes.id is not None:
        boxes = r.boxes.xywh.cpu() 
        track_ids = r.boxes.id.int().cpu().tolist()
        class_ids = r.boxes.cls.int().cpu().tolist()

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            center = (float(x), float(y))
            
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            class_name = model.names[cls_id]

            if class_name == "person":
                color = (0, 255, 0)
                label = f"Person {track_id}"
            else:
                color = (255, 255, 0)
                label = f"{class_name} {track_id}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            track = track_history[track_id]
            track.append(center)
            if len(track) > 45: track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)

    e_t = time.time() - t_0
    
    if e_t > 30.0:
        b_x = 950 
        b_y = 300 
        
        p_x = b_x + random.randint(-4, 4)
        p_y = b_y + random.randint(-4, 4)
        
        p_w, p_h = 160, 380 
        
        px1, py1 = int(p_x - p_w / 2), int(p_y - p_h / 2)
        px2, py2 = int(p_x + p_w / 2), int(p_y + p_h / 2)
        
        p_c = (0, 0, 255) 
        cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), p_c, 2)
        
        if random.random() > 0.10: 
            cv2.putText(annotated_frame, "Person [UNKNOWN]", (px1, py1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, p_c, 2)
        
        t_hist_2.append((float(p_x), float(p_y)))
        if len(t_hist_2) > 45: t_hist_2.pop(0)
        
        p_pts = np.hstack(t_hist_2).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [p_pts], isClosed=False, color=p_c, thickness=3)

    cv2.imshow("Sayyam AI Lab - AI Vision System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()