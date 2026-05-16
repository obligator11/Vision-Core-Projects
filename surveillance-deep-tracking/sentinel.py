import cv2
import time
import numpy as np
import mediapipe as mp
import multiprocessing as mp_lib
from ultralytics import YOLO
from scipy.spatial import distance as dist
import winsound

def yolo_inference_worker(input_q, output_q):
    model = YOLO('yolo11n.pt') 
    while True:
        if not input_q.empty():
            frame = input_q.get()
            # High confidence to keep books from triggering phone alerts
            results = model(frame, device='0', conf=0.68, verbose=False)[0]
            phone_detected = any(model.names[int(box.cls)] == 'cell phone' for box in results.boxes)
            if output_q.empty():
                output_q.put({'phone': phone_detected})

class AegisSentinelV4:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6)
        
        self.frame_queue = mp_lib.Queue(maxsize=1)
        self.result_queue = mp_lib.Queue(maxsize=1)
        self.yolo_proc = mp_lib.Process(target=yolo_inference_worker, 
                                       args=(self.frame_queue, self.result_queue),
                                       daemon=True)
        self.yolo_proc.start()

        self.eye_closed_start = None
        self.blink_limit = 0.38 

    def get_gaze_score(self, pts):
        """Precise Iris-to-Corner ratio."""
        # Right Eye: 33 (Outer), 133 (Inner)
        d_outer = dist.euclidean(pts[33], pts[468])
        d_inner = dist.euclidean(pts[133], pts[468])
        h_ratio = d_outer / (d_outer + d_inner) if (d_outer + d_inner) != 0 else 0.5
        
        # Vertical: 159 (Top), 145 (Bottom)
        d_top = dist.euclidean(pts[159], pts[468])
        d_bot = dist.euclidean(pts[145], pts[468])
        v_ratio = d_top / (d_top + d_bot) if (d_top + d_bot) != 0 else 0.5
        
        return h_ratio, v_ratio

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Aegis Focus Sentinel', cv2.WINDOW_NORMAL)

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            if self.frame_queue.empty(): self.frame_queue.put(frame)
            yolo_data = self.result_queue.get() if not self.result_queue.empty() else {'phone': False}
            
            res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            violations = []

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                pts = [(int(l.x * w), int(l.y * h)) for l in lms]

                # EAR for Blink Guard
                ear = (dist.euclidean(pts[159], pts[145]) + dist.euclidean(pts[158], pts[153])) / (2.0 * dist.euclidean(pts[33], pts[133]))

                if ear < 0.15:
                    if self.eye_closed_start is None: self.eye_closed_start = time.time()
                    if time.time() - self.eye_closed_start > self.blink_limit:
                        violations.append("EYES CLOSED")
                else:
                    self.eye_closed_start = None
                    h_ratio, v_ratio = self.get_gaze_score(pts)

                    # REFINED LEFT/RIGHT: 0.38 - 0.62 is the "Center Zone"
                    if h_ratio < 0.35 or h_ratio > 0.65:
                        violations.append("LOOKING AWAY (L/R)")
                    
                    # UP: Alarm if iris hits top lid (v_ratio < 0.3)
                    # DOWN: (Book) v_ratio > 0.6 is totally safe
                    if v_ratio < 0.30:
                        violations.append("LOOKING AWAY (UP)")

                # Head Orientation
                if abs(lms[1].x - lms[10].x) > 0.16:
                    violations.append("HEAD TURNED")

            if yolo_data['phone']:
                violations.append("PHONE DETECTED")

            # GLASS HUD LOGIC
            overlay = frame.copy()
            hud_color = (0, 255, 0) if not violations else (0, 0, 255)
            
            # Semi-transparent top bar
            cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Text Info
            status = "FOCUS RETAINED" if not violations else f"ALARM: {', '.join(violations)}"
            cv2.putText(frame, f"SCORE: {100 if not violations else 0}%", (20, 40), 2, 0.8, hud_color, 2)
            cv2.putText(frame, status, (w // 2 - 120, 40), 2, 0.6, hud_color, 2)

            if violations:
                winsound.Beep(2500, 80)
                # Red Alert Vignette
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 180), 15)
                cv2.putText(frame, "RETURN TO WORK", (w // 2 - 100, h - 50), 2, 1, (0, 0, 255), 3)

            # Iris Crosshairs
            if res.multi_face_landmarks:
                for eye_p in [pts[468], pts[473]]:
                    cv2.drawMarker(frame, eye_p, (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

            cv2.imshow('Aegis Focus Sentinel', frame)
            if cv2.waitKey(1) & 0xFF == 27: break

        self.yolo_proc.terminate()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mp_lib.freeze_support()
    AegisSentinelV4().run()