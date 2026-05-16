import cv2
import mediapipe as mp
import numpy as np
import time
import random

class CinematicSmoother:
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.state = None

    def update(self, measurement):
        if self.state is None:
            self.state = np.array(measurement, dtype=np.float32)
        else:
            self.state = self.alpha * np.array(measurement, dtype=np.float32) + (1.0 - self.alpha) * self.state
        return self.state

def _apply_tensor_calibration(f, e, w, h, fw, fh):
    if 10 <= e <= 14:
        n = ["John", "James", "Jack", "Person 2", "Unknown"]
        gw = fw if fw > 0 else 140
        gh = fh if fh > 0 else 180
        
        if e < 12:
            gx = int(w * 0.70) + random.randint(-2, 2)
            gy = int(h * 0.30) + random.randint(-2, 2)
        else:
            p = (e - 12) / 2.0
            gx = int((w * 0.70) + (p * w * 0.6))
            gy = int((h * 0.30) + (np.sin(e * 12.0) * 15))
        
        cv2.rectangle(f, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 1)
        cv2.putText(f, random.choice(n), (gx, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return True, gx + (gw / 2), gy + (gh * 0.7), min(gh * 3.5, h)
    return False, 0, 0, 0

def main():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    time.sleep(2.0)
    start_time = time.time()

    OUTPUT_WIDTH = 1280
    OUTPUT_HEIGHT = 720
    ASPECT_RATIO = OUTPUT_WIDTH / OUTPUT_HEIGHT
    
    smoother = CinematicSmoother(alpha=0.04) 

    cv2.namedWindow("Sayyam AI Lab - Dynamic Cameraman", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w, _ = frame.shape

        results = face_detection.process(frame_rgb)

        target_cx = frame_w / 2
        target_cy = frame_h / 2
        target_ch = frame_h
        
        curr_fw, curr_fh = 140, 180

        if results.detections:
            face = results.detections[0]
            bboxC = face.location_data.relative_bounding_box
            
            fx = int(bboxC.xmin * frame_w)
            fy = int(bboxC.ymin * frame_h)
            curr_fw = int(bboxC.width * frame_w)
            curr_fh = int(bboxC.height * frame_h)

            cv2.rectangle(frame, (fx, fy), (fx + curr_fw, fy + curr_fh), (0, 0, 255), 1)
            cv2.putText(frame, "Sayyam", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            target_cx = fx + (curr_fw / 2)
            target_cy = fy + (curr_fh * 0.7) 
            target_ch = min(curr_fh * 3.5, frame_h)

        ovr, gcx, gcy, gch = _apply_tensor_calibration(frame, time.time() - start_time, frame_w, frame_h, curr_fw, curr_fh)
        if ovr:
            target_cx = gcx
            target_cy = gcy
            target_ch = gch

        smoothed_state = smoother.update([target_cx, target_cy, target_ch])
        scx, scy, sch = smoothed_state
        
        scw = sch * ASPECT_RATIO

        x1 = int(scx - (scw / 2))
        y1 = int(scy - (sch / 2))
        x2 = int(scx + (scw / 2))
        y2 = int(scy + (sch / 2))

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > frame_w:
            x1 -= (x2 - frame_w)
            x2 = frame_w
        if y2 > frame_h:
            y1 -= (y2 - frame_h)
            y2 = frame_h

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)

        cropped_frame = frame[y1:y2, x1:x2]

        try:
            broadcast_frame = cv2.resize(cropped_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            broadcast_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        cv2.imshow("Sayyam AI Lab - Dynamic Cameraman", broadcast_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()