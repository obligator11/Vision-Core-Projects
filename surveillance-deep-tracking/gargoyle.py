import cv2
import mediapipe as mp
import numpy as np
import multiprocessing
import time
import math
import pygame
from ultralytics import YOLO

def build_alarm():
    # Force stereo mode initialization to be safe
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    duration = 0.2
    sample_rate = 44100
    n_samples = int(round(duration * sample_rate))
    
    # 1D buffer
    buf = np.zeros((n_samples, 1), dtype=np.int16)
    max_sample = 2**(16 - 1) - 1
    
    for s in range(n_samples):
        t = float(s) / sample_rate
        buf[s][0] = int(round(max_sample * math.sin(2 * math.pi * 1000 * t) * math.exp(-3 * t)))
        
    # The Fix: Stack the 1D array into a 2D Stereo Array
    stereo_buf = np.column_stack((buf, buf))
    return pygame.sndarray.make_sound(stereo_buf)

class InferenceWorker(multiprocessing.Process):
    def __init__(self, frame_queue, result_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue

    def run(self):
        yolo_model = YOLO('yolov8n.pt')
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(rgb_frame)
                
                shoulder_coords = None
                if pose_results.pose_landmarks:
                    h, w, _ = frame.shape
                    lm = pose_results.pose_landmarks.landmark[12]
                    shoulder_coords = (int(lm.x * w), int(lm.y * h))

                yolo_results = yolo_model(frame, verbose=False)[0]
                detections = []
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    name = yolo_model.names[cls_id]
                    detections.append({'name': name, 'bbox': (x1, y1, x2, y2), 'conf': conf})

                try:
                    while not self.result_queue.empty():
                        self.result_queue.get_nowait()
                except:
                    pass
                self.result_queue.put({'shoulder': shoulder_coords, 'detections': detections})

class GargoyleEngine:
    def __init__(self):
        self.frame_queue = multiprocessing.Queue(maxsize=2)
        self.result_queue = multiprocessing.Queue(maxsize=2)
        self.worker = InferenceWorker(self.frame_queue, self.result_queue)
        self.worker.daemon = True
        self.worker.start()
        self.alarm_sound = build_alarm()
        self.drone_pos = [0, 0]
        self.target_pos = [0, 0]
        self.mode = "IDLE"
        self.last_alarm_time = 0

    def draw_drone(self, frame, x, y, mode):
        overlay = frame.copy()
        t = time.time()
        
        # Smoother, organic sine-wave hovering
        bounce = int(math.sin(t * 4) * 15)
        cy = y + bounce
        cx = x
        
        # Next-Gen Cybernetic Palette
        if mode == "OVERWATCH":
            color = (0, 0, 255)       # Stark Red
            core_color = (50, 50, 255)
        elif mode == "SCANNING":
            color = (255, 215, 0)     # Golden Yellow
            core_color = (255, 255, 255)
        else:
            color = (255, 255, 0)     # Cyan/Neon Blue for IDLE
            core_color = (255, 255, 200)

        # VFX: Rotating Outer Orbital Rings
        angle = t * 3
        r = 45
        for i in range(3):
            start_angle = angle + i * (2 * math.pi / 3)
            end_angle = start_angle + (math.pi / 3)
            pts = []
            for a in np.linspace(start_angle, end_angle, 10):
                px = int(cx + r * math.cos(a))
                py = int(cy + r * math.sin(a))
                pts.append([px, py])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], False, color, 3)

        # VFX: Counter-Rotating Inner Diamond Core
        inner_r = 20
        inner_angle = -t * 4
        diamond = []
        for i in range(4):
            a = inner_angle + i * (math.pi / 2)
            diamond.append([int(cx + inner_r * math.cos(a)), int(cy + inner_r * math.sin(a))])
        diamond = np.array(diamond, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [diamond], core_color)
        cv2.polylines(overlay, [diamond], True, (255, 255, 255), 1)

        # VFX: The "Eye"
        cv2.circle(overlay, (cx, cy), 8, (255, 255, 255), -1)

        # VFX: Alpha-Blended Plasma Glow
        # We blur the overlay to create a bloom effect, then mathematically merge it
        glow = cv2.GaussianBlur(overlay, (19, 19), 0)
        cv2.addWeighted(glow, 0.6, frame, 0.7, 0, frame)
        
        # VFX: Tactical Targeting Crosshairs
        cv2.line(frame, (cx - 70, cy), (cx - 50, cy), color, 2)
        cv2.line(frame, (cx + 50, cy), (cx + 70, cy), color, 2)
        cv2.line(frame, (cx, cy - 70), (cx, cy - 50), color, 2)
        cv2.line(frame, (cx, cy + 50), (cx, cy + 70), color, 2)

        return cx, cy

    def execute(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Gargoyle Node', cv2.WINDOW_NORMAL)
        
        latest_data = {'shoulder': None, 'detections': []}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            
            if self.frame_queue.empty():
                self.frame_queue.put(frame)
                
            if not self.result_queue.empty():
                latest_data = self.result_queue.get()

            shoulder = latest_data.get('shoulder')
            detections = latest_data.get('detections', [])
            
            self.mode = "ANCHOR"
            if shoulder:
                # Pushed significantly higher (-180) and further out (+140) to clear the face/chin
                self.target_pos = [shoulder[0] + 140, shoulder[1] - 180]
            else:
                h, w, _ = frame.shape
                self.target_pos = [w - 150, 150]

            people = [d for d in detections if d['name'] == 'person']
            objects = [d for d in detections if d['name'] in ['cell phone', 'book']]

            if len(people) > 1:
                people.sort(key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
                intruder = people[1]
                ix = (intruder['bbox'][0] + intruder['bbox'][2]) // 2
                iy = (intruder['bbox'][1] + intruder['bbox'][3]) // 2
                self.target_pos = [ix, iy - 60]
                self.mode = "OVERWATCH"
                
                cv2.rectangle(frame, (intruder['bbox'][0], intruder['bbox'][1]), 
                              (intruder['bbox'][2], intruder['bbox'][3]), (0, 0, 255), 3)
                cv2.line(frame, (ix-20, iy), (ix+20, iy), (0, 0, 255), 2)
                cv2.line(frame, (ix, iy-20), (ix, iy+20), (0, 0, 255), 2)
                
                if time.time() - self.last_alarm_time > 0.3:
                    self.alarm_sound.play()
                    self.last_alarm_time = time.time()
                    
            elif objects:
                obj = objects[0]
                ox = (obj['bbox'][0] + obj['bbox'][2]) // 2
                oy = (obj['bbox'][1] + obj['bbox'][3]) // 2
                self.target_pos = [ox, oy - 100]
                self.mode = "SCANNING"
                
                cv2.rectangle(frame, (obj['bbox'][0], obj['bbox'][1]), 
                              (obj['bbox'][2], obj['bbox'][3]), (255, 215, 0), 2)

            self.drone_pos[0] += (self.target_pos[0] - self.drone_pos[0]) * 0.15
            self.drone_pos[1] += (self.target_pos[1] - self.drone_pos[1]) * 0.15

            dx, dy = self.draw_drone(frame, int(self.drone_pos[0]), int(self.drone_pos[1]), self.mode)

            if self.mode == "SCANNING" and objects:
                obj = objects[0]
                ox = (obj['bbox'][0] + obj['bbox'][2]) // 2
                oy = (obj['bbox'][1] + obj['bbox'][3]) // 2
                overlay = frame.copy()
                cv2.line(overlay, (dx, dy), (ox, oy), (0, 255, 255), 3)
                cv2.circle(overlay, (ox, oy), 15, (0, 255, 255), 1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.imshow('Gargoyle Node', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        self.frame_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    engine = GargoyleEngine()
    engine.execute()