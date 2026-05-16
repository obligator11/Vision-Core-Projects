import cv2
import mediapipe as mp
import numpy as np
import multiprocessing as mp_lib
import time
import random
import math
import pygame

def vision_worker(frame_q, result_q):
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    prev_gray = None
    prev_pts = None
    prev_wrists = None
    while True:
        if not frame_q.empty():
            frame = frame_q.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            shift = 0.0
            kinetic = 0.0
            chest = None
            if res.pose_landmarks:
                h, w = frame.shape[:2]
                lms = res.pose_landmarks.landmark
                chest_x = int((lms[11].x + lms[12].x) * w / 2)
                chest_y = int((lms[11].y + lms[12].y) * h / 2)
                chest = (chest_x, chest_y)
                wrists = [(lms[15].x * w, lms[15].y * h), (lms[16].x * w, lms[16].y * h)]
                if prev_wrists:
                    kinetic = math.dist(wrists[0], prev_wrists[0]) + math.dist(wrists[1], prev_wrists[1])
                prev_wrists = wrists
                xs = [int(lm.x * w) for lm in lms]
                ys = [int(lm.y * h) for lm in lms]
                x_min, x_max = max(0, min(xs)), min(w, max(xs))
                y_min, y_max = max(0, min(ys)), min(h, max(ys))
                mask = np.zeros_like(gray)
                cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
                if prev_gray is None or prev_pts is None or len(prev_pts) < 10:
                    prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                    prev_gray = gray.copy()
                else:
                    pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    if pts is not None and st is not None:
                        good_new = pts[st == 1]
                        good_old = prev_pts[st == 1]
                        if len(good_new) > 0:
                            shifts = np.linalg.norm(good_new - good_old, axis=1)
                            shift = np.max(shifts) if len(shifts) > 0 else 0.0
                        prev_pts = good_new.reshape(-1, 1, 2)
                    prev_gray = gray.copy()
            while not result_q.empty():
                try:
                    result_q.get_nowait()
                except:
                    pass
            result_q.put({'shift': shift, 'kinetic': kinetic, 'chest': chest})

class SurvivalEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        t = np.arange(44100)
        
        arr_red = np.sign(np.sin(2 * np.pi * t * 980 / 44100)).astype(np.float32)
        arr_red = (arr_red * 32767).astype(np.int16)
        stereo_red = np.column_stack((arr_red, arr_red))
        self.alarm = pygame.sndarray.make_sound(stereo_red)
        
        arr_warn = np.sign(np.sin(2 * np.pi * t * 1300 / 44100)) * np.square(np.sin(2 * np.pi * t * 6 / 44100))
        arr_warn = (arr_warn * 26000).astype(np.int16)
        stereo_warn = np.column_stack((arr_warn, arr_warn))
        self.warn_sound = pygame.sndarray.make_sound(stereo_warn)
        
        self.frame_q = mp_lib.Queue(maxsize=1)
        self.res_q = mp_lib.Queue(maxsize=1)
        self.worker = mp_lib.Process(target=vision_worker, args=(self.frame_q, self.res_q), daemon=True)
        self.worker.start()
        self.state = "GREEN"
        self.state_time = time.time()
        self.duration = random.uniform(2.0, 3.5)
        self.kinetic_energy = 0.0

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Red-Light: Micro-Motion Survival', cv2.WINDOW_NORMAL)
        last_res = {'shift': 0, 'kinetic': 0, 'chest': None}
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            if self.frame_q.empty():
                self.frame_q.put(frame.copy())
            if not self.res_q.empty():
                last_res = self.res_q.get()
            h, w = frame.shape[:2]
            grime = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
            frame = cv2.addWeighted(frame, 0.4, grime, 0.8, -40)
            now = time.time()
            elapsed = now - self.state_time
            if self.state == "GREEN":
                eye_color = (0, 255, 0)
                self.kinetic_energy = min(300, self.kinetic_energy + last_res.get('kinetic', 0) * 0.25)
                cv2.ellipse(frame, (w // 2, 90), (90, 15), 0, 0, 360, eye_color, 4)
                if elapsed > self.duration:
                    self.state = "WARN"
                    self.state_time = now
                    self.duration = 1.2
                    self.warn_sound.play(-1)
            elif self.state == "WARN":
                eye_color = (0, 255, 255)
                pulse = int(abs(math.sin(now * 15)) * 15)
                cv2.ellipse(frame, (w // 2, 90), (90, 20 + pulse), 0, 0, 360, eye_color, 6)
                cv2.circle(frame, (w // 2, 90), 12, (0, 255, 255), -1)
                if elapsed > self.duration:
                    self.state = "RED"
                    self.state_time = now
                    self.duration = random.uniform(1.5, 3.0)
                    self.warn_sound.stop()
                    self.alarm.play(-1)
            elif self.state == "RED":
                eye_color = (0, 0, 255)
                cv2.ellipse(frame, (w // 2, 90), (90, 55), 0, 0, 360, eye_color, -1)
                cv2.circle(frame, (w // 2, 90), 26, (0, 0, 0), -1)
                if last_res.get('shift', 0) > 3.0:
                    self.state = "EXECUTION"
                    self.state_time = now
                    self.alarm.stop()
                elif elapsed > self.duration:
                    self.state = "GREEN"
                    self.state_time = now
                    self.duration = random.uniform(2.0, 3.5)
                    self.alarm.stop()
            elif self.state == "EXECUTION":
                dx, dy = random.randint(-45, 45), random.randint(-45, 45)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(frame, M, (w, h))
                flash = np.full((h, w, 3), (0, 0, 255), dtype=np.uint8)
                frame = cv2.addWeighted(frame, 0.2, flash, 0.8, 0)
                cv2.ellipse(frame, (w // 2, 90), (90, 55), 0, 0, 360, (0, 0, 255), -1)
                if last_res.get('chest'):
                    cv2.line(frame, (w // 2, 90), last_res['chest'], (0, 0, 255), 20)
                if elapsed > 1.5:
                    self.state = "GREEN"
                    self.state_time = now
                    self.kinetic_energy = 0
            cv2.rectangle(frame, (40, h - 70), (40 + int(self.kinetic_energy), h - 35), (0, 255, 255), -1)
            cv2.rectangle(frame, (40, h - 70), (340, h - 35), (255, 255, 255), 3)
            cv2.imshow('Red-Light: Micro-Motion Survival', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mp_lib.freeze_support()
    engine = SurvivalEngine()
    engine.run()