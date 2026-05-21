import cv2
import numpy as np
import mediapipe as mp
import torch
import multiprocessing as mp_lib

class VisionWorker(mp_lib.Process):
    def __init__(self, frame_q, hand_q):
        super().__init__()
        self.frame_q = frame_q
        self.hand_q = hand_q

    def run(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        while True:
            if not self.frame_q.empty():
                frame = self.frame_q.get()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(img_rgb)
                if res.multi_hand_landmarks:
                    hm = res.multi_hand_landmarks[0]
                    h, w, _ = frame.shape
                    cx, cy = int(hm.landmark[9].x * w), int(hm.landmark[9].y * h)
                    tips = [8, 12, 16, 20]
                    open_fingers = sum(1 for tip in tips if hm.landmark[tip].y < hm.landmark[tip-2].y)
                    is_repel = open_fingers > 3
                    if self.hand_q.full():
                        try:
                            self.hand_q.get_nowait()
                        except:
                            pass
                    self.hand_q.put((cx, cy, is_repel))

class PhysicsWorker(mp_lib.Process):
    def __init__(self, hand_q, render_q, w, h):
        super().__init__()
        self.hand_q = hand_q
        self.render_q = render_q
        self.w = w
        self.h = h

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_particles = 15000
        pos = torch.rand((num_particles, 2), device=device) * torch.tensor([self.w, self.h], device=device)
        vel = torch.zeros((num_particles, 2), device=device)
        G = 8000.0
        hx, hy, is_repel = self.w // 2, self.h // 2, False
        while True:
            if not self.hand_q.empty():
                hx, hy, is_repel = self.hand_q.get()
            target = torch.tensor([hx, hy], device=device, dtype=torch.float32)
            dir_vec = target - pos
            dist_sq = torch.sum(dir_vec**2, dim=1, keepdim=True) + 50.0
            dist = torch.sqrt(dist_sq)
            force_mag = G / dist_sq
            if is_repel:
                force_mag = -force_mag * 8.0
            force = dir_vec / dist * force_mag
            vel += force
            vel *= 0.92
            pos += vel
            pos[:, 0] = torch.clamp(pos[:, 0], 0, self.w - 1)
            pos[:, 1] = torch.clamp(pos[:, 1], 0, self.h - 1)
            pos_np = pos.cpu().numpy().astype(np.int32)
            frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            frame[pos_np[:, 1], pos_np[:, 0]] = [255, 50, 200]
            if self.render_q.full():
                try:
                    self.render_q.get_nowait()
                except:
                    pass
            self.render_q.put(frame)

class VortexEngine:
    def __init__(self):
        self.w, self.h = 1280, 720
        self.frame_q = mp_lib.Queue(maxsize=2)
        self.hand_q = mp_lib.Queue(maxsize=2)
        self.render_q = mp_lib.Queue(maxsize=2)
        self.vision_worker = VisionWorker(self.frame_q, self.hand_q)
        self.physics_worker = PhysicsWorker(self.hand_q, self.render_q, self.w, self.h)
        
    def run(self):
        self.vision_worker.start()
        self.physics_worker.start()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cv2.namedWindow('Vortex', cv2.WINDOW_NORMAL)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.w, self.h))
            if not self.frame_q.full():
                self.frame_q.put(frame.copy())
            if not self.render_q.empty():
                p_frame = self.render_q.get()
                p_frame = cv2.GaussianBlur(p_frame, (5, 5), 0)
                frame = cv2.addWeighted(frame, 0.4, p_frame, 1.5, 0)
            cv2.imshow('Vortex', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        self.vision_worker.terminate()
        self.physics_worker.terminate()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    mp_lib.set_start_method('spawn', force=True)
    engine = VortexEngine()
    engine.run()