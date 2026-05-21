import cv2
import numpy as np
import torch
import mediapipe as mp
import multiprocessing as mp_lib
import math

class State:
    ORBIT = 0
    RECALL = 1
    STRIKE = 2

def get_gesture(landmarks):
    tip_ids = [8, 12, 16, 20]
    base_ids = [5, 9, 13, 17]
    wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y])
    
    tips = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in tip_ids])
    bases = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in base_ids])
    
    dists = np.linalg.norm(tips - wrist, axis=1)
    base_dists = np.linalg.norm(bases - wrist, axis=1)
    
    extended = dists > base_dists * 1.5
    
    if not any(extended):
        return State.RECALL, wrist, None
    
    if extended[0] and not any(extended[1:]):
        idx_base = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y])
        idx_tip = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y])
        vec = idx_tip - idx_base
        vec = vec / (np.linalg.norm(vec) + 1e-6)
        return State.STRIKE, idx_base, vec
        
    return State.ORBIT, np.mean(bases, axis=0), None

def boids_worker(input_q, output_q, width, height):
    num_boids = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pos = torch.rand((num_boids, 2), device=device, dtype=torch.float32)
    pos[:, 0] *= width
    pos[:, 1] *= height
    vel = (torch.rand((num_boids, 2), device=device, dtype=torch.float32) - 0.5) * 10.0
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    max_speed = 15.0
    orbit_angle = 0.0
    
    while True:
        if not input_q.empty():
            frame_data = input_q.get()
            if frame_data is None:
                break
                
            results = hands.process(frame_data)
            
            target_pos = None
            target_vec = None
            state = State.ORBIT
            
            if results.multi_hand_landmarks:
                state, norm_pos, target_vec = get_gesture(results.multi_hand_landmarks[0])
                target_pos = torch.tensor([norm_pos[0] * width, norm_pos[1] * height], device=device)
                if target_vec is not None:
                    target_vec = torch.tensor([target_vec[0], target_vec[1]], device=device)
            
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            dist = torch.norm(diff, dim=2)
            
            mask_sep = (dist > 0) & (dist < 25.0)
            sep_force = torch.sum(diff * mask_sep.unsqueeze(2), dim=1)
            
            mask_align = (dist > 0) & (dist < 50.0)
            align_count = torch.sum(mask_align, dim=1, keepdim=True).clamp(min=1)
            align_force = (torch.sum(vel.unsqueeze(0).expand(num_boids, -1, -1) * mask_align.unsqueeze(2), dim=1) / align_count) - vel
            
            mask_coh = (dist > 0) & (dist < 70.0)
            coh_count = torch.sum(mask_coh, dim=1, keepdim=True).clamp(min=1)
            coh_center = torch.sum(pos.unsqueeze(0).expand(num_boids, -1, -1) * mask_coh.unsqueeze(2), dim=1) / coh_count
            coh_force = coh_center - pos
            
            cmd_force = torch.zeros_like(vel)
            
            if target_pos is not None:
                if state == State.ORBIT:
                    orbit_angle += 0.1
                    r = 100.0
                    orbit_target = target_pos + torch.tensor([math.cos(orbit_angle)*r, math.sin(orbit_angle)*r], device=device)
                    cmd_force = (orbit_target - pos) * 0.05
                    max_speed = 10.0
                elif state == State.RECALL:
                    cmd_force = (target_pos - pos) * 0.2
                    max_speed = 30.0
                elif state == State.STRIKE:
                    cmd_force = target_vec * 50.0
                    max_speed = 40.0
            else:
                max_speed = 5.0
                
            vel += sep_force * 0.05 + align_force * 0.05 + coh_force * 0.01 + cmd_force
            
            speeds = torch.norm(vel, dim=1, keepdim=True)
            vel = torch.where(speeds > max_speed, vel / speeds * max_speed, vel)
            
            pos += vel
            
            pos[:, 0] = torch.clamp(pos[:, 0], 0, width)
            pos[:, 1] = torch.clamp(pos[:, 1], 0, height)
            
            target_out = target_pos.cpu().numpy() if target_pos is not None else None
            output_q.put((pos.cpu().numpy(), state, target_out))

class PhalanxEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cv2.namedWindow('Phalanx', cv2.WINDOW_NORMAL)
        
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        self.vignette = np.clip(1.5 - r, 0, 1)
        self.vignette = np.dstack([self.vignette]*3).astype(np.float32)
        
        self.input_q = mp_lib.Queue(maxsize=2)
        self.output_q = mp_lib.Queue(maxsize=2)
        
        self.worker = mp_lib.Process(target=boids_worker, args=(self.input_q, self.output_q, self.width, self.height))
        self.worker.daemon = True
        self.worker.start()
        
        self.boids_data = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not self.input_q.full():
                self.input_q.put(frame_rgb)
                
            if not self.output_q.empty():
                self.boids_data = self.output_q.get()
                
            frame = (frame * self.vignette).astype(np.uint8)
            
            if self.boids_data:
                pos_np, state, target_pos = self.boids_data
                
                if state == State.RECALL and target_pos is not None:
                    cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), 40, (255, 255, 255), -1)
                    cv2.circle(frame, (int(target_pos[0]), int(target_pos[1])), 60, (255, 200, 0), 4)
                    
                for p in pos_np:
                    if state == State.STRIKE:
                        color = (0, 0, 255)
                    elif state == State.RECALL:
                        color = (255, 255, 255)
                    else:
                        color = (0, 255, 255)
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, color, -1)
            
            cv2.imshow('Phalanx', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        self.input_q.put(None)
        self.worker.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    mp_lib.freeze_support()
    engine = PhalanxEngine()
    engine.run()