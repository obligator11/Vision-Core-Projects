import cv2
import numpy as np
import mediapipe as mp
import math
import multiprocessing as mp_lib
import time
import random

def worker_process(frame_queue, shared_data):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            h, w, c = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            shared_data[0] = 0.0
            shared_data[100] = 0.0
            
            if results.multi_hand_landmarks:
                for idx, lm in enumerate(results.multi_hand_landmarks):
                    if idx > 1:
                        break
                    
                    offset = idx * 100
                    shared_data[offset] = 1.0
                    
                    cx, cy = lm.landmark[9].x * w, lm.landmark[9].y * h
                    
                    shared_data[offset + 4] = cx
                    shared_data[offset + 5] = cy
                    
                    for i in range(21):
                        shared_data[offset + 6 + i*2] = lm.landmark[i].x * w
                        shared_data[offset + 7 + i*2] = lm.landmark[i].y * h

class ApexEngine:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        
        phi = (1 + math.sqrt(5)) / 2
        self.outer_nodes = np.array([
            [-1,  phi,  0], [ 1,  phi,  0], [-1, -phi,  0], [ 1, -phi,  0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi,  0, -1], [ phi,  0,  1], [-phi,  0, -1], [-phi,  0,  1]
        ]) * 45
        
        self.outer_edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if np.linalg.norm(self.outer_nodes[i] - self.outer_nodes[j]) < 95:
                    self.outer_edges.append((i, j))
                    
        self.inner_nodes = np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ]) * 20
        
        self.inner_edges = [
            (0,2), (0,3), (0,4), (0,5), (1,2), (1,3), 
            (1,4), (1,5), (2,4), (2,5), (3,4), (3,5)
        ]
        
        self.hand_conns = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        self.colors = [
            (0, 255, 255), (255, 0, 255), (255, 215, 0), 
            (50, 255, 50), (0, 100, 255), (255, 255, 255)
        ]
        self.color_idx = {0: 0, 100: 1}
        self.history_x = {0: [], 100: []}
        self.wave_cooldown = {0: 0, 100: 0}

        self.constructs = [
            {"base_x": int(w * 0.2), "base_y": int(h * 0.5), "x": int(w * 0.2), "y": int(h * 0.5), "attached": -1, "color": self.colors[0]},
            {"base_x": int(w * 0.8), "base_y": int(h * 0.5), "x": int(w * 0.8), "y": int(h * 0.5), "attached": -1, "color": self.colors[1]}
        ]
        
        self.shockwave_active = False
        self.shockwave_radius = 0
        self.shockwave_center = (0, 0)
        self.particles = []

    def project(self, t, cx, cy, nodes, speed_mult, expansion=1.0):
        rx, ry, rz = t * 1.2 * speed_mult, t * 1.5 * speed_mult, t * 0.8 * speed_mult
        
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        projected = []
        
        for node in nodes:
            rotated = R @ (node * expansion)
            z = rotated[2] + 400
            f = 400
            if z != 0:
                x = (rotated[0] * f) / z + cx
                y = (rotated[1] * f) / z + cy
            else:
                x, y = cx, cy
            projected.append((int(x), int(y)))
        return projected

    def draw_skeleton(self, frame, offset, shared_data, color):
        pts = []
        for i in range(21):
            px = int(shared_data[offset + 6 + i*2])
            py = int(shared_data[offset + 7 + i*2])
            pts.append((px, py))
            
        for conn in self.hand_conns:
            pt1 = pts[conn[0]]
            pt2 = pts[conn[1]]
            cv2.line(frame, pt1, pt2, color, 3, cv2.LINE_AA)
            
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

    def draw_tethers(self, frame, h_offset, shared_data, cx, cy, color):
        tips = [4, 8, 12, 16, 20]
        for tip in tips:
            px = int(shared_data[h_offset + 6 + tip*2])
            py = int(shared_data[h_offset + 7 + tip*2])
            
            mid_x = (px + cx) // 2 + random.randint(-20, 20)
            mid_y = (py + cy) // 2 + random.randint(-20, 20)
            
            cv2.line(frame, (px, py), (mid_x, mid_y), color, 2, cv2.LINE_AA)
            cv2.line(frame, (mid_x, mid_y), (cx, cy), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)

    def process_logic(self, frame, shared_data):
        t = time.time()
        hand_centers = {}
        
        for h_offset in [0, 100]:
            if self.wave_cooldown[h_offset] > 0:
                self.wave_cooldown[h_offset] -= 1
                
            if shared_data[h_offset] == 1.0:
                hx, hy = shared_data[h_offset + 4], shared_data[h_offset + 5]
                hand_centers[h_offset] = (hx, hy)
                
                skel_color = (255, 255, 255)
                self.draw_skeleton(frame, h_offset, shared_data, skel_color)
                
                self.history_x[h_offset].append(hx)
                if len(self.history_x[h_offset]) > 15:
                    self.history_x[h_offset].pop(0)
                    
                if self.wave_cooldown[h_offset] == 0 and len(self.history_x[h_offset]) == 15:
                    sc = 0
                    ls = 0
                    for i in range(1, 15):
                        dx = self.history_x[h_offset][i] - self.history_x[h_offset][i-1]
                        if abs(dx) > 12:
                            s = 1 if dx > 0 else -1
                            if ls != 0 and s != ls:
                                sc += 1
                            ls = s
                    if sc >= 3:
                        self.color_idx[h_offset] = (self.color_idx[h_offset] + 1) % len(self.colors)
                        self.wave_cooldown[h_offset] = 25
                        for construct in self.constructs:
                            if construct["attached"] == h_offset:
                                construct["color"] = self.colors[self.color_idx[h_offset]]
            else:
                self.history_x[h_offset].clear()

        expansion_factor = 1.0
        
        for construct in self.constructs:
            if construct["attached"] == -1:
                for h_offset, (hx, hy) in hand_centers.items():
                    dist = math.hypot(construct["x"] - hx, construct["y"] - hy)
                    if dist < 120:
                        construct["attached"] = h_offset
                        construct["color"] = self.colors[self.color_idx[h_offset]]
                        break
            else:
                h_offset = construct["attached"]
                if h_offset in hand_centers:
                    construct["x"] = int(hand_centers[h_offset][0])
                    construct["y"] = int(hand_centers[h_offset][1])
                    self.draw_tethers(frame, h_offset, shared_data, construct["x"], construct["y"], construct["color"])
                    
                    self.particles.append({
                        "x": construct["x"] + random.randint(-10, 10),
                        "y": construct["y"] + random.randint(-10, 10),
                        "radius": random.randint(8, 15),
                        "color": construct["color"],
                        "life": 255
                    })
                else:
                    construct["attached"] = -1
                    construct["x"] = construct["base_x"]
                    construct["y"] = construct["base_y"]

        if self.constructs[0]["attached"] != -1 and self.constructs[1]["attached"] != -1:
            if self.constructs[0]["attached"] != self.constructs[1]["attached"]:
                dist = math.hypot(self.constructs[0]["x"] - self.constructs[1]["x"], self.constructs[0]["y"] - self.constructs[1]["y"])
                
                if dist < 350:
                    expansion_factor = 1.0 + ((350 - dist) / 100.0)
                    cv2.line(frame, (self.constructs[0]["x"], self.constructs[0]["y"]), 
                             (self.constructs[1]["x"], self.constructs[1]["y"]), 
                             (255, 255, 255), int(expansion_factor * 2), cv2.LINE_AA)

                if dist < 140 and not self.shockwave_active:
                    self.shockwave_active = True
                    self.shockwave_radius = 10
                    self.shockwave_center = ((self.constructs[0]["x"] + self.constructs[1]["x"]) // 2, 
                                             (self.constructs[0]["y"] + self.constructs[1]["y"]) // 2)
                    
                    for construct in self.constructs:
                        construct["attached"] = -1
                        construct["x"] = construct["base_x"]
                        construct["y"] = construct["base_y"]

        for p in self.particles[:]:
            p["life"] -= 15
            p["radius"] *= 0.9
            if p["life"] <= 0 or p["radius"] < 1:
                self.particles.remove(p)
            else:
                cv2.circle(frame, (int(p["x"]), int(p["y"])), int(p["radius"]), p["color"], -1, cv2.LINE_AA)

        if self.shockwave_active:
            sx = random.randint(-25, 25)
            sy = random.randint(-25, 25)
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            frame = cv2.warpAffine(frame, M, (self.w, self.h))
            
            overlay = frame.copy()
            cv2.circle(overlay, self.shockwave_center, self.shockwave_radius, (255, 255, 255), -1)
            cv2.addWeighted(overlay, max(0, 1.0 - (self.shockwave_radius / 800)), frame, min(1.0, self.shockwave_radius / 800), 0, frame)
            
            cv2.circle(frame, self.shockwave_center, self.shockwave_radius, (255, 0, 255), 25)
            cv2.circle(frame, self.shockwave_center, self.shockwave_radius + 40, (0, 255, 255), 10)
            
            self.shockwave_radius += 85
            if self.shockwave_radius > 1600:
                self.shockwave_active = False

        for construct in self.constructs:
            out_pts = self.project(t, construct["x"], construct["y"], self.outer_nodes, 1.0, expansion_factor)
            in_pts = self.project(t, construct["x"], construct["y"], self.inner_nodes, -2.5, 1.0)
            
            for edge in self.outer_edges:
                cv2.line(frame, out_pts[edge[0]], out_pts[edge[1]], construct["color"], 2, cv2.LINE_AA)
            for pt in out_pts:
                cv2.circle(frame, pt, 4, construct["color"], -1, cv2.LINE_AA)
                
            for edge in self.inner_edges:
                cv2.line(frame, in_pts[edge[0]], in_pts[edge[1]], (255, 255, 255), 1, cv2.LINE_AA)
            for pt in in_pts:
                cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)
                
        return frame

def main():
    frame_queue = mp_lib.Queue(maxsize=1)
    shared_data = mp_lib.Array('d', [0.0] * 200)
    
    worker = mp_lib.Process(target=worker_process, args=(frame_queue, shared_data))
    worker.daemon = True
    worker.start()
    
    cap = cv2.VideoCapture(0)
    ret, initial_frame = cap.read()
    if not ret:
        return
        
    h, w = initial_frame.shape[:2]
    engine = ApexEngine(w, h)
    
    cv2.namedWindow("Project Apex Override", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        if frame_queue.empty():
            frame_queue.put(frame)
            
        dark_bg = np.zeros_like(frame)
        hud_frame = cv2.addWeighted(frame, 0.2, dark_bg, 0.8, 0)
        
        hud_frame = engine.process_logic(hud_frame, shared_data)
            
        cv2.imshow("Project Apex Override", hud_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    frame_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    worker.join()

if __name__ == '__main__':
    try:
        mp_lib.set_start_method('spawn')
    except RuntimeError:
        pass
    main()