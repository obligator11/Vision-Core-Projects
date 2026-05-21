import cv2
import mediapipe as mp
import numpy as np
import torch
import multiprocessing
import pyaudio
import threading

class SpatialAudioEngine:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.sample_rate = 44100
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=2, rate=self.sample_rate, output=True)
        self.left_freq = 0.0
        self.right_freq = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._play_loop, daemon=True)
        self.thread.start()

    def set_frequencies(self, left_freq, right_freq):
        self.left_freq = left_freq
        self.right_freq = right_freq

    def _play_loop(self):
        chunk_size = 1024
        t = 0
        while self.running:
            if self.left_freq < 10 and self.right_freq < 10:
                samples = np.zeros((chunk_size, 2), dtype=np.float32)
                t = 0
            else:
                time_arr = np.arange(t, t + chunk_size) / self.sample_rate
                
                lf = self.left_freq
                if lf > 1500:
                    lf = lf + 300 * np.sin(2 * np.pi * 15 * time_arr)
                    
                rf = self.right_freq
                if rf > 1500:
                    rf = rf + 300 * np.sin(2 * np.pi * 15 * time_arr)

                left_wave = np.sin(2 * np.pi * lf * time_arr)
                right_wave = np.sin(2 * np.pi * rf * time_arr)
                samples = np.column_stack((left_wave, right_wave)).astype(np.float32)
                t += chunk_size
            self.stream.write(samples.tobytes())

    def close(self):
        self.running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def sensor_worker(frame_queue, data_queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            input_batch = transform(rgb).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_map = prediction.cpu().numpy()

            left_wrist = None
            right_wrist = None
            depth_l = 0
            depth_r = 0

            if results.pose_landmarks:
                h, w = frame.shape[:2]
                l_node = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                r_node = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                lx, ly = int(l_node.x * w), int(l_node.y * h)
                rx, ry = int(r_node.x * w), int(r_node.y * h)

                lx = max(0, min(w - 1, lx))
                ly = max(0, min(h - 1, ly))
                rx = max(0, min(w - 1, rx))
                ry = max(0, min(h - 1, ry))

                left_wrist = (lx, ly)
                right_wrist = (rx, ry)
                depth_l = depth_map[ly, lx]
                depth_r = depth_map[ry, rx]

            edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
            neon_edges = np.zeros_like(frame)
            neon_edges[edges > 0] = [255, 100, 0]

            while not data_queue.empty():
                try:
                    data_queue.get_nowait()
                except:
                    pass
            data_queue.put({
                "left_wrist": left_wrist,
                "right_wrist": right_wrist,
                "depth_l": depth_l,
                "depth_r": depth_r,
                "neon_edges": neon_edges
            })

class EchoLocationEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_queue = multiprocessing.Queue(maxsize=2)
        self.data_queue = multiprocessing.Queue(maxsize=2)
        self.worker = multiprocessing.Process(target=sensor_worker, args=(self.frame_queue, self.data_queue))
        self.worker.daemon = True
        self.worker.start()
        self.audio = SpatialAudioEngine()
        self.fade_frames = 0
        self.max_fade = 30
        cv2.namedWindow("Sayyam AI Lab: Echo-Location HUD", cv2.WINDOW_NORMAL)

    def run(self):
        latest_data = None
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            if self.frame_queue.empty():
                try:
                    self.frame_queue.put_nowait(frame)
                except:
                    pass

            if not self.data_queue.empty():
                try:
                    latest_data = self.data_queue.get_nowait()
                except:
                    pass

            canvas = np.zeros_like(frame)

            if latest_data:
                lw = latest_data["left_wrist"]
                rw = latest_data["right_wrist"]
                dl = latest_data["depth_l"]
                dr = latest_data["depth_r"]
                edges = latest_data["neon_edges"]

                freq_l = np.clip(dl * 3.0, 150, 2500) if lw else 0
                freq_r = np.clip(dr * 3.0, 150, 2500) if rw else 0
                self.audio.set_frequencies(freq_l, freq_r)

                if lw and rw:
                    dist = np.linalg.norm(np.array(lw) - np.array(rw))
                    if dist < 40 and self.fade_frames == 0:
                        self.fade_frames = self.max_fade

                if self.fade_frames > 0:
                    alpha = self.fade_frames / self.max_fade
                    canvas = cv2.addWeighted(canvas, 1.0, edges, alpha, 0)
                    self.fade_frames -= 1

            cv2.imshow("Sayyam AI Lab: Echo-Location HUD", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.audio.close()
        self.frame_queue.put(None)
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    engine = EchoLocationEngine()
    engine.run()