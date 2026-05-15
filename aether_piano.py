import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
from collections import deque

class RetroLegendsAudioEngine:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 1024)
        pygame.init()
        pygame.mixer.set_num_channels(32)
        self.sounds = {}
        self._generate_retro_sfx()

    def _generate_retro_sfx(self):
        # 0. Mario Coin
        self.sounds[0] = self._create_sfx([987, 1318], [0.1, 0.4])
        # 1. Sonic Ring (High Chime Sweep)
        self.sounds[1] = self._create_sweep(1500, 2500, 0.2, vol=0.5)
        # 2. Zelda Secret (Fanfare)
        self.sounds[2] = self._create_sfx([784, 830, 880, 932], [0.08]*4)
        # 3. Pac-Man Death (Descending Bloom)
        self.sounds[3] = self._create_sweep(800, 100, 0.5, vol=0.6)
        # 4. Tetris Line Clear (Pulse)
        self.sounds[4] = self._create_sfx([440, 880], [0.05, 0.1])
        # 5. Pokemon Level Up
        self.sounds[5] = self._create_sfx([1046, 1174, 1318, 1397, 1567], [0.05]*5)
        # 6. Megaman Shoot
        self.sounds[6] = self._create_sweep(1000, 200, 0.1, vol=0.4)
        # 7. Final Fantasy Victory (Intro)
        self.sounds[7] = self._create_sfx([1046, 1046, 1046, 1046, 783, 932, 1046], [0.07, 0.07, 0.07, 0.15, 0.15, 0.15, 0.2])

    def _create_sfx(self, freqs, durations, vol=0.5):
        sample_rate = 44100
        full_wave = np.array([], dtype=np.float32)
        for f, d in zip(freqs, durations):
            t = np.linspace(0, d, int(sample_rate * d), False)
            wave = np.sin(f * t * 2 * np.pi)
            attack = np.linspace(0, 1, int(len(t) * 0.1))
            wave[:len(attack)] *= attack
            wave *= np.exp(-3 * t)
            full_wave = np.append(full_wave, wave)
        audio = (full_wave * vol * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

    def _create_sweep(self, f1, f2, duration, vol=0.4):
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        freqs = np.linspace(f1, f2, len(t))
        wave = np.sin(2 * np.pi * freqs * t)
        attack = np.linspace(0, 1, int(len(t) * 0.1))
        wave[:len(attack)] *= attack
        wave *= np.exp(-2 * t)
        audio = (wave * vol * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

    def play_sfx(self, index):
        if index in self.sounds:
            self.sounds[index].play()

class KinematicSensor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
        self.mp_draw = mp.solutions.drawing_utils
        self.history_y = deque(maxlen=5)
        self.last_strike = 0

    def is_strike(self, landmark):
        curr_time = time.time()
        self.history_y.append(landmark.y)
        if len(self.history_y) < 3: return False
        vel_y = self.history_y[-1] - self.history_y[-2]
        if vel_y > 0.006 and (curr_time - self.last_strike > 0.1):
            self.last_strike = curr_time
            return True
        return False

class AetherRetroV16:
    def __init__(self):
        self.audio = RetroLegendsAudioEngine()
        self.sensor = KinematicSensor()
        self.cap = cv2.VideoCapture(0)
        cv2.namedWindow("Aether-OS V16: Retro Legends", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Aether-OS V16: Retro Legends", self._calibrate)
        self.points = []
        self.M, self.invM = None, None
        self.calibrated = False
        self.grid_w, self.grid_h = 500, 1280
        self.num_keys = 8
        self.key_h = self.grid_h // self.num_keys
        self.active_hits = {}
        self.temp_point = None

    def _calibrate(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                if len(self.points) == 4:
                    src = np.float32(self.points)
                    dst = np.float32([[0,0], [self.grid_w,0], [self.grid_w,self.grid_h], [0,self.grid_h]])
                    self.M = cv2.getPerspectiveTransform(src, dst)
                    self.invM = cv2.getPerspectiveTransform(dst, src)
                    self.calibrated = True
        elif event == cv2.EVENT_MOUSEMOVE:
            self.temp_point = (x, y)

    def _render(self, frame):
        hollow = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.uint8)
        curr_time = time.time()
        for i in range(self.num_keys):
            y = i * self.key_h
            is_hit = i in self.active_hits and curr_time - self.active_hits[i] < 0.12
            color = (0, 255, 255) if is_hit else (50, 0, 150)
            cv2.rectangle(hollow, (0, y+2), (self.grid_w, y + self.key_h-2), (15, 15, 15), -1)
            cv2.rectangle(hollow, (0, y+2), (self.grid_w, y + self.key_h-2), color, 6)
            if is_hit:
                sub = hollow[y+2:y+self.key_h-2, :]
                cv2.addWeighted(sub, 0.4, np.full(sub.shape, color, dtype=np.uint8), 0.6, 0, dst=sub)
        warped = cv2.warpPerspective(hollow, self.invM, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 0.7, warped, 0.9, 0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.sensor.hands.process(rgb)
            if not self.calibrated:
                for i, p in enumerate(self.points):
                    cv2.circle(frame, tuple(p), 10, (0, 0, 255), -1)
                    if i > 0: cv2.line(frame, tuple(self.points[i-1]), tuple(self.points[i]), (255, 255, 255), 2)
                if 0 < len(self.points) < 4 and self.temp_point:
                    cv2.line(frame, tuple(self.points[-1]), self.temp_point, (255, 255, 255), 1)
                cv2.putText(frame, "CLICK 4 CORNERS: VERTICAL OBSIDIAN MODE", (50, 80), 1, 2, (0, 255, 0), 2)
            else:
                frame = self._render(frame)
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                tip = hand.landmark[8]
                self.sensor.mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
                if self.calibrated and self.sensor.is_strike(tip):
                    ix, iy = int(tip.x * w), int(tip.y * h)
                    pt = np.array([[[float(ix), float(iy)]]], dtype=np.float32)
                    m_pt = cv2.perspectiveTransform(pt, self.M)[0][0]
                    if 0 <= m_pt[0] < self.grid_w and 0 <= m_pt[1] < self.grid_h:
                        idx = int(m_pt[1] // self.key_h)
                        self.active_hits[idx] = time.time()
                        self.audio.play_sfx(idx)
            cv2.imshow("Aether-OS V16: Retro Legends", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    AetherRetroV16().run()