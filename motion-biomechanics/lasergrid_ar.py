

import cv2
import mediapipe as mp
import numpy as np
import pygame
import sys
import os
import json
import math
import random

# --- CONFIG ---
HIGH_SCORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscore.json")
CLOSE_CALL_DIST = 55        # px: inside this distance (but not touching) = a "clutch"
CLOSE_CALL_RESET_DIST = 90  # px: must clear this far away before another clutch can trigger
CLUTCH_BASE_POINTS = 25
DIFFICULTY_RAMP_SECONDS = 45.0   # time to roughly double laser speed
NEW_LASER_MILESTONES = [15, 35, 60]  # survival-second thresholds that add a laser


# --- AUDIO SYNTHESIS FALLBACK ---
def generate_sound(freq=440, duration=0.1, wave_type='sine', volume=0.3):
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    if wave_type == 'sine':
        wave = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        wave = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        wave = 2 * (t * freq - np.floor(t * freq + 0.5))
    sound_array = np.zeros((n_samples, 2), dtype=np.int16)
    audio_signal = (wave * volume * 32767).astype(np.int16)
    sound_array[:, 0], sound_array[:, 1] = audio_signal, audio_signal
    return pygame.sndarray.make_sound(sound_array)


def generate_sweep(f_start=400, f_end=1200, duration=0.18, volume=0.35):
    """Rising-pitch chirp used for the 'clutch' near-miss sound."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, False)
    freq_t = np.linspace(f_start, f_end, n_samples)
    phase = 2 * np.pi * np.cumsum(freq_t) / sample_rate
    wave = np.sin(phase)
    envelope = np.linspace(1.0, 0.0, n_samples) ** 0.5
    audio_signal = (wave * envelope * volume * 32767).astype(np.int16)
    sound_array = np.zeros((n_samples, 2), dtype=np.int16)
    sound_array[:, 0], sound_array[:, 1] = audio_signal, audio_signal
    return pygame.sndarray.make_sound(sound_array)


# --- CORE ENGINE CLASSES ---

class PoseEngine:
    def __init__(self, frame_width, frame_height):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.w, self.h = frame_width, frame_height
        self.bones = []

    def process(self, frame_rgb):
        results = self.pose.process(frame_rgb)
        self.bones = []
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        for connection in self.mp_pose.POSE_CONNECTIONS:
            idx1, idx2 = connection
            lm1, lm2 = landmarks[idx1], landmarks[idx2]
            if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                x1, y1 = int(lm1.x * self.w), int(lm1.y * self.h)
                x2, y2 = int(lm2.x * self.w), int(lm2.y * self.h)
                self.bones.append(((x1, y1), (x2, y2)))
        return results

    def get_bones_array(self):
        if not self.bones:
            return np.empty((0, 2, 2))
        return np.array(self.bones, dtype=np.float32)


class LaserManager:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.lasers = []
        self.reset()

    def reset(self):
        # HUMAN MODE: no diagonals, gentle base speeds -- ramps up via update_difficulty().
        self.lasers = [
            {"type": "horizontal", "y": self.h * 0.1, "speed": 1.0, "dir": 1, "hue": 0},
            {"type": "vertical", "x": self.w * 0.1, "speed": 1.5, "dir": 1, "hue": 120},
        ]
        self.next_milestone_idx = 0

    def add_laser(self):
        """Called at score milestones to escalate difficulty."""
        if random.random() < 0.5:
            y = random.choice([self.h * 0.3, self.h * 0.5, self.h * 0.7, self.h * 0.9])
            self.lasers.append({"type": "horizontal", "y": y, "speed": 1.2, "dir": random.choice([-1, 1]), "hue": 200})
        else:
            x = random.choice([self.w * 0.3, self.w * 0.5, self.w * 0.7, self.w * 0.9])
            self.lasers.append({"type": "vertical", "x": x, "speed": 1.6, "dir": random.choice([-1, 1]), "hue": 300})

    def update(self, speed_multiplier=1.0):
        for l in self.lasers:
            step = l["speed"] * speed_multiplier * l["dir"]
            if l["type"] == "horizontal":
                l["y"] += step
                if l["y"] > self.h - 50 or l["y"] < 50:
                    l["dir"] *= -1
            elif l["type"] == "vertical":
                l["x"] += step
                if l["x"] > self.w - 50 or l["x"] < 50:
                    l["dir"] *= -1

    def get_laser_segments(self):
        segments = []
        for l in self.lasers:
            if l["type"] == "horizontal":
                segments.append(((0, l["y"]), (self.w, l["y"])))
            elif l["type"] == "vertical":
                segments.append(((l["x"], 0), (l["x"], self.h)))
        return np.array(segments, dtype=np.float32)

    def draw(self, surface, t):
        for l in self.lasers:
            pulse = (math.sin(t * 6 + l["hue"]) + 1) / 2  # 0..1 shimmer
            core = (255, int(60 + 120 * pulse), int(60 + 120 * pulse))
            glow = (255, 130, 130)
            if l["type"] == "horizontal":
                p1, p2 = (0, int(l["y"])), (self.w, int(l["y"]))
            else:
                p1, p2 = (int(l["x"]), 0), (int(l["x"]), self.h)
            pygame.draw.line(surface, glow, p1, p2, 10)
            pygame.draw.line(surface, core, p1, p2, 4)


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color")

    def __init__(self, x, y):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(80, 320)
        self.x, self.y = x, y
        self.vx, self.vy = math.cos(angle) * speed, math.sin(angle) * speed
        self.max_life = random.uniform(0.4, 0.9)
        self.life = self.max_life
        self.color = random.choice([(255, 80, 80), (255, 160, 60), (255, 220, 120)])

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 250 * dt  # gravity
        self.life -= dt

    def draw(self, surface):
        if self.life <= 0:
            return
        alpha = max(0, min(255, int(255 * (self.life / self.max_life))))
        r = max(1, int(6 * (self.life / self.max_life)))
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (r, r), r)
        surface.blit(s, (int(self.x - r), int(self.y - r)))


class MainGame:
    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()

        try:
            self.sfx_hit = pygame.mixer.Sound("laser_break.wav")
            self.sfx_start = pygame.mixer.Sound("checkpoint.wav")
        except (pygame.error, FileNotFoundError):
            self.sfx_hit = generate_sound(freq=150, duration=0.6, wave_type='square', volume=0.7)
            self.sfx_start = generate_sound(freq=800, duration=0.3, wave_type='sine', volume=0.5)
        self.sfx_clutch = generate_sweep()
        self.sfx_milestone = generate_sound(freq=600, duration=0.25, wave_type='sine', volume=0.4)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("ERROR: Could not open webcam (index 0). Check that a camera is connected "
                  "and not in use by another application.")
            pygame.quit()
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, test_frame = self.cap.read()
        if not ok or test_frame is None:
            print("ERROR: Webcam opened but returned no frame. Try a different camera index "
                  "or check camera permissions.")
            self.cap.release()
            pygame.quit()
            sys.exit(1)
        self.frame_h, self.frame_w = test_frame.shape[:2]

        self.screen = pygame.display.set_mode((self.frame_w, self.frame_h), pygame.RESIZABLE)
        pygame.display.set_caption("⚡ LaserGrid AR: Rogue Agent Heist")
        self.clock = pygame.time.Clock()

        self.pose_engine = PoseEngine(self.frame_w, self.frame_h)
        self.laser_manager = LaserManager(self.frame_w, self.frame_h)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

        self.state = "START_SCREEN"
        self.score = 0.0
        self.bonus_points = 0
        self.streak = 0
        self.level = 1
        self.countdown_timer = 0
        self.paused = False

        self.font = pygame.font.SysFont("Impact", 48)
        self.large_font = pygame.font.SysFont("Impact", 120)
        self.small_font = pygame.font.SysFont("Impact", 28)

        self.high_score = self.load_high_score()
        self.is_new_high_score = False

        self.particles = []
        self.shake_timer = 0.0
        self.shake_mag = 0
        self.flash_timer = 0.0
        self.flash_color = (255, 255, 255)

        self.close_call_active = False
        self.time_elapsed = 0.0

    # --- persistence ---
    def load_high_score(self):
        try:
            with open(HIGH_SCORE_PATH, "r") as f:
                data = json.load(f)
                return int(data.get("high_score", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return 0

    def save_high_score(self):
        try:
            with open(HIGH_SCORE_PATH, "w") as f:
                json.dump({"high_score": self.high_score}, f)
        except OSError:
            pass  # non-fatal: just don't persist this run

    # --- gesture check ---
    def check_palm_open(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        tips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        open_fingers = 0
        for tip_idx, mcp_idx in zip(tips, mcps):
            tip, mcp = hand_landmarks.landmark[tip_idx], hand_landmarks.landmark[mcp_idx]
            dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            dist_mcp = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
            if dist_tip > dist_mcp * 1.3:
                open_fingers += 1
        return open_fingers >= 3

    # --- collision + near-miss ---
    def check_intersections_vectorized(self, bones, lasers):
        if bones.shape[0] == 0 or lasers.shape[0] == 0:
            return False, None
        A = bones[:, 0, :][:, np.newaxis, :]
        B = bones[:, 1, :][:, np.newaxis, :]
        C = lasers[:, 0, :][np.newaxis, :, :]
        D = lasers[:, 1, :][np.newaxis, :, :]

        den = (A[..., 0] - B[..., 0]) * (C[..., 1] - D[..., 1]) - (A[..., 1] - B[..., 1]) * (C[..., 0] - D[..., 0])
        den = np.where(den == 0, 1e-10, den)

        t = ((A[..., 0] - C[..., 0]) * (C[..., 1] - D[..., 1]) - (A[..., 1] - C[..., 1]) * (C[..., 0] - D[..., 0])) / den
        u = ((A[..., 0] - C[..., 0]) * (A[..., 1] - B[..., 1]) - (A[..., 1] - C[..., 1]) * (A[..., 0] - B[..., 0])) / den

        intersects = (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        if np.any(intersects):
            idx = np.argwhere(intersects)[0]
            bone_i = idx[0]
            hit_x = A[bone_i, 0, 0] + t[bone_i, idx[1]] * (B[bone_i, 0, 0] - A[bone_i, 0, 0])
            hit_y = A[bone_i, 0, 1] + t[bone_i, idx[1]] * (B[bone_i, 0, 1] - A[bone_i, 0, 1])
            return True, (float(hit_x), float(hit_y))
        return False, None

    def min_distance_to_lasers(self, bones):
        """Closest distance from any joint/bone endpoint to any laser line -- used for the near-miss 'clutch' mechanic."""
        if bones.shape[0] == 0 or not self.laser_manager.lasers:
            return 9999.0
        pts = np.vstack([bones[:, 0, :], bones[:, 1, :]])
        min_dist = 9999.0
        for l in self.laser_manager.lasers:
            if l["type"] == "horizontal":
                d = np.min(np.abs(pts[:, 1] - l["y"]))
            else:
                d = np.min(np.abs(pts[:, 0] - l["x"]))
            min_dist = min(min_dist, float(d))
        return min_dist

    # --- fx helpers ---
    def trigger_hit_fx(self, pos):
        if pos is not None:
            for _ in range(35):
                self.particles.append(Particle(*pos))
        self.shake_timer = 0.35
        self.shake_mag = 14
        self.flash_timer = 0.15
        self.flash_color = (255, 60, 60)

    def trigger_clutch_fx(self):
        self.sfx_clutch.play()
        self.flash_timer = 0.12
        self.flash_color = (80, 255, 255)

    def draw_centered_text(self, surface, text, color, y_offset=0, use_large=False, outline=True):
        f = self.large_font if use_large else self.font
        if outline:
            outline_surf = f.render(text, True, (0, 0, 0))
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                r = outline_surf.get_rect(center=(self.frame_w // 2 + dx, self.frame_h // 2 + y_offset + dy))
                surface.blit(outline_surf, r)
        txt_surf = f.render(text, True, color)
        rect = txt_surf.get_rect(center=(self.frame_w // 2, self.frame_h // 2 + y_offset))
        surface.blit(txt_surf, rect)

    def reset_run(self):
        self.laser_manager.reset()
        self.score = 0.0
        self.bonus_points = 0
        self.streak = 0
        self.level = 1
        self.particles = []
        self.close_call_active = False
        self.is_new_high_score = False

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            dt = min(dt, 0.05)  # clamp so a lag spike can't teleport a laser through the player
            self.time_elapsed += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p and self.state == "PLAYING":
                    self.paused = not self.paused

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- MENU / GESTURE STATES ---
            if self.state in ["START_SCREEN", "GAME_OVER"]:
                hand_results = self.hands.process(frame_rgb)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        if self.check_palm_open(hand_landmarks):
                            self.sfx_start.play()
                            self.state = "COUNTDOWN"
                            self.countdown_timer = 3.0
                            self.reset_run()

            # --- COUNTDOWN STATE ---
            elif self.state == "COUNTDOWN":
                self.pose_engine.process(frame_rgb)
                self.countdown_timer -= dt
                if self.countdown_timer <= 0:
                    self.state = "PLAYING"

            # --- GAMEPLAY STATE ---
            elif self.state == "PLAYING" and not self.paused:
                self.pose_engine.process(frame_rgb)

                speed_multiplier = 1.0 + (self.score / DIFFICULTY_RAMP_SECONDS)
                self.laser_manager.update(speed_multiplier)
                self.score += dt

                # difficulty milestones
                idx = self.laser_manager.next_milestone_idx
                if idx < len(NEW_LASER_MILESTONES) and self.score >= NEW_LASER_MILESTONES[idx]:
                    self.laser_manager.add_laser()
                    self.laser_manager.next_milestone_idx += 1
                    self.level += 1
                    self.sfx_milestone.play()
                    self.flash_timer = 0.1
                    self.flash_color = (255, 220, 90)

                bones_mat = self.pose_engine.get_bones_array()
                lasers_mat = self.laser_manager.get_laser_segments()

                hit, hit_pos = self.check_intersections_vectorized(bones_mat, lasers_mat)
                if hit:
                    self.state = "GAME_OVER"
                    self.sfx_hit.play()
                    self.trigger_hit_fx(hit_pos)
                    total_score = int(self.score) + self.bonus_points
                    if total_score > self.high_score:
                        self.high_score = total_score
                        self.is_new_high_score = True
                        self.save_high_score()
                else:
                    dist = self.min_distance_to_lasers(bones_mat)
                    if dist < CLOSE_CALL_DIST and not self.close_call_active:
                        self.close_call_active = True
                        self.streak += 1
                        self.bonus_points += CLUTCH_BASE_POINTS * self.streak
                        self.trigger_clutch_fx()
                    elif dist > CLOSE_CALL_RESET_DIST:
                        self.close_call_active = False

            # update fx timers regardless of pause so menus feel responsive
            if self.shake_timer > 0:
                self.shake_timer -= dt
            if self.flash_timer > 0:
                self.flash_timer -= dt
            for p in self.particles:
                p.update(dt)
            self.particles = [p for p in self.particles if p.life > 0]

            # --- RENDERING ---
            frame_surface = pygame.surfarray.make_surface(np.swapaxes(frame_rgb, 0, 1))
            win_w, win_h = self.screen.get_size()
            win_w, win_h = max(win_w, 1), max(win_h, 1)  # guard against a minimized/zero-size window
            scaled_bg = pygame.transform.scale(frame_surface, (win_w, win_h))

            offset_x, offset_y = 0, 0
            if self.shake_timer > 0:
                offset_x = random.randint(-self.shake_mag, self.shake_mag)
                offset_y = random.randint(-self.shake_mag, self.shake_mag)

            self.screen.fill((0, 0, 0))
            self.screen.blit(scaled_bg, (offset_x, offset_y))

            ar_layer = pygame.Surface((self.frame_w, self.frame_h), pygame.SRCALPHA)

            if self.state == "START_SCREEN":
                ar_layer.fill((0, 0, 0, 150))
                self.draw_centered_text(ar_layer, "LASERGRID AR", (0, 255, 255), -70)
                self.draw_centered_text(ar_layer, "SHOW OPEN PALM TO START", (255, 255, 255), 30)
                hs_txt = self.small_font.render(f"HIGH SCORE: {self.high_score}", True, (255, 220, 90))
                ar_layer.blit(hs_txt, hs_txt.get_rect(center=(self.frame_w // 2, self.frame_h // 2 + 90)))

            elif self.state == "COUNTDOWN":
                for bone in self.pose_engine.bones:
                    pygame.draw.line(ar_layer, (0, 255, 255), bone[0], bone[1], 4)
                ar_layer.fill((0, 0, 0, 100))
                self.draw_centered_text(ar_layer, "STEP BACK!", (255, 255, 0), -80)
                self.draw_centered_text(ar_layer, str(math.ceil(self.countdown_timer)), (255, 255, 255), 40, use_large=True)

            elif self.state == "PLAYING":
                for bone in self.pose_engine.bones:
                    pygame.draw.line(ar_layer, (0, 255, 255), bone[0], bone[1], 4)
                self.laser_manager.draw(ar_layer, self.time_elapsed)
                for p in self.particles:
                    p.draw(ar_layer)
                if self.paused:
                    ar_layer.fill((0, 0, 0, 160))
                    self.draw_centered_text(ar_layer, "PAUSED", (255, 255, 255), 0, use_large=True)
                    self.draw_centered_text(ar_layer, "PRESS P TO RESUME", (200, 200, 200), 80)

            elif self.state == "GAME_OVER":
                ar_layer.fill((60, 0, 0, 140))
                for p in self.particles:
                    p.draw(ar_layer)
                self.draw_centered_text(ar_layer, "AGENT CAUGHT", (255, 50, 50), -110)
                total_score = int(self.score) + self.bonus_points
                self.draw_centered_text(ar_layer, f"SCORE: {total_score}", (255, 255, 255), -30)
                if self.is_new_high_score:
                    self.draw_centered_text(ar_layer, "NEW HIGH SCORE!", (255, 220, 90), 30)
                else:
                    self.draw_centered_text(ar_layer, f"BEST: {self.high_score}", (200, 200, 200), 30)
                self.draw_centered_text(ar_layer, "SHOW OPEN PALM TO RESTART", (255, 255, 255), 100)

            if self.flash_timer > 0:
                alpha = int(180 * (self.flash_timer / 0.15))
                flash_surf = pygame.Surface((self.frame_w, self.frame_h), pygame.SRCALPHA)
                flash_surf.fill((*self.flash_color, min(alpha, 180)))
                ar_layer.blit(flash_surf, (0, 0))

            scaled_ar = pygame.transform.scale(ar_layer, (win_w, win_h))
            self.screen.blit(scaled_ar, (offset_x, offset_y))

            if self.state == "PLAYING":
                total_score = int(self.score) + self.bonus_points
                hud_lines = [
                    (f"SURVIVAL: {int(self.score)}s", (255, 255, 255)),
                    (f"BONUS: {self.bonus_points}", (255, 220, 90)),
                    (f"SCORE: {total_score}", (0, 255, 200)),
                    (f"LEVEL {self.level}", (0, 200, 255)),
                ]
                for i, (txt, color) in enumerate(hud_lines):
                    surf = self.small_font.render(txt, True, color)
                    self.screen.blit(surf, (20, 20 + i * 32))
                if self.streak > 1:
                    streak_surf = self.font.render(f"{self.streak}x CLUTCH!", True, (80, 255, 255))
                    self.screen.blit(streak_surf, (win_w - streak_surf.get_width() - 20, 20))

            pygame.display.flip()

        self.cap.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = MainGame()
    game.run()