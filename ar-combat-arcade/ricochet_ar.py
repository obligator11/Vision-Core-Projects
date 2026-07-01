import pygame
import cv2
import numpy as np
import math
import random
import sys
import mediapipe as mp

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
FPS = 60

STATE_WAITING = "WAITING"
STATE_PLAYING = "PLAYING"
STATE_GAMEOVER = "GAMEOVER"

HEAD_HIT_PADDING = 1.25
TRAIL_LENGTH = 10
MAX_LIVES = 3
INVULN_DURATION = 1.2       # seconds of safety after taking a hit
CATCH_BONUS_INTERVAL = 5    # frames trapped between bonus "extra collision" points
MAX_HELD_FRAMES = 90        # force-eject a ball that's been trapped too long

NEON_CYAN = (0, 255, 255)
NEON_MAGENTA = (255, 0, 200)
NEON_GREEN = (80, 255, 130)
NEON_RED = (255, 50, 90)
NEON_YELLOW = (255, 230, 60)
NEON_PURPLE = (190, 100, 255)


# ==========================================
# VISUAL FX HELPERS
# ==========================================
class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(2, 9)
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([math.cos(angle), math.sin(angle)]) * speed
        self.life = random.uniform(0.25, 0.55)
        self.max_life = self.life
        self.color = color
        self.radius = random.uniform(2, 5)

    def update(self, dt):
        self.pos += self.vel
        self.vel *= 0.92
        self.life -= dt
        return self.life > 0

    def draw(self, surface):
        if self.life <= 0:
            return
        alpha = max(0, min(255, int(255 * (self.life / self.max_life))))
        s = pygame.Surface((self.radius * 4, self.radius * 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (int(self.radius * 2), int(self.radius * 2)), int(self.radius))
        surface.blit(s, (self.pos[0] - self.radius * 2, self.pos[1] - self.radius * 2))


class FloatingText:
    def __init__(self, x, y, text, color, size=30):
        self.pos = np.array([x, y], dtype=np.float64)
        self.text = text
        self.color = color
        self.life = 0.8
        self.max_life = self.life
        self.font = pygame.font.SysFont("arial", size, bold=True)

    def update(self, dt):
        self.pos[1] -= 55 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surface):
        alpha = max(0, min(255, int(255 * (self.life / self.max_life))))
        surf = self.font.render(self.text, True, self.color)
        surf.set_alpha(alpha)
        rect = surf.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
        surface.blit(surf, rect)


class Laser:
    def __init__(self, x, y, vx, vy, speed=12):
        self.pos = np.array([x, y], dtype=np.float64)
        v = np.array([vx, vy], dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm == 0:
            norm = 1
        self.vel = (v / norm) * speed
        self.radius = 7
        self.color = NEON_CYAN
        self.active = True
        self.trail = []

        # --- catch-mechanic state ---
        self.was_inside = False   # was this ball inside the hand hull last frame?
        self.held_frames = 0      # how many consecutive frames it's been trapped

    def update(self, dt):
        self.trail.append(tuple(self.pos))
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

        self.pos += self.vel
        margin = 100
        if (self.pos[0] < -margin or self.pos[0] > WINDOW_WIDTH + margin or
                self.pos[1] < -margin or self.pos[1] > WINDOW_HEIGHT + margin):
            self.active = False

    def draw(self, surface):
        n = len(self.trail)
        glow_color = NEON_PURPLE if self.held_frames > 0 else self.color
        for i, p in enumerate(self.trail):
            alpha = int(180 * (i / max(1, n)))
            r = max(1, int(self.radius * 0.6 * (i / max(1, n))))
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*glow_color, alpha), (r, r), r)
            surface.blit(s, (p[0] - r, p[1] - r))

        ring_color = NEON_PURPLE if self.held_frames > 0 else (0, 150, 150)
        pygame.draw.circle(surface, ring_color, (int(self.pos[0]), int(self.pos[1])), self.radius + 5, 2)
        pygame.draw.circle(surface, glow_color, (int(self.pos[0]), int(self.pos[1])), self.radius)
        pygame.draw.circle(surface, (255, 255, 255), (int(self.pos[0]), int(self.pos[1])), int(self.radius * 0.45))


class RicochetEngine:
    def __init__(self):
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Ricochet AR: Physics Engine")
        self.clock = pygame.time.Clock()
        self.font_huge = pygame.font.SysFont("arial", 72, bold=True)
        self.font_big = pygame.font.SysFont("arial", 48, bold=True)
        self.font_small = pygame.font.SysFont("arial", 26, bold=True)
        self.font_hearts = pygame.font.SysFont("arial", 30, bold=True)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(min_detection_confidence=0.6)

        self.sound_fire = self._generate_synth_wave(frequency=300, duration=0.1, wave_type='sawtooth')
        self.sound_deflect = self._generate_synth_wave(frequency=880, duration=0.08, wave_type='sine')
        self.sound_catch = self._generate_synth_wave(frequency=1300, duration=0.05, wave_type='sine')
        self.sound_hit = self._generate_synth_wave(frequency=140, duration=0.35, wave_type='sawtooth')
        self.sound_start = self._generate_synth_wave(frequency=660, duration=0.12, wave_type='sine')
        self.sound_combo = self._generate_synth_wave(frequency=1100, duration=0.07, wave_type='sine')

        self.lasers = []
        self.particles = []
        self.floating_texts = []
        self.spawn_timer = 0.0
        self.score = 0
        self.combo = 0
        self.best_combo = 0

        self.lives = MAX_LIVES
        self.invuln_timer = 0.0

        self.head_center = None
        self.head_radius = 0
        self.head_bbox = None

        self.current_w, self.current_h = WINDOW_WIDTH, WINDOW_HEIGHT
        self.state = STATE_WAITING
        self.gesture_hold_timer = 0.0

        self.shake_timer = 0.0
        self.shake_mag = 0.0
        self.pulse_t = 0.0

        self.scanline_surface = None
        self.vignette_surface = None
        self._build_static_overlays(self.current_w, self.current_h)

    # ------------------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------------------
    def _generate_synth_wave(self, frequency, duration, wave_type='sine'):
        sample_rate = 44100
        n_samples = int(round(duration * sample_rate))
        t = np.linspace(0, duration, n_samples, False)

        if wave_type == 'sine':
            wave = np.sin(frequency * t * 2 * np.pi)
        else:
            wave = 2 * (t * frequency - np.floor(t * frequency + 0.5))

        audio = np.zeros((n_samples, 2), dtype=np.int16)
        audio[:, 0] = wave * 16000
        audio[:, 1] = wave * 16000
        return pygame.sndarray.make_sound(audio)

    # ------------------------------------------------------------------
    # STATIC NEON OVERLAYS
    # ------------------------------------------------------------------
    def _build_static_overlays(self, w, h):
        scan = pygame.Surface((w, h), pygame.SRCALPHA)
        for y in range(0, h, 3):
            pygame.draw.line(scan, (0, 255, 255, 14), (0, y), (w, y))
        self.scanline_surface = scan

        vig = pygame.Surface((w, h), pygame.SRCALPHA)
        max_dist = math.hypot(w / 2, h / 2)
        steps = 24
        for i in range(steps):
            t = i / steps
            radius = max_dist * (1 - t)
            alpha = int(110 * t)
            pygame.draw.circle(vig, (0, 0, 0, alpha), (w // 2, h // 2), int(radius), width=int(max_dist / steps) + 2)
        self.vignette_surface = vig

    # ------------------------------------------------------------------
    # GESTURE: open palm
    # ------------------------------------------------------------------
    def is_open_palm(self, landmarks):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        extended = 0
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                extended += 1

        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        if abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x):
            extended += 1

        return extended >= 4

    # ------------------------------------------------------------------
    # HEAD TARGET
    # ------------------------------------------------------------------
    def update_head_info(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        if not results.detections:
            return

        h, w = frame.shape[:2]
        det = results.detections[0]
        rb = det.location_data.relative_bounding_box

        x = int(rb.xmin * w)
        y = int(rb.ymin * h)
        bw = int(rb.width * w)
        bh = int(rb.height * h)

        self.head_bbox = (x, y, bw, bh)
        self.head_center = (x + bw // 2, y + bh // 2)
        self.head_radius = int((max(bw, bh) / 2) * HEAD_HIT_PADDING)

    # ------------------------------------------------------------------
    # HAND SHIELD — the hand itself is the collider
    # ------------------------------------------------------------------
    def get_hand_hull(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark],
            dtype=np.int32
        )
        return cv2.convexHull(pts)

    def compute_surface_normal(self, contour, impact_point):
        pts = contour.reshape(-1, 2)
        dist_sq = np.sum((pts - impact_point) ** 2, axis=1)
        idx = np.argmin(dist_sq)

        N = len(pts)
        spread = max(1, min(5, N // 3))
        p_prev = pts[(idx - spread) % N]
        p_next = pts[(idx + spread) % N]

        D = p_next - p_prev
        normal = np.array([-D[1], D[0]], dtype=np.float64)

        norm_mag = np.linalg.norm(normal)
        if norm_mag == 0:
            return np.array([0, 1], dtype=np.float64)
        return normal / norm_mag

    # ------------------------------------------------------------------
    # SPAWNING — balls launch toward the player's head
    # ------------------------------------------------------------------
    def spawn_projectile(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])

        if edge == 'top':
            x, y = random.randint(0, self.current_w), 0
        elif edge == 'bottom':
            x, y = random.randint(0, self.current_w), self.current_h
        elif edge == 'left':
            x, y = 0, random.randint(0, self.current_h)
        else:
            x, y = self.current_w, 0

        if self.head_center is not None:
            tx, ty = self.head_center
        else:
            tx, ty = self.current_w // 2, self.current_h // 2

        vx = tx - x + random.randint(-30, 30)
        vy = ty - y + random.randint(-30, 30)

        speed = 15 + min(8, self.score * 0.15)
        self.lasers.append(Laser(x, y, vx, vy, speed=speed))
        self.sound_fire.play()

    # ------------------------------------------------------------------
    # FX SPAWNERS
    # ------------------------------------------------------------------
    def spawn_burst(self, x, y, color, count=18):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def trigger_shake(self, magnitude, duration):
        self.shake_mag = magnitude
        self.shake_timer = duration

    # ------------------------------------------------------------------
    # UI OVERLAYS
    # ------------------------------------------------------------------
    def draw_centered_text(self, text, font, color, y_offset=0, glow=False):
        if glow:
            glow_surf = font.render(text, True, color)
            glow_surf.set_alpha(70)
            for ox, oy in [(-4, 0), (4, 0), (0, -4), (0, 4)]:
                r = glow_surf.get_rect(center=(self.current_w // 2 + ox, self.current_h // 2 + y_offset + oy))
                self.screen.blit(glow_surf, r)

        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(self.current_w // 2, self.current_h // 2 + y_offset))
        shadow = font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, (rect.x + 3, rect.y + 3))
        self.screen.blit(surf, rect)

    def draw_hud(self):
        score_surf = self.font_small.render(f"BLOCKS  {self.score}", True, NEON_CYAN)
        self.screen.blit(score_surf, (24, 22))

        combo_color = NEON_YELLOW if self.combo >= 3 else (255, 255, 255)
        combo_surf = self.font_small.render(f"COMBO x{self.combo}", True, combo_color)
        self.screen.blit(combo_surf, (24, 56))

        # Lives as pulsing hearts, top-right
        heart_color = NEON_RED
        if self.invuln_timer > 0 and int(self.pulse_t * 10) % 2 == 0:
            heart_color = (255, 255, 255)
        hearts_str = "\u2665 " * self.lives + "\u2661 " * (MAX_LIVES - self.lives)
        hearts_surf = self.font_hearts.render(hearts_str.strip(), True, heart_color)
        self.screen.blit(hearts_surf, (self.current_w - hearts_surf.get_width() - 24, 22))

    def draw_target_reticle(self):
        if self.head_center is None:
            return
        cx, cy = self.head_center
        r = self.head_radius
        pulse = 4 * math.sin(self.pulse_t * 6)
        color = (255, 255, 255) if self.invuln_timer > 0 else NEON_RED
        pygame.draw.circle(self.screen, color, (cx, cy), int(r + pulse), 2)
        for ang in (0, 90, 180, 270):
            rad = math.radians(ang + self.pulse_t * 40)
            x1 = cx + math.cos(rad) * (r + 14)
            y1 = cy + math.sin(rad) * (r + 14)
            x2 = cx + math.cos(rad) * (r + 22)
            y2 = cy + math.sin(rad) * (r + 22)
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 3)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            self.pulse_t += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.current_w, self.current_h = event.w, event.h
                    self.screen = pygame.display.set_mode((self.current_w, self.current_h), pygame.RESIZABLE)
                    self._build_static_overlays(self.current_w, self.current_h)

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.current_w, self.current_h))

            self.update_head_info(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb)
            hand_landmarks = None
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            shield_contour = None
            if self.state == STATE_PLAYING and hand_landmarks is not None:
                shield_contour = self.get_hand_hull(frame, hand_landmarks)
                if shield_contour is not None:
                    cv2.drawContours(frame, [shield_contour], -1, (0, 255, 0), 4)

            if self.head_bbox is not None:
                x, y, bw, bh = self.head_bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 200), 2)

            # --- Palm gesture handling (start / restart) ---
            if self.state in (STATE_WAITING, STATE_GAMEOVER):
                if hand_landmarks is not None and self.is_open_palm(hand_landmarks.landmark):
                    self.gesture_hold_timer += dt
                else:
                    self.gesture_hold_timer = 0.0

                if self.gesture_hold_timer > 0.6:
                    self.state = STATE_PLAYING
                    self.lasers = []
                    self.particles = []
                    self.floating_texts = []
                    self.score = 0
                    self.combo = 0
                    self.lives = MAX_LIVES
                    self.invuln_timer = 0.0
                    self.spawn_timer = 0.0
                    self.gesture_hold_timer = 0.0
                    self.sound_start.play()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.swapaxes(frame_rgb, 0, 1)
            frame_surface = pygame.surfarray.make_surface(frame_rgb)

            shake_x = shake_y = 0
            if self.shake_timer > 0:
                shake_x = random.randint(-int(self.shake_mag), int(self.shake_mag))
                shake_y = random.randint(-int(self.shake_mag), int(self.shake_mag))
                self.shake_timer -= dt

            self.screen.fill((0, 0, 0))
            self.screen.blit(frame_surface, (shake_x, shake_y))

            # --- PLAYING state: spawn, move, collide ---
            if self.state == STATE_PLAYING:
                self.draw_target_reticle()

                if self.invuln_timer > 0:
                    self.invuln_timer -= dt

                self.spawn_timer += dt
                spawn_interval = max(0.35, 0.8 - self.score * 0.02)
                if self.spawn_timer > spawn_interval:
                    self.spawn_projectile()
                    self.spawn_timer = 0

                for laser in self.lasers:
                    laser.update(dt)

                    inside_now = False
                    if shield_contour is not None and cv2.contourArea(shield_contour) > 1000:
                        impact_coords = (float(laser.pos[0]), float(laser.pos[1]))
                        inside_now = cv2.pointPolygonTest(shield_contour, impact_coords, False) >= 0

                    if inside_now:
                        normal = self.compute_surface_normal(shield_contour, laser.pos)
                        if np.dot(laser.vel, normal) > 0:
                            normal = -normal
                        dot_product = np.dot(laser.vel, normal)
                        laser.vel = laser.vel - (2 * dot_product * normal)

                        if not laser.was_inside:
                            # Fresh impact: standard block, push out a bit, award a point
                            laser.pos += normal * (laser.radius + 8)
                            self.sound_deflect.play()
                            self.score += 1
                            self.combo += 1
                            self.best_combo = max(self.best_combo, self.combo)

                            self.spawn_burst(laser.pos[0], laser.pos[1], NEON_GREEN, count=16)
                            label = "BLOCKED!" if self.combo < 3 else f"COMBO x{self.combo}!"
                            color = NEON_GREEN if self.combo < 3 else NEON_YELLOW
                            self.floating_texts.append(
                                FloatingText(laser.pos[0], laser.pos[1] - 20, label, color,
                                             size=24 if self.combo < 3 else 30))
                            if self.combo >= 3:
                                self.sound_combo.play()
                            self.trigger_shake(4, 0.08)

                            laser.was_inside = True
                            laser.held_frames = 1
                        else:
                            # Trapped: ball is being "held" inside the hand — keep juggling it
                            laser.pos += normal * (laser.radius + 4)
                            laser.held_frames += 1

                            if laser.held_frames % CATCH_BONUS_INTERVAL == 0:
                                self.score += 1
                                self.combo += 1
                                self.best_combo = max(self.best_combo, self.combo)
                                self.sound_catch.play()
                                self.spawn_burst(laser.pos[0], laser.pos[1], NEON_PURPLE, count=8)
                                self.floating_texts.append(
                                    FloatingText(laser.pos[0], laser.pos[1] - 20, "EXTRA!", NEON_PURPLE, size=22))

                            if laser.held_frames > MAX_HELD_FRAMES:
                                # Been trapped too long — force it back out cleanly
                                laser.was_inside = False
                                laser.held_frames = 0
                    else:
                        laser.was_inside = False
                        laser.held_frames = 0

                        # Only an undeflected, untrapped ball can reach the head
                        if self.head_center is not None and laser.active and self.invuln_timer <= 0:
                            dist = np.linalg.norm(laser.pos - np.array(self.head_center))
                            if dist < self.head_radius:
                                self.lives -= 1
                                self.combo = 0
                                self.invuln_timer = INVULN_DURATION
                                self.sound_hit.play()
                                laser.active = False
                                self.spawn_burst(laser.pos[0], laser.pos[1], NEON_RED, count=40)
                                self.floating_texts.append(
                                    FloatingText(laser.pos[0], laser.pos[1] - 20, "-1 LIFE", NEON_RED, size=32))

                                if self.lives <= 0:
                                    self.state = STATE_GAMEOVER
                                    self.trigger_shake(18, 0.4)
                                else:
                                    self.trigger_shake(10, 0.25)

                    laser.draw(self.screen)

                self.lasers = [l for l in self.lasers if l.active]
                self.draw_hud()

            elif self.state == STATE_WAITING:
                glow_alpha = 150 + int(80 * math.sin(self.pulse_t * 3))
                title_color = (0, min(255, 200 + glow_alpha // 4), 255)
                self.draw_centered_text("RICOCHET AR", self.font_huge, title_color, y_offset=-60, glow=True)
                pulse_scale = 1 + 0.05 * math.sin(self.pulse_t * 4)
                prompt_font = pygame.font.SysFont("arial", int(28 * pulse_scale), bold=True)
                self.draw_centered_text("\u270b SHOW AN OPEN PALM TO START", prompt_font, NEON_YELLOW, y_offset=20)

            elif self.state == STATE_GAMEOVER:
                for laser in self.lasers:
                    laser.draw(self.screen)
                shake_red = 60 + int(40 * math.sin(self.pulse_t * 10))
                self.draw_centered_text("GAME OVER", self.font_huge, (255, shake_red, shake_red), y_offset=-70,
                                         glow=True)
                self.draw_centered_text(f"BLOCKED {self.score}  |  BEST COMBO x{self.best_combo}",
                                         self.font_small, (255, 255, 255), y_offset=0)
                pulse_scale = 1 + 0.05 * math.sin(self.pulse_t * 4)
                prompt_font = pygame.font.SysFont("arial", int(26 * pulse_scale), bold=True)
                self.draw_centered_text("\u270b SHOW PALM TO RETRY", prompt_font, NEON_YELLOW, y_offset=50)

            # --- Particles & floating text ---
            self.particles = [p for p in self.particles if p.update(dt)]
            for p in self.particles:
                p.draw(self.screen)

            self.floating_texts = [t for t in self.floating_texts if t.update(dt)]
            for t in self.floating_texts:
                t.draw(self.screen)

            # --- Cyberpunk overlay layers ---
            self.screen.blit(self.scanline_surface, (0, 0))
            self.screen.blit(self.vignette_surface, (0, 0))

            pygame.display.flip()

        self.cap.release()
        self.hands.close()
        self.face_detector.close()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = RicochetEngine()
    app.run()