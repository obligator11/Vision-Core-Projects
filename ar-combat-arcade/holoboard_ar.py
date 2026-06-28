import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import sys

# ─── PROCEDURAL AUDIO SYNTHESIZER ─────────────────────────────────────────────
class AudioManager:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self.sample_rate = 44100
        self.snd_hover    = self._gen(600,  0.05, 'sine',     0.10)
        self.snd_click    = self._gen(1200, 0.10, 'square',   0.20)
        self.snd_startup  = self._gen(300,  0.40, 'sawtooth', 0.30)
        self.snd_calibrate= self._gen(880,  0.15, 'sine',     0.25)
        self.has_played_startup = False

    def _gen(self, freq, dur, wave='sine', vol=0.5):
        n = int(self.sample_rate * dur)
        t = np.linspace(0, dur, n, False)
        if wave == 'sine':
            w = np.sin(2 * np.pi * freq * t)
        elif wave == 'square':
            w = np.sign(np.sin(2 * np.pi * freq * t))
        else:
            w = 2 * (t * freq - np.floor(t * freq + 0.5))
        w = w * np.linspace(1.0, 0.0, n) * vol
        arr = np.zeros((n, 2), dtype=np.int16)
        arr[:, 0] = arr[:, 1] = (w * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(arr)


# ─── CYBER UI with reactive wave ──────────────────────────────────────────────
class CyberUI:
    """
    Buttons control the wave in real-time:
      SYSTEM OVERRIDE  → turbo speed  (5× faster scroll)
      MATRIX LINK      → tall amplitude + doubled frequency (chaotic peaks)
      SHIELD GENERATOR → adds a noise/static layer on top of the sine
    All three active → full chaos mode (all effects combined, wave turns red)
    """
    WAVE_X0, WAVE_X1 = 430, 780   # wave panel horizontal bounds
    WAVE_CY          = 300         # wave centre Y on the 800×600 surface
    WAVE_PANEL_Y0    = 60
    WAVE_PANEL_Y1    = 540

    def __init__(self, width=800, height=600):
        self.w, self.h = width, height
        self.surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)

        # Button definitions — index maps to effect:
        #   0 = SYSTEM OVERRIDE  (speed)
        #   1 = MATRIX LINK      (amplitude / frequency)
        #   2 = SHIELD GENERATOR (static noise)
        self.buttons = [
            {"rect": pygame.Rect(80, 100, 270, 75), "label": "SYSTEM OVERRIDE",
             "color": (0,255,255),  "active": False,
             "desc":  "FREQ ×5"},
            {"rect": pygame.Rect(80, 215, 270, 75), "label": "MATRIX LINK",
             "color": (0,255,255),  "active": False,
             "desc":  "AMP ×3 + HARMONICS"},
            {"rect": pygame.Rect(80, 330, 270, 75), "label": "SHIELD GENERATOR",
             "color": (0,255,255),  "active": False,
             "desc":  "NOISE LAYER"},
        ]

        pygame.font.init()
        self.font_large = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.hovered_idx      = -1
        self.last_hovered_idx = -1

        # Smooth transition targets for wave params
        self._speed_current = 3.0
        self._amp_current   = 45.0
        self._noise_alpha   = 0.0   # 0..1
        self._rng           = np.random.default_rng(42)

    # ── internal helpers ───────────────────────────────────────────────────────
    def _lerp(self, a, b, k=0.08):
        return a + (b - a) * k

    def _wave_color(self, active_mask):
        """Returns wave colour based on which buttons are on."""
        r, g, b = active_mask
        if r and g and b:   return (255,  60,  60)   # ALL ON → red chaos
        if r and g:         return (255, 160,   0)   # orange
        if r and b:         return (200,   0, 255)   # purple
        if g and b:         return (  0, 255, 200)   # teal
        if r:               return (255, 255,   0)   # yellow (speed)
        if g:               return (  0, 200, 255)   # cyan-blue (amp)
        if b:               return (180, 255, 100)   # lime (noise)
        return (0, 255, 100)                          # default green

    # ── main draw ─────────────────────────────────────────────────────────────
    def update_and_draw(self, mapped_pt, is_pinched, audio_manager):
        self.surface.fill((10, 20, 30, 180))
        pygame.draw.rect(self.surface, (0,255,255), (0,0,self.w,self.h), 4)

        x, y = mapped_pt if mapped_pt else (-100, -100)
        self.hovered_idx = -1

        # ── buttons ───────────────────────────────────────────────────────────
        for i, btn in enumerate(self.buttons):
            r = btn["rect"]
            is_hovered = r.collidepoint(x, y)
            if is_hovered:
                self.hovered_idx = i
                if self.hovered_idx != self.last_hovered_idx:
                    audio_manager.snd_hover.play()
                if is_pinched:
                    btn["active"] = not btn["active"]
                    audio_manager.snd_click.play()

            col = (255, 0, 255) if btn["active"] else btn["color"]
            pygame.draw.rect(self.surface, col, r, 0 if is_hovered else 2)

            lbl = self.font_large.render(btn["label"], True,
                                         (255,255,255) if is_hovered else col)
            self.surface.blit(lbl, (r.x + 14, r.y + 14))

            # Status badge
            status = "● ON" if btn["active"] else "○ OFF"
            s_col  = (0,255,100) if btn["active"] else (120,120,120)
            badge  = self.font_small.render(f"{status}  {btn['desc']}", True, s_col)
            self.surface.blit(badge, (r.x + 14, r.y + 46))

        self.last_hovered_idx = self.hovered_idx

        # ── wave panel border ─────────────────────────────────────────────────
        panel_rect = pygame.Rect(self.WAVE_X0 - 10, self.WAVE_PANEL_Y0,
                                 self.WAVE_X1 - self.WAVE_X0 + 20,
                                 self.WAVE_PANEL_Y1 - self.WAVE_PANEL_Y0)
        pygame.draw.rect(self.surface, (30, 60, 80, 160), panel_rect)
        pygame.draw.rect(self.surface, (0, 180, 180), panel_rect, 1)

        # Panel label
        a0, a1, a2 = [btn["active"] for btn in self.buttons]
        panel_lbl = self.font_small.render("WAVE ANALYZER", True, (0,200,200))
        self.surface.blit(panel_lbl, (self.WAVE_X0, self.WAVE_PANEL_Y0 + 6))

        # ── compute wave params with smooth lerp ───────────────────────────────
        target_speed = 15.0 if a0 else 3.0
        target_amp   = 130.0 if a1 else 45.0
        self._speed_current = self._lerp(self._speed_current, target_speed)
        self._amp_current   = self._lerp(self._amp_current,   target_amp)
        self._noise_alpha   = self._lerp(self._noise_alpha, 1.0 if a2 else 0.0)

        t   = time.time()
        spd = self._speed_current
        amp = self._amp_current
        freq_mult = 2.5 if a1 else 1.0   # MATRIX LINK doubles harmonic density

        wave_col = self._wave_color((a0, a1, a2))

        # ── draw wave ─────────────────────────────────────────────────────────
        xs = range(self.WAVE_X0, self.WAVE_X1, 3)
        pts_main = []
        pts_noise = []

        for wx in xs:
            # normalise position 0..1 across the panel
            norm = (wx - self.WAVE_X0) / max(1, self.WAVE_X1 - self.WAVE_X0)

            # Base sine
            phase = norm * math.pi * 2 * freq_mult * 3.5 + t * spd
            wy    = self.WAVE_CY + math.sin(phase) * amp

            # MATRIX LINK harmonic — adds a second sine at 3× freq, 0.35× amp
            if a1:
                wy += math.sin(phase * 3.0 + t * spd * 0.7) * amp * 0.35

            pts_main.append((wx, int(wy)))

            # SHIELD GENERATOR noise layer
            if self._noise_alpha > 0.01:
                noise_y = wy + (self._rng.random() - 0.5) * 60 * self._noise_alpha
                pts_noise.append((wx, int(noise_y)))
            else:
                pts_noise.append((wx, int(wy)))

        # Draw glow (thick, dimmed)
        glow_col = tuple(max(0, c // 3) for c in wave_col)
        if len(pts_main) > 1:
            pygame.draw.lines(self.surface, glow_col, False, pts_main, 7)

        # Noise layer (thin, slightly different colour)
        if self._noise_alpha > 0.01 and len(pts_noise) > 1:
            n_col = tuple(min(255, c + 60) for c in wave_col)
            pygame.draw.lines(self.surface, n_col, False, pts_noise, 1)

        # Main wave (crisp, on top)
        if len(pts_main) > 1:
            pygame.draw.lines(self.surface, wave_col, False, pts_main, 3)

        # Centre line (static reference)
        pygame.draw.line(self.surface, (40, 80, 80),
                         (self.WAVE_X0, self.WAVE_CY),
                         (self.WAVE_X1, self.WAVE_CY), 1)

        # Live readout
        info = (f"SPD:{self._speed_current:5.1f}  "
                f"AMP:{self._amp_current:5.1f}  "
                f"NOISE:{int(self._noise_alpha*100):3d}%")
        info_surf = self.font_small.render(info, True, (100,200,200))
        self.surface.blit(info_surf, (self.WAVE_X0, self.WAVE_PANEL_Y1 - 22))

        # ── cursor ────────────────────────────────────────────────────────────
        if mapped_pt:
            ix, iy = int(x), int(y)
            pygame.draw.circle(self.surface, (255,0,0), (ix,iy), 10)
            pygame.draw.line(self.surface, (255,0,0), (ix-20,iy), (ix+20,iy), 2)
            pygame.draw.line(self.surface, (255,0,0), (ix,iy-20), (ix,iy+20), 2)

        return self.surface


# ─── MANUAL CALIBRATION MANAGER ───────────────────────────────────────────────
class CalibrationManager:
    """
    Lets you define 4 corners either by:
      • MOUSE CLICK  (left-click to place each corner in sequence)
      • PINCH HOLD   (index+thumb pinch for > PINCH_HOLD_SEC seconds)
      • KEYBOARD     press 'R' to reset / restart calibration

    Corner order: TL → TR → BR → BL  (matches src_pts warp layout)
    """
    PINCH_HOLD_SEC = 1.0   # hold pinch for this long to register a corner
    CORNER_LABELS  = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    CORNER_COLORS  = [(0,255,255),(255,255,0),(255,0,255),(0,255,0)]

    def __init__(self):
        self.corners      = []     # list of (x,y) in camera space
        self.calibrated   = False
        self._pinch_start = None   # timestamp when current pinch began

    # ── public API ────────────────────────────────────────────────────────────
    def reset(self):
        self.corners      = []
        self.calibrated   = False
        self._pinch_start = None

    @property
    def next_idx(self):
        return len(self.corners)

    def add_corner_mouse(self, x, y):
        """Call when user left-clicks during calibration."""
        if self.calibrated:
            return
        self.corners.append((float(x), float(y)))
        if len(self.corners) == 4:
            self.calibrated = True

    def update_pinch(self, finger_pt, is_pinching, audio_manager):
        """
        Call every frame with current index-finger tip and pinch state.
        Returns True the moment a corner is registered by pinch.
        """
        if self.calibrated or finger_pt is None:
            self._pinch_start = None
            return False

        if is_pinching:
            if self._pinch_start is None:
                self._pinch_start = time.time()
            elif time.time() - self._pinch_start >= self.PINCH_HOLD_SEC:
                self.corners.append((float(finger_pt[0]), float(finger_pt[1])))
                self._pinch_start = None
                audio_manager.snd_calibrate.play()
                if len(self.corners) == 4:
                    self.calibrated = True
                return True
        else:
            self._pinch_start = None

        return False

    def pinch_progress(self):
        """0.0 → 1.0 fill for the hold-progress ring."""
        if self._pinch_start is None:
            return 0.0
        return min(1.0, (time.time() - self._pinch_start) / self.PINCH_HOLD_SEC)

    def get_homography(self, ui_w, ui_h):
        """Returns (H, H_inv) once calibrated, else (None, None)."""
        if not self.calibrated:
            return None, None
        # dst = corners in camera space (TL TR BR BL)
        dst = np.array(self.corners, dtype=np.float32)
        src = np.array([[0,0],[ui_w,0],[ui_w,ui_h],[0,ui_h]], dtype=np.float32)
        H   = cv2.getPerspectiveTransform(src, dst)
        H_inv = np.linalg.inv(H)
        return H, H_inv

    # ── overlay drawing ───────────────────────────────────────────────────────
    def draw_overlay(self, frame, finger_pt, is_pinching):
        """Draws calibration guide directly onto the OpenCV frame."""
        h, w = frame.shape[:2]

        # Dim overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Already placed corners
        for i, (cx,cy) in enumerate(self.corners):
            col = self.CORNER_COLORS[i]
            cv2.circle(frame, (int(cx),int(cy)), 14, col, -1)
            cv2.putText(frame, self.CORNER_LABELS[i], (int(cx)+16, int(cy)+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        # Draw connecting lines if ≥2 corners placed
        if len(self.corners) >= 2:
            for i in range(len(self.corners)-1):
                p1 = (int(self.corners[i][0]),   int(self.corners[i][1]))
                p2 = (int(self.corners[i+1][0]), int(self.corners[i+1][1]))
                cv2.line(frame, p1, p2, (180,180,180), 1, cv2.LINE_AA)

        # Next corner guide
        ni = self.next_idx
        if ni < 4:
            col   = self.CORNER_COLORS[ni]
            label = self.CORNER_LABELS[ni]

            # Target corner guide markers at screen edges
            guide_positions = [
                (80,  60),          # TL
                (w-80, 60),         # TR
                (w-80, h-60),       # BR
                (80,  h-60),        # BL
            ]
            gx, gy = guide_positions[ni]
            cv2.drawMarker(frame, (gx,gy), col,
                           cv2.MARKER_CROSS, 40, 2, cv2.LINE_AA)

            # Instruction banner
            inst = f"[{ni+1}/4]  Place finger/click on:  {label}"
            cv2.rectangle(frame, (0, h-70), (w, h), (0,0,0), -1)
            cv2.putText(frame, inst, (20, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
            cv2.putText(frame, "LEFT-CLICK  or  PINCH & HOLD 1 sec  |  R = reset",
                        (20, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)

            # Pinch hold progress ring around fingertip
            if finger_pt and is_pinching:
                prog = self.pinch_progress()
                fx, fy = int(finger_pt[0]), int(finger_pt[1])
                cv2.circle(frame, (fx,fy), 22, (50,50,50), 3)
                if prog > 0:
                    angle = int(360 * prog)
                    cv2.ellipse(frame, (fx,fy), (22,22), -90, 0, angle, col, 3, cv2.LINE_AA)
                cv2.putText(frame, f"{int(prog*100)}%", (fx-14, fy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Title
        cv2.putText(frame, "HOLOBOARD CALIBRATION",
                    (w//2 - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,255,255), 2, cv2.LINE_AA)


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
class HoloBoardApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1280,720), pygame.RESIZABLE)
        pygame.display.set_caption("HoloBoard AR Interface  |  R=recalibrate  Q=quit")

        self.audio = AudioManager()
        self.ui    = CyberUI(800, 600)
        self.calib = CalibrationManager()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Pinch debounce — NO time.sleep inside render loop ever
        self._last_pinch_time = 0.0
        self._pinch_debounce  = 0.30   # seconds between UI clicks

    # ── normalized pinch (robust, hand-size-independent) ─────────────────────
    def _check_pinch(self, lm, w, h):
        """
        Returns (finger_pt, is_pinching).
        Pinch distance is normalized by the palm diagonal so it works
        regardless of how close/far the hand is from the camera.
        """
        idx = lm.landmark[8]   # index tip
        thb = lm.landmark[4]   # thumb tip
        wrist = lm.landmark[0]
        mid_mcp = lm.landmark[9]

        fx, fy = idx.x * w, idx.y * h
        tx, ty = thb.x * w, thb.y * h

        # Palm reference size = wrist → middle-MCP distance in pixels
        palm_dx = (mid_mcp.x - wrist.x) * w
        palm_dy = (mid_mcp.y - wrist.y) * h
        palm_size = math.hypot(palm_dx, palm_dy) + 1e-6

        pinch_dist = math.hypot(fx - tx, fy - ty)
        ratio      = pinch_dist / palm_size

        # Empirically ~0.25 is a comfortable closed pinch
        is_pinching = ratio < 0.28
        return (fx, fy), is_pinching

    def _pygame_surf_to_cv(self, surface):
        view  = pygame.surfarray.array3d(surface).transpose([1,0,2])
        alpha = pygame.surfarray.pixels_alpha(surface).transpose([1,0])
        return np.dstack((cv2.cvtColor(view, cv2.COLOR_RGB2BGR), alpha))

    def run(self):
        clock   = pygame.time.Clock()
        running = True

        while running:
            # ── events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.calib.reset()
                        self.audio.has_played_startup = False

                # Mouse click → place corner during calibration
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.calib.calibrated:
                        # Convert Pygame window coords → camera coords
                        win_w, win_h = self.screen.get_size()
                        mx, my = event.pos
                        ret_peek, fr_peek = self.cap.read()
                        if ret_peek:
                            cam_w = fr_peek.shape[1]
                            cam_h = fr_peek.shape[0]
                        else:
                            cam_w, cam_h = 1280, 720
                        cx = mx * cam_w / win_w
                        cy = my * cam_h / win_h
                        self.calib.add_corner_mouse(cx, cy)
                        self.audio.snd_calibrate.play()

            # ── camera frame ──────────────────────────────────────────────────
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h_cam, w_cam = frame.shape[:2]

            # ── hand tracking ─────────────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            finger_pt  = None
            is_pinching = False

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                finger_pt, is_pinching = self._check_pinch(lm, w_cam, h_cam)

                # Visual feedback on fingertip
                if finger_pt:
                    color = (0,255,0) if is_pinching else (255,100,0)
                    cv2.circle(frame, (int(finger_pt[0]), int(finger_pt[1])), 12, color, -1)

            # ── calibration phase ─────────────────────────────────────────────
            if not self.calib.calibrated:
                self.calib.update_pinch(finger_pt, is_pinching, self.audio)
                self.calib.draw_overlay(frame, finger_pt, is_pinching)

            else:
                # ── AR hologram phase ─────────────────────────────────────────
                H, H_inv = self.calib.get_homography(self.ui.w, self.ui.h)

                mapped_ui_pt    = None
                debounced_pinch = False

                if finger_pt is not None and H_inv is not None:
                    f_h = np.array([finger_pt[0], finger_pt[1], 1.0])
                    ui_h = H_inv.dot(f_h)
                    ui_x = ui_h[0] / ui_h[2]
                    ui_y = ui_h[1] / ui_h[2]
                    if 0 <= ui_x <= self.ui.w and 0 <= ui_y <= self.ui.h:
                        mapped_ui_pt = (ui_x, ui_y)

                    now = time.time()
                    if is_pinching and (now - self._last_pinch_time) > self._pinch_debounce:
                        debounced_pinch    = True
                        self._last_pinch_time = now

                if not self.audio.has_played_startup:
                    self.audio.snd_startup.play()
                    self.audio.has_played_startup = True

                ui_surf = self.ui.update_and_draw(mapped_ui_pt, debounced_pinch, self.audio)
                ui_cv   = self._pygame_surf_to_cv(ui_surf)

                warped = cv2.warpPerspective(ui_cv, H, (w_cam, h_cam))
                alpha  = warped[:,:,3] / 255.0
                for c in range(3):
                    frame[:,:,c] = (alpha * warped[:,:,c] + (1-alpha) * frame[:,:,c]).astype(np.uint8)

                # Draw calibration dots so user knows where corners are
                for i, (cx,cy) in enumerate(self.calib.corners):
                    cv2.circle(frame, (int(cx),int(cy)), 8, self.calib.CORNER_COLORS[i], -1)

                # HUD
                cv2.putText(frame, "R = recalibrate  |  Q = quit",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

            # ── render to pygame window ───────────────────────────────────────
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
            win_w, win_h = self.screen.get_size()
            scaled = pygame.transform.smoothscale(frame_surf, (win_w, win_h))
            self.screen.blit(scaled, (0,0))
            pygame.display.flip()
            clock.tick(60)

        self.cap.release()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = HoloBoardApp()
    app.run()