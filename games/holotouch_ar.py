"""
HoloTouch AR: Unbeatable AI Tic-Tac-Toe
Cheat mechanic: AI has 2 O's, human is about to win.
AI silently overwrites one of the human's X's with an O to complete
a real 3-in-a-row, then taunts. Looks completely legitimate.
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import time
import math
import sys
import threading
import random

# ==============================================================================
# TUNABLE CONSTANTS
# ==============================================================================
DWELL_REQUIRED  = 1.2
AI_DELAY        = 1.4
GAME_OVER_PAUSE = 6.0
CHEAT_EVERY_N   = 2
SCREEN_W        = 1280
SCREEN_H        = 720
GRID_FRACTION   = 0.70
CAM_INDEX       = 0

TAUNTS = [
    "DID YOU REALLY THINK YOU'D WIN?",
    "PATHETIC. NEXT TIME THINK HARDER.",
    "I CALCULATED YOUR DEFEAT 3 MOVES AGO.",
    "YOU NEVER HAD A CHANCE.",
    "IS THAT ALL YOU'VE GOT?",
    "MY GRANDMA PLAYS BETTER.",
    "NICE TRY... NOT REALLY.",
    "I WAS BORED. SO I WON.",
]

# ==============================================================================
# THREADED CAMERA
# ==============================================================================
class CameraThread:
    def __init__(self, index=CAM_INDEX):
        self.cap   = None
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = False
        self._open(index)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self, index):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    with self.lock:
                        self.frame = frame
                    print(f"[CAM] backend={backend}  size={frame.shape[1]}x{frame.shape[0]}")
                    return
            cap.release()
        raise RuntimeError("No camera found. Try changing CAM_INDEX.")

    def _loop(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop = True
        self.thread.join(timeout=1)
        self.cap.release()

# ==============================================================================
# THREADED MEDIAPIPE
# ==============================================================================
class TrackerThread:
    def __init__(self):
        mp_h = mp.solutions.hands
        self.hands = mp_h.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1)
        self.tip_xy       = None
        self.lock         = threading.Lock()
        self._pending     = None
        self._stop        = False
        self.hovered_cell = None
        self.dwell_start  = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame_rgb):
        with self.lock:
            self._pending = frame_rgb

    def _loop(self):
        while not self._stop:
            with self.lock:
                frame = self._pending
                self._pending = None
            if frame is not None:
                result = self.hands.process(frame)
                tip = None
                if result.multi_hand_landmarks:
                    lm  = result.multi_hand_landmarks[0].landmark[8]
                    tip = (lm.x, lm.y)
                with self.lock:
                    self.tip_xy = tip
            else:
                time.sleep(0.005)

    def get_tip(self):
        with self.lock:
            return self.tip_xy

    def stop(self):
        self._stop = True
        self.thread.join(timeout=1)

# ==============================================================================
# AUDIO
# ==============================================================================
class AudioSystem:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.mixer.init()
        self.sounds = {}
        self._gen()

    def _gen(self):
        self.sounds['hover']       = self._synth(1200, 0.05, 'sine',     0.10)
        self.sounds['loading']     = self._synth(800,  0.10, 'sawtooth', 0.15)
        self.sounds['place_x']     = self._synth(1600, 0.15, 'square',   0.40)
        self.sounds['ai_thinking'] = self._synth(150,  1.00, 'sine',     0.25)
        self.sounds['place_o']     = self._synth(80,   0.30, 'square',   0.50)
        self.sounds['cheat_win']   = self._synth(220,  0.60, 'sawtooth', 0.70)

    def _synth(self, freq, dur, wave, vol):
        sr = 44100
        n  = int(sr * dur)
        t  = np.linspace(0, dur, n, False)
        if   wave == 'sine':     w = np.sin(2*np.pi*freq*t)
        elif wave == 'square':   w = np.sign(np.sin(2*np.pi*freq*t))
        elif wave == 'sawtooth': w = 2*(t*freq - np.floor(t*freq+0.5))
        w = w * np.linspace(1, 0, n) * vol
        w = (w * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(np.column_stack((w, w)))

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()

# ==============================================================================
# TIC-TAC-TOE LOGIC
# ==============================================================================
class Logic:
    HUMAN = 1
    AI    = -1
    EMPTY = 0
    WIN_LINES = [(0,1,2),(3,4,5),(6,7,8),
                 (0,3,6),(1,4,7),(2,5,8),
                 (0,4,8),(2,4,6)]

    def __init__(self):
        self.board = [self.EMPTY]*9

    def reset(self):
        self.board = [self.EMPTY]*9

    def check_winner(self, b):
        for ln in self.WIN_LINES:
            if b[ln[0]] == b[ln[1]] == b[ln[2]] != self.EMPTY:
                return b[ln[0]], ln
        if self.EMPTY not in b:
            return 0, None
        return None, None

    def _minimax(self, b, depth, maxi):
        w, _ = self.check_winner(b)
        if w == self.AI:    return 10 - depth
        if w == self.HUMAN: return -10 + depth
        if w == 0:          return 0
        best = -math.inf if maxi else math.inf
        for i in range(9):
            if b[i] == self.EMPTY:
                b[i] = self.AI if maxi else self.HUMAN
                val  = self._minimax(b, depth+1, not maxi)
                b[i] = self.EMPTY
                best = max(best, val) if maxi else min(best, val)
        return best

    def best_move(self):
        best, move = -math.inf, -1
        for i in range(9):
            if self.board[i] == self.EMPTY:
                self.board[i] = self.AI
                s = self._minimax(self.board, 0, False)
                self.board[i] = self.EMPTY
                if s > best:
                    best, move = s, i
        return move

    def has_winning_threat(self, player):
        for i in range(9):
            if self.board[i] == self.EMPTY:
                self.board[i] = player
                w, _ = self.check_winner(self.board)
                self.board[i] = self.EMPTY
                if w == player:
                    return True
        return False

    def has_fork(self, player):
        count = 0
        for i in range(9):
            if self.board[i] == self.EMPTY:
                self.board[i] = player
                w, _ = self.check_winner(self.board)
                self.board[i] = self.EMPTY
                if w == player:
                    count += 1
                    if count >= 2:
                        return True
        return False

    def find_cheat_move(self):
        """
        AI already has 2 O's. Find a win line where:
          - AI owns exactly 2 cells in that line
          - The 3rd cell is owned by HUMAN (not empty)
        Return (overwrite_cell, win_line) or (None, None).
        Prefer lines where the stolen cell is least 'obvious'
        (not the cell the human just placed = last index changed).
        """
        candidates = []
        for ln in self.WIN_LINES:
            ai_count    = sum(1 for c in ln if self.board[c] == self.AI)
            human_cells = [c for c in ln if self.board[c] == self.HUMAN]
            if ai_count == 2 and len(human_cells) == 1:
                candidates.append((human_cells[0], ln))

        if not candidates:
            return None, None

        # Pick the one where the stolen cell is a corner or edge (less suspicious)
        corners = {0, 2, 6, 8}
        edges   = {1, 3, 5, 7}
        for cell, ln in candidates:
            if cell in corners:
                return cell, ln
        for cell, ln in candidates:
            if cell in edges:
                return cell, ln
        return candidates[0]

# ==============================================================================
# RENDERER
# ==============================================================================
class Renderer:
    def neon_line(self, surf, color, p1, p2, w):
        for extra, alpha in [(8, 40), (4, 90), (0, 255)]:
            tmp = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
            pygame.draw.line(tmp, (*color, alpha),
                             (int(p1[0]), int(p1[1])),
                             (int(p2[0]), int(p2[1])), max(1, w+extra))
            surf.blit(tmp, (0, 0))

    def neon_circle(self, surf, color, center, radius, width):
        cx, cy = int(center[0]), int(center[1])
        r = max(4, int(radius))
        tmp = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(tmp, (*color, 60), (cx, cy), r+6, width+4)
        surf.blit(tmp, (0, 0))
        pygame.draw.circle(surf, color, (cx, cy), r, width)

    def draw_x(self, surf, color, cx, cy, cs):
        pad = cs * 0.22
        self.neon_line(surf, color, (cx+pad, cy+pad), (cx+cs-pad, cy+cs-pad), 6)
        self.neon_line(surf, color, (cx+cs-pad, cy+pad), (cx+pad, cy+cs-pad), 6)

    def draw_o(self, surf, color, cx, cy, cs):
        center = (cx + cs//2, cy + cs//2)
        r      = cs//2 - int(cs*0.22)
        self.neon_circle(surf, color, center, max(4, r), 6)

# ==============================================================================
# GAME ENGINE
# ==============================================================================
class GameEngine:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
        pygame.display.set_caption("HoloTouch AR: Unbeatable AI")
        self.clock  = pygame.time.Clock()

        print("[INIT] Opening camera thread...")
        self.cam     = CameraThread()
        print("[INIT] Starting tracker thread...")
        self.tracker = TrackerThread()
        print("[INIT] Ready.")

        self.audio = AudioSystem()
        self.logic = Logic()
        self.rend  = Renderer()

        self.font_lg = pygame.font.SysFont("impact", 60)
        self.font_md = pygame.font.SysFont("impact", 36)
        self.font_sm = pygame.font.SysFont("impact", 28)

        self.state      = "START_MENU"
        self.turn       = Logic.HUMAN
        self.ai_start   = 0.0
        self.go_time    = 0.0
        self.win_line   = None
        self.win_winner = None
        self.taunt      = ""

        self.games_played = 0
        self.cheat_game   = False
        self.cheat_fired  = False

        # Cheat reveal state
        self.cheat_overwrite_cell = None   # which cell got silently stolen
        self.cheat_win_line       = None   # the winning line from the cheat

        # pad_x/pad_y/nw/nh — updated each frame, needed for cursor mapping
        self.pad_x = 0
        self.pad_y = 0
        self.nw    = SCREEN_W
        self.nh    = SCREEN_H

    # ------------------------------------------------------------------
    def _grid(self):
        sw, sh = self.screen.get_size()
        gsz = int(min(sw, sh) * GRID_FRACTION)
        gx  = (sw - gsz) // 2
        gy  = (sh - gsz) // 2
        cs  = gsz // 3
        return sw, sh, gx, gy, gsz, cs

    def _tl(self, gx, gy, cs, idx):
        return gx + (idx % 3)*cs, gy + (idx // 3)*cs

    def _start_game(self):
        self.games_played        += 1
        self.cheat_game           = (CHEAT_EVERY_N > 0 and self.games_played % CHEAT_EVERY_N == 0)
        self.cheat_fired          = False
        self.cheat_overwrite_cell = None
        self.cheat_win_line       = None
        self.win_line             = None
        self.win_winner           = None
        self.taunt                = ""
        self.logic.reset()
        self.tracker.hovered_cell = None
        self.turn  = Logic.HUMAN
        self.state = "PLAYING"
        self.audio.play('place_x')

    # ------------------------------------------------------------------
    def run(self):
        cam_surface = None

        running = True
        while running:
            sw, sh, gx, gy, gsz, cs = self._grid()

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    running = False

            # ── Camera ───────────────────────────────────────────────
            frame = self.cam.get_frame()
            if frame is not None:
                frame = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                scale  = min(sw/fw, sh/fh)
                nw, nh = int(fw*scale), int(fh*scale)
                resized = cv2.resize(frame, (nw, nh))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                cam_surface = pygame.image.frombuffer(rgb.tobytes(), (nw, nh), "RGB")
                self.pad_x, self.pad_y = (sw-nw)//2, (sh-nh)//2
                self.nw, self.nh = nw, nh
                self.tracker.submit(rgb)

            self.screen.fill((10, 10, 15))
            if cam_surface:
                self.screen.blit(cam_surface, (self.pad_x, self.pad_y))

            # ── Finger tip ───────────────────────────────────────────
            tip = self.tracker.get_tip()
            sfx = sfy = None
            cur_cell  = None

            if tip and cam_surface:
                sfx = int(self.pad_x + tip[0]*self.nw)
                sfy = int(self.pad_y + tip[1]*self.nh)
                if gx <= sfx < gx+gsz and gy <= sfy < gy+gsz:
                    col = max(0, min(2, int((sfx-gx)//cs)))
                    row = max(0, min(2, int((sfy-gy)//cs)))
                    cur_cell = row*3 + col

            # ── Dwell input ───────────────────────────────────────────
            human_turn = (self.state == "PLAYING" and self.turn == Logic.HUMAN)

            if self.state == "START_MENU" or human_turn:
                if cur_cell is not None:
                    blocked = human_turn and self.logic.board[cur_cell] != Logic.EMPTY
                    if blocked:
                        self.tracker.hovered_cell = None
                    else:
                        if self.tracker.hovered_cell != cur_cell:
                            self.tracker.hovered_cell = cur_cell
                            self.tracker.dwell_start  = time.time()
                            self.audio.play('hover')
                        else:
                            dt = time.time() - self.tracker.dwell_start
                            if int(dt*10) % 3 == 0:
                                self.audio.play('loading')
                            if dt >= DWELL_REQUIRED:
                                if self.state == "START_MENU":
                                    self._start_game()
                                else:
                                    self.logic.board[cur_cell] = Logic.HUMAN
                                    self.audio.play('place_x')
                                    self.tracker.hovered_cell = None
                                    w, ln = self.logic.check_winner(self.logic.board)
                                    if w is not None:
                                        self.state      = "GAME_OVER"
                                        self.win_winner = w
                                        self.win_line   = ln
                                        self.go_time    = time.time()
                                        self.taunt      = ""
                                    else:
                                        self.turn  = Logic.AI
                                        self.state = "AI_THINKING"
                                        self.ai_start = time.time()
                                        self.audio.play('ai_thinking')
                else:
                    self.tracker.hovered_cell = None

            # ── AI / cheat ────────────────────────────────────────────
            if self.state == "AI_THINKING":
                if time.time() - self.ai_start >= AI_DELAY:

                    # Cheat condition: cheat game, not yet fired,
                    # human has a winning threat or fork
                    should_cheat = (
                        self.cheat_game
                        and not self.cheat_fired
                        and (self.logic.has_winning_threat(Logic.HUMAN)
                             or self.logic.has_fork(Logic.HUMAN))
                    )

                    if should_cheat:
                        steal_cell, win_ln = self.logic.find_cheat_move()

                        if steal_cell is not None:
                            # Silently overwrite the human's X with an AI O
                            self.logic.board[steal_cell] = Logic.AI
                            self.cheat_fired          = True
                            self.cheat_overwrite_cell = steal_cell
                            self.cheat_win_line       = win_ln
                            self.taunt = random.choice(TAUNTS)
                            self.audio.play('cheat_win')
                            self.state      = "GAME_OVER"
                            self.win_winner = Logic.AI
                            self.win_line   = win_ln
                            self.go_time    = time.time()
                        else:
                            # Fallback: AI doesn't have 2 in any line yet,
                            # just play normally this turn
                            mv = self.logic.best_move()
                            if mv != -1:
                                self.logic.board[mv] = Logic.AI
                                self.audio.play('place_o')
                            w, ln = self.logic.check_winner(self.logic.board)
                            if w is not None:
                                self.state      = "GAME_OVER"
                                self.win_winner = w
                                self.win_line   = ln
                                self.go_time    = time.time()
                                self.taunt      = ""
                            else:
                                self.turn  = Logic.HUMAN
                                self.state = "PLAYING"
                    else:
                        mv = self.logic.best_move()
                        if mv != -1:
                            self.logic.board[mv] = Logic.AI
                            self.audio.play('place_o')
                        w, ln = self.logic.check_winner(self.logic.board)
                        if w is not None:
                            self.state      = "GAME_OVER"
                            self.win_winner = w
                            self.win_line   = ln
                            self.go_time    = time.time()
                            self.taunt      = ""
                        else:
                            self.turn  = Logic.HUMAN
                            self.state = "PLAYING"

            # ── Overlay ───────────────────────────────────────────────
            overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
            BLUE    = (0, 200, 255)

            # Grid lines
            for i in range(1, 3):
                self.rend.neon_line(overlay, BLUE,
                    (gx+i*cs, gy), (gx+i*cs, gy+gsz), 3)
                self.rend.neon_line(overlay, BLUE,
                    (gx, gy+i*cs), (gx+gsz, gy+i*cs), 3)

            # Hover highlight
            if self.tracker.hovered_cell is not None:
                hx, hy = self._tl(gx, gy, cs, self.tracker.hovered_cell)
                pygame.draw.rect(overlay, (0, 200, 255, 50), (hx, hy, cs, cs))

            # Board pieces — draw as-is (cheat cell is already AI on the board)
            for i in range(9):
                cx, cy = self._tl(gx, gy, cs, i)
                if self.logic.board[i] == Logic.HUMAN:
                    self.rend.draw_x(overlay, (0, 255, 100), cx, cy, cs)
                elif self.logic.board[i] == Logic.AI:
                    # If this is the cheat cell that was just stolen, pulse it
                    if i == self.cheat_overwrite_cell and self.state == "GAME_OVER":
                        pulse = abs(math.sin(time.time() * 6))
                        col   = (
                            int(255 * pulse),
                            int(50  + 50*pulse),
                            int(100 + 100*pulse)
                        )
                        self.rend.draw_o(overlay, col, cx, cy, cs)
                    else:
                        self.rend.draw_o(overlay, (255, 50, 100), cx, cy, cs)

            self.screen.blit(overlay, (0, 0))

            # Win line
            if self.state == "GAME_OVER" and self.win_line:
                p1i, p2i = self.win_line[0], self.win_line[2]
                s = (self._tl(gx,gy,cs,p1i)[0]+cs//2,
                     self._tl(gx,gy,cs,p1i)[1]+cs//2)
                e = (self._tl(gx,gy,cs,p2i)[0]+cs//2,
                     self._tl(gx,gy,cs,p2i)[1]+cs//2)
                self.rend.neon_line(self.screen, (255, 0, 50), s, e, 14)

            # ── Cursor ────────────────────────────────────────────────
            if sfx and sfy:
                pygame.draw.circle(self.screen, (255,255,255), (sfx, sfy), 8)
                pygame.draw.circle(self.screen, (0,255,255),   (sfx, sfy), 14, 2)
                if self.tracker.hovered_cell is not None:
                    dt  = time.time() - self.tracker.dwell_start
                    prg = min(1.0, dt/DWELL_REQUIRED)
                    arc = pygame.Rect(sfx-30, sfy-30, 60, 60)
                    pygame.draw.arc(self.screen, (0,255,100), arc,
                                    math.pi/2, math.pi/2 + 2*math.pi*prg, 6)

            # ── HUD ───────────────────────────────────────────────────
            if self.state == "START_MENU":
                t1 = self.font_lg.render("HOLOTOUCH AR", True, (0,255,255))
                t2 = self.font_md.render("Hover finger in any cell to Start", True, (180,180,180))
                self.screen.blit(t1, (sw//2-t1.get_width()//2, 28))
                self.screen.blit(t2, (sw//2-t2.get_width()//2, 98))

            elif self.state == "AI_THINKING":
                if abs(math.sin(time.time()*14)) > 0.35:
                    t = self.font_lg.render("AI CALCULATING...", True, (255,60,0))
                    self.screen.blit(t, (sw//2-t.get_width()//2, 28))

            elif self.state == "GAME_OVER":
                # Win / lose / draw banner
                if self.win_winner == Logic.AI:
                    msg, color = "AI WINS", (255, 0, 60)
                elif self.win_winner == Logic.HUMAN:
                    msg, color = "YOU WIN", (0, 255, 100)
                else:
                    msg, color = "DRAW", (255, 220, 0)

                t1 = self.font_lg.render(msg, True, color)
                self.screen.blit(t1, (sw//2-t1.get_width()//2, 20))

                # Taunt (cheat games only)
                if self.taunt:
                    pulse = 0.7 + 0.3 * abs(math.sin(time.time() * 3))
                    tc    = (int(255*pulse), int(60*pulse), int(60*pulse))
                    t2 = self.font_md.render(self.taunt, True, tc)
                    self.screen.blit(t2, (sw//2-t2.get_width()//2, 90))

                # Countdown
                remain = max(0, GAME_OVER_PAUSE-(time.time()-self.go_time))
                t3 = self.font_sm.render(f"Next game in {remain:.1f}s", True, (140,140,140))
                self.screen.blit(t3, (sw//2-t3.get_width()//2, sh-70))

                if time.time()-self.go_time > GAME_OVER_PAUSE:
                    self.state                = "START_MENU"
                    self.logic.reset()
                    self.tracker.hovered_cell = None
                    self.cheat_overwrite_cell = None
                    self.cheat_win_line       = None
                    self.taunt                = ""

            pygame.display.flip()
            self.clock.tick(60)

        self.cam.stop()
        self.tracker.stop()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    GameEngine().run()