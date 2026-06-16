import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import sys
import threading
from collections import deque

# --- DETAILED CONFIGURATION & INITIALIZATION ---
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

BASE_W, BASE_H = 1280, 720
screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
pygame.display.set_caption("⚡ Finger Hacker Grid ⚡")
clock = pygame.time.Clock()

# --- ASYNCHRONOUS HIGH-PERFORMANCE VIDEO STREAM POOL ---
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()

# --- PROCEDURAL REAL-TIME RAM AUDIO SYNTHESIZER ---
class SoundManager:
    SAMPLE_RATE = 44100

    @staticmethod
    def play_synth_sound(freq=440, duration=0.1, wave_type='sine', volume=0.3):
        threading.Thread(target=SoundManager._synthesize, args=(freq, duration, wave_type, volume), daemon=True).start()

    @staticmethod
    def _synthesize(freq, duration, wave_type, volume):
        frames = int(SoundManager.SAMPLE_RATE * duration)
        t = np.linspace(0, duration, frames, False)
        
        if wave_type == 'sine':
            data = np.sin(2 * np.pi * freq * t)
        elif wave_type == 'square':
            data = np.sign(np.sin(2 * np.pi * freq * t))
        elif wave_type == 'sawtooth':
            data = 2 * (t * freq - np.floor(t * freq + 0.5))
        else:
            data = np.random.normal(0, 1, frames)

        # Apply smooth structural decay envelope to prevent hardware clipping pops
        envelope = np.exp(-4 * np.linspace(0, 1, frames))
        audio_array = (data * envelope * volume * 32767).astype(np.int16)
        
        # Balance dual stereo streams smoothly
        stereo_buffer = np.zeros((frames, 2), dtype=np.int16)
        stereo_buffer[:, 0] = audio_array
        stereo_buffer[:, 1] = audio_array
        
        sound = pygame.sndarray.make_sound(stereo_buffer)
        sound.play()

# --- CYBER MATRIX NODES & TRAPS ARCHITECTURE ---
class GridNode:
    def __init__(self, id_val, grid_x, grid_y, pixel_x, pixel_y, node_type='standard'):
        self.id = id_val
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = pixel_x
        self.y = pixel_y
        self.node_type = node_type  # 'start', 'standard', 'trap', 'core'
        self.hacked = False
        self.hack_progress = 0.0  # Normalized 0.0 -> 1.0
        self.pulse_phase = random.uniform(0, 2 * math.pi)

    def update(self, dt):
        self.pulse_phase += dt * 4.0
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi

    def draw(self, surface, base_radius=18):
        pulse_scale = 1.0 + 0.15 * math.sin(self.pulse_phase)
        radius = int(base_radius * pulse_scale)
        
        # Determine Color Palettes based on structural clearance state
        if self.node_type == 'start':
            color = (0, 255, 150)
        elif self.node_type == 'core':
            color = (0, 180, 255) if not self.hacked else (0, 255, 100)
        elif self.node_type == 'trap':
            color = (255, 0, 80)
        else:
            color = (0, 220, 220) if not self.hacked else (0, 255, 100)

        # Handle progress mapping visualization
        if self.hack_progress > 0.0 and not self.hacked:
            pygame.draw.circle(surface, (255, 200, 0), (self.x, self.y), radius + 6, 2)
            
        pygame.draw.circle(surface, color, (self.x, self.y), radius, 0)
        pygame.draw.circle(surface, (255, 255, 255), (self.x, self.y), int(radius * 0.4), 0)

# --- THE GAME ENGINE CORE ---
class FingerHackerGridGame:
    def __init__(self):
        self.nodes = {}
        self.connections = []
        self.active_node = None
        self.hovered_node = None
        
        self.score = 0
        self.level = 1
        self.time_remaining = 45.0
        self.glitch_timer = 0.0
        self.screen_shake = 0
        self.game_over = False
        self.win_state = False

        # CV Hand Interface Setup
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.cursor_smooth_x, self.cursor_smooth_y = BASE_W // 2, BASE_H // 2
        self.pinch_active = False
        
        # Adjustable Virtual Window Layout Specifications
        self.camera_scale = 0.25
        self.camera_dragging = False
        self.camera_rect = pygame.Rect(BASE_W - int(640 * self.camera_scale) - 20, 20, int(640 * self.camera_scale), int(480 * self.camera_scale))
        self.offset_x = 0
        self.offset_y = 0

        self.generate_matrix_level()

    def generate_matrix_level(self):
        self.nodes.clear()
        self.connections.clear()
        self.active_node = None
        self.time_remaining = max(15.0, 50.0 - (self.level * 4))
        
        cols, rows = 4, 3
        padding_x = BASE_W // (cols + 1)
        padding_y = (BASE_H - 150) // (rows + 1)

        node_id = 0
        for r in range(rows):
            for c in range(cols):
                px = padding_x * (c + 1) + random.randint(-20, 20)
                py = padding_y * (r + 1) + 100 + random.randint(-20, 20)
                
                # Determine special entity layout nodes
                node_type = 'standard'
                if r == 0 and c == 0:
                    node_type = 'start'
                elif r == rows - 1 and c == cols - 1:
                    node_type = 'core'
                elif random.random() < 0.22 and not (r == 0 and c == 0):
                    node_type = 'trap'
                    
                self.nodes[node_id] = GridNode(node_id, c, r, px, py, node_type)
                node_id += 1
                
        # Inject standard base node clear rules
        self.nodes[0].hacked = True
        self.active_node = self.nodes[0]

    def process_frame_landmarks(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_detector.process(rgb_frame)
        
        detected_coords = None
        self.pinch_active = False
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                index_tip = hand_lms.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_lms.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                
                # Mirror coordinates naturally
                cx, cy = int((1.0 - index_tip.x) * BASE_W), int(index_tip.y * BASE_H)
                detected_coords = (cx, cy)
                
                # Metric Space distance mapping formula
                dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                if dist < 0.04:
                    self.pinch_active = True
                break

        # Apply structural smoothing factors to clear pixel jumps
        if detected_coords:
            alpha = 0.28
            self.cursor_smooth_x = int(self.cursor_smooth_x + alpha * (detected_coords[0] - self.cursor_smooth_x))
            self.cursor_smooth_y = int(self.cursor_smooth_y + alpha * (detected_coords[1] - self.cursor_smooth_y))

    def update_game_logic(self, dt):
        if self.game_over or self.win_state:
            return

        self.time_remaining -= dt
        if self.time_remaining <= 0:
            self.time_remaining = 0
            self.game_over = True
            SoundManager.play_synth_sound(120, 0.8, 'sawtooth', 0.5)

        # Trigger glitch visualizations during critical counts
        if self.time_remaining < 10.0:
            if random.random() < 0.12:
                self.glitch_timer = random.uniform(0.05, 0.2)
                self.screen_shake = random.randint(4, 12)

        # Handle screen shake decay matrices
        if self.screen_shake > 0:
            self.screen_shake = int(self.screen_shake * 0.8)

        if self.glitch_timer > 0:
            self.glitch_timer -= dt

        # Evaluate interaction collision vectors
        self.hovered_node = None
        for node in self.nodes.values():
            node.update(dt)
            dist = math.hypot(self.cursor_smooth_x - node.x, self.cursor_smooth_y - node.y)
            if dist < 32:
                self.hovered_node = node

        # Handle gesture command actions
        if self.hovered_node is not None:
            node = self.hovered_node
            if self.pinch_active:
                if self.active_node and node != self.active_node:
                    # Connection conditions logic: spatial boundary validation
                    spatial_valid = math.hypot(self.active_node.x - node.x, self.active_node.y - node.y) < 380
                    if spatial_valid and (self.active_node.hacked or self.active_node.node_type == 'start'):
                        if (self.active_node.id, node.id) not in self.connections and (node.id, self.active_node.id) not in self.connections:
                            self.connections.append((self.active_node.id, node.id))
                            SoundManager.play_synth_sound(680, 0.08, 'sine', 0.25)
                        self.active_node = node
                
                # Progress structural bypass engine rules
                if not node.hacked:
                    if node.node_type == 'trap':
                        self.screen_shake = 20
                        self.glitch_timer = 0.4
                        self.time_remaining -= 8.0
                        SoundManager.play_synth_sound(90, 0.5, 'square', 0.4)
                        node.hacked = True  # Neutralize trap after fire
                    else:
                        node.hack_progress += dt * 0.85
                        if random.random() < 0.3:
                            SoundManager.play_synth_sound(random.randint(800, 1500), 0.02, 'square', 0.1)
                        if node.hack_progress >= 1.0:
                            node.hack_progress = 1.0
                            node.hacked = True
                            self.score += 150
                            SoundManager.play_synth_sound(880, 0.25, 'sine', 0.3)
                            
                            if node.node_type == 'core':
                                self.win_sequence_trigger()
            else:
                # Tap / Select logic execution state
                if pygame.mouse.get_pressed()[0] or random.random() < 0.02: # Air select verification trace
                    self.active_node = node

    def win_sequence_trigger(self):
        self.score += int(self.time_remaining * 20)
        self.level += 1
        SoundManager.play_synth_sound(1200, 0.15, 'sine', 0.4)
        pygame.time.wait(100)
        SoundManager.play_synth_sound(1600, 0.3, 'sine', 0.4)
        self.generate_matrix_level()

    def draw_hud(self, surface):
        # Background Grid Mesh Architecture
        grid_space = 40
        for x in range(0, BASE_W, grid_space):
            pygame.draw.line(surface, (10, 24, 34), (x, 0), (x, BASE_H))
        for y in range(0, BASE_H, grid_space):
            pygame.draw.line(surface, (10, 24, 34), (0, y), (BASE_W, y))

        # Render Active Verified Cyber Connections Pipeline
        for parent_id, child_id in self.connections:
            p_node = self.nodes[parent_id]
            c_node = self.nodes[child_id]
            color = (0, 255, 180) if p_node.hacked and c_node.hacked else (0, 130, 130)
            pygame.draw.line(surface, color, (p_node.x, p_node.y), (c_node.x, c_node.y), 4)

        # Draw Nodes Matrix Group
        for node in self.nodes.values():
            node.draw(surface)

        # Target Vector Track Rings
        if self.active_node:
            pygame.draw.circle(surface, (255, 255, 0), (self.active_node.x, self.active_node.y), 26, 2)

        # Draw High-Visibility Cyber Dashboard Controls
        font = pygame.font.SysFont("Courier New", 26, bold=True)
        big_font = pygame.font.SysFont("Courier New", 42, bold=True)
        
        # Upper telemetry header background
        pygame.draw.rect(surface, (6, 14, 22), (0, 0, BASE_W, 80))
        pygame.draw.line(surface, (0, 180, 180), (0, 80), (BASE_W, 80), 2)
        
        txt_score = font.render(f"SYSTEM_CREDITS: {self.score}", True, (255, 255, 255))
        txt_level = font.render(f"NODE_DEPTH: LVL_{self.level:02d}", True, (0, 255, 255))
        
        timer_color = (0, 255, 100) if self.time_remaining > 15 else (255, 40, 40)
        txt_timer = big_font.render(f"BYPASS_SEC: {self.time_remaining:.2f}s", True, timer_color)
        
        surface.blit(txt_score, (30, 25))
        surface.blit(txt_level, (380, 25))
        surface.blit(txt_timer, (BASE_W - 420, 18))

        # Real-time Hand-Tracking Ring UI
        cursor_color = (255, 50, 50) if self.pinch_active else (0, 255, 255)
        cursor_radius = 12 if self.pinch_active else 18
        pygame.draw.circle(surface, cursor_color, (self.cursor_smooth_x, self.cursor_smooth_y), cursor_radius, 3)
        pygame.draw.circle(surface, (255, 255, 255), (self.cursor_smooth_x, self.cursor_smooth_y), 3, 0)

        # Operational Instruction panel
        instr_font = pygame.font.SysFont("Courier New", 14, bold=False)
        txt_instr = instr_font.render("SYSTEM PROTOCOLS: [HOVER INDEX] TARGET NODE | [PINCH FINGER/HOLD] BYPASS CORE DATA | [DRAG FINGER] CONNECT SYSTEM PATH", True, (0, 150, 150))
        surface.blit(txt_instr, (30, BASE_H - 30))

        # Display terminal screens for failure parameters
        if self.game_over:
            overlay = pygame.Surface((BASE_W, BASE_H), pygame.SRCALPHA)
            overlay.fill((20, 4, 4, 220))
            surface.blit(overlay, (0, 0))
            txt_fail = big_font.render("!!! CONNECTION TERMINATED !!!", True, (255, 0, 50))
            txt_reset = font.render("PRESS [SPACEBAR] TO PURGE MEMORY & REBOOT", True, (200, 200, 200))
            surface.blit(txt_fail, (BASE_W // 2 - 360, BASE_H // 2 - 40))
            surface.blit(txt_reset, (BASE_W // 2 - 320, BASE_H // 2 + 30))

# --- MASTER CONTROL EXECUTION LOOP ---
def main():
    global screen, BASE_W, BASE_H
    
    # Initialize multi-threaded camera acquisition modules
    stream = VideoStream(src=0)
    game = FingerHackerGridGame()
    
    last_time = time.monotonic()
    
    # Pre-configure dynamic workspace frame layer buffer
    display_surface = pygame.Surface((BASE_W, BASE_H))

    while True:
        current_time = time.monotonic()
        dt = current_time - last_time
        last_time = current_time

        # Handle Native Pygame Resizable Screen & Interface Interaction Events
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                stream.release()
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.VIDEORESIZE:
                BASE_W, BASE_H = event.w, event.h
                screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
                # Re-anchor adjustable camera display positions inside new window size boundaries
                game.camera_rect.x = BASE_W - game.camera_rect.width - 20
                display_surface = pygame.Surface((BASE_W, BASE_H))
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game.game_over:
                    game.__init__()
                    
            # Handle adjustable camera bounding layout window mouse drag controls
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and game.camera_rect.collidepoint(event.pos):
                    game.camera_dragging = True
                    mouse_x, mouse_y = event.pos
                    game.offset_x = game.camera_rect.x - mouse_x
                    game.offset_y = game.camera_rect.y - mouse_y
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    game.camera_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if game.camera_dragging:
                    mouse_x, mouse_y = event.pos
                    game.camera_rect.x = max(0, min(BASE_W - game.camera_rect.width, mouse_x + game.offset_x))
                    game.camera_rect.y = max(0, min(BASE_H - game.camera_rect.height, mouse_y + game.offset_y))

        # Read frames from the asynchronous daemon worker thread queue
        ret, frame = stream.read()
        if ret and frame is not None:
            game.process_frame_landmarks(frame)

        # Clear display buffers with cyber space darkness
        display_surface.fill((3, 8, 12))
        
        # Advance state tracking timelines
        game.update_game_logic(dt)
        game.draw_hud(display_surface)

        # Handle adjustable camera feed container graphics integration
        if ret and frame is not None:
            # Resize matrix to perfectly match localized config bounds
            frame_resized = cv2.resize(frame, (game.camera_rect.width, game.camera_rect.height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            # Transpose arrays into specific coordinate matrices required by Pygame
            frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
            
            # Fix: .inflate() method is correct and creates a copy offset rectangle layout
            pygame.draw.rect(display_surface, (0, 255, 255), game.camera_rect.inflate(4, 4), 2)
            display_surface.blit(frame_surface, game.camera_rect.topleft)

        # Apply high-retention viral arcade screen shake matrices
        shake_offset_x = random.randint(-game.screen_shake, game.screen_shake) if game.screen_shake > 0 else 0
        shake_offset_y = random.randint(-game.screen_shake, game.screen_shake) if game.screen_shake > 0 else 0
        
        # Push completed buffer modifications directly to physical screen monitors
        if game.glitch_timer > 0 and random.random() < 0.4:
            # Glitch rendering: chromatic shift simulations
            screen.blit(display_surface, (shake_offset_x + random.randint(-5, 5), shake_offset_y), special_flags=pygame.BLEND_RGB_ADD)
        else:
            screen.blit(display_surface, (shake_offset_x, shake_offset_y))
            
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()