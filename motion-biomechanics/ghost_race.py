import cv2
import numpy as np
import mediapipe as mp
import time
import math
import pygame

# Initialize Pygame Mixer with strict low-latency parameters
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
pygame.mixer.init()

# =====================================================================
# PROCEDURAL HARDWARE AUDIO GENERATOR
# =====================================================================
def generate_synth_beep(frequency, duration_ms, wave_type="sine", volume=0.3):
    sample_rate = 44100
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, num_samples, False)
    
    if wave_type == "sine":
        samples = np.sin(2 * np.pi * frequency * t)
    elif wave_type == "square":
        samples = np.sign(np.sin(2 * np.pi * frequency * t))
    else:
        samples = 2.0 * (t * frequency - np.floor(0.5 + t * frequency))
        
    envelope = np.exp(-4 * np.linspace(0, 1, num_samples))
    samples = samples * envelope
    
    audio_buffer = np.int16(samples * 32767 * volume)
    stereo_buffer = np.column_stack((audio_buffer, audio_buffer))
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_buffer))

SOUND_START = generate_synth_beep(880, 250, "sine", 0.4)       
SOUND_GHOST = generate_synth_beep(440, 400, "square", 0.25)     
SOUND_SCORE = generate_synth_beep(1200, 80, "sine", 0.3)        
SOUND_WIN   = generate_synth_beep(660, 600, "sine", 0.35)   
SOUND_LOSE  = generate_synth_beep(150, 700, "square", 0.5)      

# =====================================================================
# CORE ENGINE CONSTANTS & STRUCTURES
# =====================================================================
class GameState:
    STANDBY = 0
    ROUND1_RECORDING = 1
    INTERMISSION = 2
    ROUND2_PLAYBACK = 3
    GAMEOVER = 4

class TargetOrb:
    def __init__(self, x, y, spawn_time):
        self.x = x
        self.y = y
        self.spawn_time = spawn_time
        self.radius = 35
        self.cleared_by_player = False
        self.cleared_by_ghost = False

class TimeWarpEngine:
    def __init__(self):
        self.state = GameState.STANDBY
        self.width, self.height = 800, 600
        
        # MediaPipe Tracking Solutions Unified
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Force single hand tracking to maximize frame processing throughput
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        self.recorded_timeline = []  
        self.game_targets = []
        self.score_player = 0
        self.score_ghost = 0
        
        self.fist_timer_start = None
        self.state_transition_time = 0.0
        self.round_duration = 15.0  
        
        self.trail_player = []
        self.trail_ghost = []
        
        # Simplified, ultra-stable structural connections
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17), # Palm
            (5, 8), (9, 12), (13, 16), (17, 20) # Main fingertip vectors
        ]

    def generate_procedural_targets(self):
        self.game_targets = []
        for i in range(1, int(self.round_duration) + 1):
            t_spawn = float(i) - 0.5
            x = int(400 + 260 * math.sin(i * 1.5))
            y = int(150 + (i * 40) % 300)
            self.game_targets.append(TargetOrb(x, y, t_spawn))

    def get_interpolated_ghost_position(self, current_timeline_pos):
        if not self.recorded_timeline:
            return None
            
        if current_timeline_pos <= self.recorded_timeline[0][0]:
            return self.recorded_timeline[0][1], self.recorded_timeline[0][2]
        if current_timeline_pos >= self.recorded_timeline[-1][0]:
            return self.recorded_timeline[-1][1], self.recorded_timeline[-1][2]
            
        low, high = 0, len(self.recorded_timeline) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if self.recorded_timeline[mid][0] < current_timeline_pos:
                low = mid
            else:
                high = mid
                
        p1, p2 = self.recorded_timeline[low], self.recorded_timeline[high]
        time_delta = p2[0] - p1[0]
        alpha = (current_timeline_pos - p1[0]) / time_delta if time_delta > 0 else 0.0
        
        x_g = int((1.0 - alpha) * p1[1] + alpha * p2[1])
        y_g = int((1.0 - alpha) * p1[2] + alpha * p2[2])
        return x_g, y_g

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        cv2.namedWindow("PROJECT CHRONOS: PAST VS PRESENT", cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            canvas = cv2.resize(frame, (self.width, self.height))
            
            # Sci-Fi Interface Mask Overlay
            canvas = cv2.addWeighted(canvas, 0.4, np.zeros(canvas.shape, canvas.dtype), 0.6, 0)
            
            # Process tracking parameters synchronously to avoid coordinate flight errors
            rgb_matrix = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_matrix)
            
            active_pos = None
            is_fist = False
            hand_pts = []
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract absolute scaling points mapping coordinates
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * self.width), int(lm.y * self.height)
                    hand_pts.append((cx, cy))
                
                # Tracking cursor bound directly to the base of the index finger joint
                active_pos = hand_pts[8]
                
                # Bulletproof Fist Gesture Math (Fingertips are closer to wrist than knuckles)
                # Distance from fingertips (8, 12, 16, 20) to wrist base (0)
                d_wrist = math.hypot(hand_pts[8][0] - hand_pts[0][0], hand_pts[8][1] - hand_pts[0][1])
                d_knuckle = math.hypot(hand_pts[5][0] - hand_pts[0][0], hand_pts[5][1] - hand_pts[0][1])
                if d_wrist < d_knuckle:
                    is_fist = True

            current_clock = time.time()
            timeline_t = current_clock - self.state_transition_time
            
            # =====================================================================
            # GAME ENGINE MACHINE FLOWS
            # =====================================================================
            if self.state == GameState.STANDBY:
                cv2.putText(canvas, "SYSTEM READY: MOVEMENT HUB", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "MAKE A TIGHT FIST FOR 1.5 SECONDS TO TRIGGER START", (50, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                if is_fist and active_pos is not None:
                    # Draw visual loading dial over tracking point
                    cv2.circle(canvas, active_pos, 45, (0, 255, 255), 2, cv2.LINE_AA)
                    if self.fist_timer_start is None:
                        self.fist_timer_start = current_clock
                    elif current_clock - self.fist_timer_start >= 1.5:
                        SOUND_START.play()
                        self.generate_procedural_targets()
                        self.recorded_timeline.clear()
                        self.trail_player.clear()
                        self.trail_ghost.clear()
                        self.score_player = 0
                        self.score_ghost = 0
                        self.state = GameState.ROUND1_RECORDING
                        self.state_transition_time = current_clock
                        self.fist_timer_start = None
                else:
                    self.fist_timer_start = None
                    
            elif self.state == GameState.ROUND1_RECORDING:
                rem_time = max(0, int(self.round_duration - timeline_t))
                cv2.putText(canvas, f"ROUND 1: RECORDING RUN - TIME LEFT: {rem_time}s", 
                            (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                if active_pos:
                    self.recorded_timeline.append((timeline_t, active_pos[0], active_pos[1]))
                    self.trail_player.append(active_pos)
                    if len(self.trail_player) > 15:
                        self.trail_player.pop(0)
                    
                for target in self.game_targets:
                    if timeline_t >= target.spawn_time and not target.cleared_by_player:
                        cv2.circle(canvas, (target.x, target.y), target.radius, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(canvas, (target.x, target.y), 6, (0, 165, 255), -1)
                        if active_pos:
                            if math.hypot(active_pos[0] - target.x, active_pos[1] - target.y) < target.radius:
                                target.cleared_by_player = True
                                self.score_player += 1
                                SOUND_SCORE.play()
                                
                cv2.putText(canvas, f"SCORE: {self.score_player}", (self.width - 200, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                
                if timeline_t >= self.round_duration:
                    SOUND_GHOST.play()
                    self.state = GameState.INTERMISSION
                    self.state_transition_time = current_clock
                    
            elif self.state == GameState.INTERMISSION:
                countdown = 3 - int(timeline_t)
                cv2.putText(canvas, f"GENERATING GHOST DATA FLUX... STARTING ROUND 2 IN: {countdown}", (50, 280), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                if timeline_t >= 3.0:
                    for target in self.game_targets:
                        target.cleared_by_player = False
                    self.state = GameState.ROUND2_PLAYBACK
                    self.state_transition_time = current_clock
                    
            elif self.state == GameState.ROUND2_PLAYBACK:
                cv2.putText(canvas, "YOU VS PAST YOU", (40, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3, cv2.LINE_AA)
                
                if active_pos:
                    self.trail_player.append(active_pos)
                    if len(self.trail_player) > 15:
                        self.trail_player.pop(0)
                        
                ghost_pos = self.get_interpolated_ghost_position(timeline_t)
                if ghost_pos:
                    self.trail_ghost.append(ghost_pos)
                    if len(self.trail_ghost) > 15:
                        self.trail_ghost.pop(0)
                        
                    # Smooth Ghost Rendering Overlay Alpha Mix
                    overlay = canvas.copy()
                    cv2.circle(overlay, ghost_pos, 25, (255, 0, 140), -1, cv2.LINE_AA)
                    cv2.putText(overlay, "GHOST", (ghost_pos[0] - 25, ghost_pos[1] - 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

                for target in self.game_targets:
                    if timeline_t >= target.spawn_time and not (target.cleared_by_player and target.cleared_by_ghost):
                        cv2.circle(canvas, (target.x, target.y), target.radius, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        if active_pos and not target.cleared_by_player:
                            if math.hypot(active_pos[0] - target.x, active_pos[1] - target.y) < target.radius:
                                target.cleared_by_player = True
                                self.score_player += 1
                                SOUND_SCORE.play()
                                
                        if ghost_pos and not target.cleared_by_ghost:
                            if math.hypot(ghost_pos[0] - target.x, ghost_pos[1] - target.y) < target.radius:
                                target.cleared_by_ghost = True
                                self.score_ghost += 1

                cv2.putText(canvas, f"PRESENT: {self.score_player}", (self.width - 240, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"GHOST:   {self.score_ghost}", (self.width - 240, 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 140), 2, cv2.LINE_AA)
                
                if timeline_t >= self.round_duration:
                    if self.score_player >= self.score_ghost:
                        SOUND_WIN.play()
                    else:
                        SOUND_LOSE.play()
                    self.state = GameState.GAMEOVER
                    self.state_transition_time = current_clock
                    
            elif self.state == GameState.GAMEOVER:
                msg = "VICTORY: TIMELINE CONQUERED!" if self.score_player >= self.score_ghost else "DEFEAT: GHOST WON THE RUN"
                cv2.putText(canvas, msg, (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if self.score_player >= self.score_ghost else (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, "PRESS [SPACE] TO RESET THE TIMELINE CHRONOS", (140, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 1, cv2.LINE_AA)

            # =====================================================================
            # GRAPHICS RENDER LAYER (Trails & Stable Bone Vectors)
            # =====================================================================
            # Present Run Trail Line
            for idx in range(1, len(self.trail_player)):
                cv2.line(canvas, self.trail_player[idx-1], self.trail_player[idx], (0, 255, 0), 3, cv2.LINE_AA)
            # Ghost Run Trail Line
            for idx in range(1, len(self.trail_ghost)):
                cv2.line(canvas, self.trail_ghost[idx-1], self.trail_ghost[idx], (255, 0, 140), 2, cv2.LINE_AA)

            # Rigid High-Contrast Hand Mesh Topology
            if len(hand_pts) > 0:
                for connection in self.bone_connections:
                    p1 = hand_pts[connection[0]]
                    p2 = hand_pts[connection[1]]
                    cv2.line(canvas, p1, p2, (0, 140, 255), 2, cv2.LINE_AA)
                for pt in hand_pts:
                    cv2.circle(canvas, pt, 4, (255, 255, 255), -1)

            if active_pos and self.state != GameState.GAMEOVER:
                cv2.circle(canvas, active_pos, 8, (0, 255, 120), -1, cv2.LINE_AA)

            cv2.imshow("PROJECT CHRONOS: PAST VS PRESENT", canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32 and self.state == GameState.GAMEOVER:
                self.state = GameState.STANDBY
                
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    engine = TimeWarpEngine()
    engine.run()