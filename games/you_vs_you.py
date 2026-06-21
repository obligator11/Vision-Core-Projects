import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time
import threading
from collections import deque

# =====================================================================
# SYSTEM CONFIGURATION & HARDWARE INITIALIZATION
# =====================================================================
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# High-DPI Resizable Canvas Constants
BASE_W, BASE_H = 1280, 720
screen = pygame.display.set_mode((BASE_W, BASE_H), pygame.RESIZABLE)
pygame.display.set_caption("⚡ You vs Perfect You: Thread-Isolated Kinetic Arena ⚡")
clock = pygame.time.Clock()

# Zero-Latency Multi-Threaded Memory Register Exchange State
class ThreadedSharedState:
    def __init__(self):
        self.raw_frame = None
        self.processed_points = None
        self.system_running = True
        self.lock = threading.Lock()

shared_state = ThreadedSharedState()

# =====================================================================
# REAL-TIME PROCEDURAL AUDIO SYNTHESIZER (LAG-FREE INTERRUPTS)
# =====================================================================
def generate_procedural_sound(frequency=440, duration=0.1, wave_type='sine', volume=0.25):
    """Synthesizes raw mathematical frequencies into RAM arrays instantly."""
    sample_rate = 44100
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, False)
    
    if wave_type == 'sine':
        wave_data = np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'sawtooth':
        wave_data = 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif wave_type == 'square':
        wave_data = np.sign(np.sin(2 * np.pi * frequency * t))
    else:
        wave_data = np.sin(2 * np.pi * frequency * t)
        
    # Smoothed exponential decay boundary mask to prevent audio pops
    envelope = np.exp(-4 * np.linspace(0, 1, total_samples))
    audio_buffer = (wave_data * envelope * volume * 32767).astype(np.int16)
    
    # Mirror into a contiguous memory structure for stereo hardware cards
    stereo_buffer = np.repeat(audio_buffer[:, np.newaxis], 2, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_buffer))

SOUND_SUCCESS = generate_procedural_sound(frequency=587.33, duration=0.12, wave_type='sine', volume=0.2)  # D5 Chime
SOUND_ERROR = generate_procedural_sound(frequency=146.83, duration=0.15, wave_type='sawtooth', volume=0.15) # Low D Buzz
SOUND_COMBO = generate_procedural_sound(frequency=880.00, duration=0.08, wave_type='sine', volume=0.25) # High A Chime

# =====================================================================
# ASYNCHRONOUS PIPELINE WORKERS (FRAME INGESTION & INFERENCE)
# =====================================================================
class AsyncVideoCapturePool:
    def __init__(self, source_idx=0):
        self.cap = cv2.VideoCapture(source_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.worker_thread = threading.Thread(target=self._capture_lifecycle, daemon=True)

    def start(self):
        self.worker_thread.start()

    def _capture_lifecycle(self):
        while True:
            with shared_state.lock:
                if not shared_state.system_running:
                    break
            ret, frame = self.cap.read()
            if ret:
                with shared_state.lock:
                    shared_state.raw_frame = frame
            else:
                time.sleep(0.005)

    def release_hardware(self):
        self.cap.release()

class AsyncKinematicInferenceEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # Fixed spatial tracking reliability architecture
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.worker_thread = threading.Thread(target=self._inference_lifecycle, daemon=True)

    def start(self):
        self.worker_thread.start()

    def _inference_lifecycle(self):
        while True:
            with shared_state.lock:
                if not shared_state.system_running:
                    break
                frame_to_process = None
                if shared_state.raw_frame is not None:
                    frame_to_process = shared_state.raw_frame.copy()

            if frame_to_process is not None:
                # Mirroring optimization handled prior to coordinate transformation matrix conversion
                frame_to_process = cv2.flip(frame_to_process, 1)
                rgb_matrix = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                results = self.pose_model.process(rgb_matrix)
                
                extracted_landmarks = None
                if results.pose_landmarks:
                    extracted_landmarks = {}
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        extracted_landmarks[idx] = (lm.x, lm.y, lm.z, lm.visibility)

                with shared_state.lock:
                    shared_state.processed_points = extracted_landmarks
            
            time.sleep(0.01) # Isolated model calculation clock constraint

# =====================================================================
# GEOMETRY ENGINE & VISUAL STYLING HANDLERS
# =====================================================================
class GameState:
    IDLE = 0
    RECORDING = 1
    COMPUTING = 2
    CHALLENGE_ACTIVE = 3

class KinematicTopology:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Structural bones tracked and verified
        self.structural_joints = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]

    def compute_torso_centroid_normalization(self, points):
        """Translates point arrays relative to the torso centroid to neutralize spatial offsets."""
        if not points:
            return None
        try:
            # Anchor indices for calculating centroid: shoulders (11, 12) and hips (23, 24)
            cx = (points[11][0] + points[12][0] + points[23][0] + points[24][0]) / 4.0
            cy = (points[11][1] + points[12][1] + points[23][1] + points[24][1]) / 4.0
            cz = (points[11][2] + points[12][2] + points[23][2] + points[24][2]) / 4.0
            
            normalized_map = {}
            for idx, pt in points.items():
                normalized_map[idx] = (pt[0] - cx, pt[1] - cy, pt[2] - cz, pt[3])
            return normalized_map
        except KeyError:
            return points

    def evaluate_pose_similarity(self, current, target, spatial_tolerance=0.14):
        """Calculates distance errors per segment and derives an aggregate accuracy score."""
        if not current or not target:
            return 0.0, []

        mismatch_nodes = []
        accumulated_variance = 0.0
        active_segments = 0

        for bone_start, bone_end in self.structural_joints:
            idx_s, idx_e = bone_start.value, bone_end.value
            if idx_s in current and idx_e in current and idx_s in target and idx_e in target:
                # Create localized direction vector representations
                vec_curr = np.array([current[idx_e][0] - current[idx_s][0], current[idx_e][1] - current[idx_s][1]])
                vec_targ = np.array([target[idx_e][0] - target[idx_s][0], target[idx_e][1] - target[idx_s][1]])
                
                segment_delta = np.linalg.norm(vec_curr - vec_targ)
                accumulated_variance += segment_delta
                active_segments += 1
                
                if segment_delta > spatial_tolerance:
                    mismatch_nodes.append(idx_e)

        if active_segments == 0:
            return 0.0, []

        mean_variance = accumulated_variance / active_segments
        # Map mean error exponentially to a responsive 0-100 UI score
        clamped_accuracy = max(0.0, min(100.0, 100.0 * math.exp(-2.8 * mean_variance)))
        return clamped_accuracy, list(set(mismatch_nodes))

class FeedbackParticle:
    def __init__(self, x, y, base_color):
        self.x = x
        self.y = y
        self.vx = np.random.uniform(-4, 4)
        self.vy = np.random.uniform(-6, -2)
        self.radius = np.random.randint(3, 7)
        self.alpha = 255
        self.color = base_color

    def advance(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.22 # Gravity acceleration constant
        self.alpha -= 14
        return self.alpha > 0

    def render(self, surface):
        if self.alpha > 0:
            p_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(p_surface, (*self.color, self.alpha), (self.radius, self.radius), self.radius)
            surface.blit(p_surface, (int(self.x - self.radius), int(self.y - self.radius)))

class NeonHUDDashboard:
    def __init__(self):
        self.title_font = pygame.font.SysFont("comicsansms", 38, bold=True)
        self.label_font = pygame.font.SysFont("arial", 20, bold=True)
        self.stats_font = pygame.font.SysFont("consolas", 16, bold=True)
        self.particle_pool = []
        self.topology = KinematicTopology()

    def generate_burst_effect(self, x, y, color):
        for _ in range(6):
            self.particle_pool.append(FeedbackParticle(x, y, color))

    def flush_and_update_particles(self, surface):
        self.particle_pool = [p for p in self.particle_pool if p.advance()]
        for p in self.particle_pool:
            p.render(surface)

    def draw_wireframe(self, surface, points, cx, cy, scaler, primary_color, mismatch_filter=None, width=4):
        if not points:
            return
        mismatch_set = set(mismatch_filter) if mismatch_filter else set()

        # Bone Line Pipelines Rendering
        for joint_start, joint_end in self.topology.structural_joints:
            idx_s, idx_e = joint_start.value, joint_end.value
            if idx_s in points and idx_e in points:
                a_x = int(cx + points[idx_s][0] * scaler)
                a_y = int(cy + points[idx_s][1] * scaler)
                b_x = int(cx + points[idx_e][0] * scaler)
                b_y = int(cy + points[idx_e][1] * scaler)
                
                # Turn wireframe red if a joint falls outside acceptable spatial boundary vectors
                active_color = (244, 63, 94) if (idx_e in mismatch_set or idx_s in mismatch_set) else primary_color
                pygame.draw.line(surface, active_color, (a_x, a_y), (b_x, b_y), width)

        # Joint Nodes Dot Rendering
        for idx, pt in points.items():
            if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                x_coord = int(cx + pt[0] * scaler)
                y_coord = int(cy + pt[1] * scaler)
                node_color = (255, 0, 0) if idx in mismatch_set else (255, 255, 255)
                pygame.draw.circle(surface, node_color, (x_coord, y_coord), int(width * 1.5))

    def draw_hud_panels(self, surface, current_score, streak, scaling_factor, app_state, time_left=0.0):
        canvas_w, canvas_h = surface.get_size()

        # Top Control Panel Card Frame
        control_card = pygame.Surface((canvas_w - 40, 75), pygame.SRCALPHA)
        control_card.fill((15, 23, 42, 210)) # Dark slate alpha mask
        pygame.draw.rect(control_card, (56, 189, 248), (0, 0, canvas_w - 40, 75), 2, border_radius=10)
        surface.blit(control_card, (20, 20))

        if app_state == GameState.IDLE:
            msg, text_color = "SYSTEM STANDBY -> PRESS [SPACEBAR] TO CAPTURE TARGET BASELINE", (241, 245, 249)
        elif app_state == GameState.RECORDING:
            msg, text_color = f"⚡ LIVE RECORDING SOURCE MOVEMENTS IN PROGRESS ({time_left:.1f}s) ⚡", (239, 68, 68)
        elif app_state == GameState.COMPUTING:
            msg, text_color = "AI ALGORITHMIC TEMPORAL SMOOTHING RUNNING...", (234, 179, 8)
        else:
            msg, text_color = f"🎯 CHALLENGE ACTIVE: MATCH THE GHOST RECORDING! ({time_left:.1f}s)", (34, 197, 94)

        surface.blit(self.label_font.render(msg, True, text_color), (45, 42))

        # Side Statistical Matrix Frame Panel
        stat_card = pygame.Surface((260, 190), pygame.SRCALPHA)
        stat_card.fill((15, 23, 42, 190))
        pygame.draw.rect(stat_card, (168, 85, 247), (0, 0, 260, 190), 2, border_radius=12)

        lbl_score = self.title_font.render(f"{int(current_score)}", True, (255, 255, 255))
        lbl_streak = self.label_font.render(f"COMBO STREAK: {streak}", True, (56, 189, 248))
        lbl_diff = self.label_font.render(f"DIFFICULTY: x{scaling_factor:.2f}", True, (234, 179, 8))

        stat_card.blit(lbl_score, (20, 15))
        stat_card.blit(lbl_streak, (20, 95))
        stat_card.blit(lbl_diff, (20, 135))
        surface.blit(stat_card, (20, 115))

# =====================================================================
# SYSTEM CORE PROCESS MANAGEMENT RUNNER
# =====================================================================
def run_application_engine():
    # Instantiate asynchronous worker pools
    camera_stream = AsyncVideoCapturePool(source_idx=0)
    camera_stream.start()

    inference_engine = AsyncKinematicInferenceEngine()
    inference_engine.start()

    topology_evaluator = KinematicTopology()
    hud = NeonHUDDashboard()

    # Core Performance Tracking Registries
    active_state = GameState.IDLE
    captured_sequence_history = []
    perfect_ai_sequence_trajectory = []
    
    player_accumulated_score = 0.0
    streak_register = 0
    difficulty_mod = 1.0
    
    state_timer_counter = 0.0
    trajectory_playback_idx = 0
    audio_chime_gate = 0.0
    last_frame_timestamp = time.time()

    is_engine_active = True
    while is_engine_active:
        current_time = time.time()
        delta_t = current_time - last_frame_timestamp
        last_frame_timestamp = current_time

        if audio_chime_gate > 0.0:
            audio_chime_gate -= delta_t

        # Pygame Window System Event Trapping Loops
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_engine_active = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    is_engine_active = False
                elif event.key == pygame.K_SPACE and active_state == GameState.IDLE:
                    # Initialize recording pipeline configuration variables
                    active_state = GameState.RECORDING
                    state_timer_counter = 5.0 # Collect pristine timeline frames for 5 full seconds
                    captured_sequence_history.clear()
                    perfect_ai_sequence_trajectory.clear()
                    trajectory_playback_idx = 0

        # Safely extract thread-isolated coordinate frame matrices
        active_raw_landmarks = None
        with shared_state.lock:
            if shared_state.processed_points is not None:
                active_raw_landmarks = shared_state.processed_points.copy()

        # Canvas Setup
        screen.fill((9, 13, 26)) # Midnight Blue Deep Matte Field
        canvas_w, canvas_h = screen.get_size()

        # Core Game State Machine Optimization Loops
        if active_state == GameState.RECORDING:
            state_timer_counter -= delta_t
            if active_raw_landmarks:
                captured_sequence_history.append(topology_evaluator.compute_torso_centroid_normalization(active_raw_landmarks))
            if state_timer_counter <= 0.0:
                active_state = GameState.COMPUTING
                state_timer_counter = 1.2 # UI transition freeze gap duration threshold

        elif active_state == GameState.COMPUTING:
            state_timer_counter -= delta_t
            if state_timer_counter <= 0.0:
                # GESTURE SYSTEM: Box-Filter Spatial-Smoothing Optimization (Creates the "Perfect" trajectory)
                raw_length = len(captured_sequence_history)
                if raw_length > 4:
                    for idx in range(raw_length):
                        # Construct a 5-frame temporal sliding window slice to eliminate human noise jitter
                        sample_window = []
                        for offset in range(max(0, idx - 2), min(raw_length, idx + 3)):
                            if captured_sequence_history[offset]:
                                sample_window.append(captured_sequence_history[offset])
                        
                        if not sample_window:
                            perfect_ai_sequence_trajectory.append(captured_sequence_history[idx])
                            continue
                        
                        optimized_frame = {}
                        first_valid_idx = sample_window[0].keys()
                        for joint_id in first_valid_idx:
                            xs = [frame[joint_id][0] for frame in sample_window if joint_id in frame]
                            ys = [frame[joint_id][1] for frame in sample_window if joint_id in frame]
                            zs = [frame[joint_id][2] for frame in sample_window if joint_id in frame]
                            vis = [frame[joint_id][3] for frame in sample_window if joint_id in frame]
                            
                            # Standardize trajectories and clean extensions uniformly
                            optimized_frame[joint_id] = (
                                float(np.mean(xs)), 
                                float(np.mean(ys)), 
                                float(np.mean(zs)), 
                                float(np.mean(vis))
                            )
                        perfect_ai_sequence_trajectory.append(optimized_frame)
                else:
                    perfect_ai_sequence_trajectory = list(captured_sequence_history)

                if perfect_ai_sequence_trajectory:
                    active_state = GameState.CHALLENGE_ACTIVE
                    state_timer_counter = len(perfect_ai_sequence_trajectory) * 0.033 # Calculated playback timing envelope
                    trajectory_playback_idx = 0
                else:
                    active_state = GameState.IDLE

        elif active_state == GameState.CHALLENGE_ACTIVE:
            state_timer_counter -= delta_t
            # Derive current comparative playback sequence frame index tracking position
            trajectory_playback_idx = int(((len(perfect_ai_sequence_trajectory) * 0.033 - state_timer_counter) / 0.033))
            if trajectory_playback_idx >= len(perfect_ai_sequence_trajectory) or state_timer_counter <= 0.0:
                active_state = GameState.IDLE
                # Scale up baseline difficulty attributes if player finishes high-accuracy sequences successfully
                if player_accumulated_score > 600:
                    difficulty_mod += 0.20

        # COMPARATIVE REAL-TIME ANALYSIS ENGINE
        live_normalized_skeleton = topology_evaluator.compute_torso_centroid_normalization(active_raw_landmarks) if active_raw_landmarks else None
        mismatch_tracking_list = []
        frame_accuracy_score = 0.0

        if active_state == GameState.CHALLENGE_ACTIVE and live_normalized_skeleton and trajectory_playback_idx < len(perfect_ai_sequence_trajectory):
            target_ai_skeleton = perfect_ai_sequence_trajectory[trajectory_playback_idx]
            # Dynamic margins narrow down acceptance fields as difficulty multipliers increase
            optimized_margin = max(0.07, 0.15 - (difficulty_mod * 0.012))
            
            frame_accuracy_score, mismatch_tracking_list = topology_evaluator.evaluate_pose_similarity(
                live_normalized_skeleton, target_ai_skeleton, spatial_tolerance=optimized_margin
            )

            # Performance Tally Adjustments & Feedback Triggers
            if frame_accuracy_score >= 72.0:
                player_accumulated_score += frame_accuracy_score * delta_t * difficulty_mod
                streak_register += 1
                if audio_chime_gate <= 0.0:
                    if streak_register % 20 == 0:
                        SOUND_COMBO.play()
                    else:
                        SOUND_SUCCESS.play()
                    audio_chime_gate = 0.10
                hud.generate_burst_effect(canvas_w - 160, 220, (34, 211, 238))
            elif frame_accuracy_score < 40.0:
                streak_register = 0
                if audio_chime_gate <= 0.0:
                    SOUND_ERROR.play()
                    audio_chime_gate = 0.30

        # =====================================================================
        # ASYNC GRAPHICS PRESENTATION LAYER
        # =====================================================================
        # Dynamic Camera Picture-In-Picture Overlay Matrix Surface Conversion
        local_bgr_frame = None
        with shared_state.lock:
            if shared_state.raw_frame is not None:
                local_bgr_frame = shared_state.raw_frame.copy()

        if local_bgr_frame is not None:
            local_bgr_frame = cv2.flip(local_bgr_frame, 1)
            rgb_surface = pygame.surfarray.make_surface(cv2.cvtColor(local_bgr_frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
            overlay_w, overlay_h = 280, 210
            scaled_overlay = pygame.transform.scale(rgb_surface, (overlay_w, overlay_h))
            pos_x, pos_y = canvas_w - overlay_w - 20, 115
            screen.blit(scaled_overlay, (pos_x, pos_y))
            pygame.draw.rect(screen, (168, 85, 247), (pos_x, pos_y, overlay_w, overlay_h), 2, border_radius=6)

        # Draw Viewport Splits for Comparative Wireframe Layouts
        split_vertical_y = canvas_h // 2 + 50
        render_scale = min(canvas_w, canvas_h) * 0.48

        if active_state == GameState.CHALLENGE_ACTIVE:
            ghost_center_x = canvas_w // 4 + 40
            player_center_x = (canvas_w // 4) * 3 - 40

            screen.blit(hud.stats_font.render("AI PERFECT CLONE GHOST (TARGET)", True, (168, 85, 247)), (ghost_center_x - 140, 115))
            screen.blit(hud.stats_font.render("YOUR LIVE INTERACTION FIELD", True, (56, 189, 248)), (player_center_x - 120, 115))

            # 1. Render Ghost AI Perfect Clone Trajectory (Purple Aura Shape Profile)
            if trajectory_playback_idx < len(perfect_ai_sequence_trajectory):
                hud.draw_wireframe(
                    screen, perfect_ai_sequence_trajectory[trajectory_playback_idx],
                    ghost_center_x, split_vertical_y, render_scale, (168, 85, 247), width=5
                )

            # 2. Render Live Player Wireframe (Highlights anomalous nodes in Red)
            if live_normalized_skeleton:
                hud.draw_wireframe(
                    screen, live_normalized_skeleton,
                    player_center_x, split_vertical_y, render_scale, (56, 189, 248),
                    mismatch_filter=mismatch_tracking_list, width=5
                )

            # Smooth Accuracy Slider Meter Hud Core Interface Block Group
            bar_w, bar_h = 360, 22
            bar_x, bar_y = canvas_w // 2 - bar_w // 2, canvas_h - 55
            pygame.draw.rect(screen, (30, 41, 59), (bar_x, bar_y, bar_w, bar_h), border_radius=8)
            
            fill_color = (244, 63, 94) if frame_accuracy_score < 50.0 else (34, 197, 94)
            fill_width = int(bar_w * (frame_accuracy_score / 100.0))
            if fill_width > 0:
                pygame.draw.rect(screen, fill_color, (bar_x, bar_y, fill_width, bar_h), border_radius=8)
            pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=8)

            txt_pct = hud.stats_font.render(f"MATCH COEFFICIENT: {frame_accuracy_score:.1f}%", True, (255, 255, 255))
            screen.blit(txt_pct, (bar_x + 5, bar_y - 24))

        else:
            # Standby/Recording Configuration Mode: Single Centered Skeletons Viewport Presenter
            unified_center_x = canvas_w // 2
            if live_normalized_skeleton:
                fallback_color = (239, 68, 68) if active_state == GameState.RECORDING else (241, 245, 249)
                hud.draw_wireframe(screen, live_normalized_skeleton, unified_center_x, split_vertical_y, render_scale, fallback_color, width=4)

        # Render Particle Systems and Dash Panels
        hud.flush_and_update_particles(screen)
        hud.draw_hud_panels(screen, player_accumulated_score, streak_register, difficulty_mod, active_state, state_timer_counter)

        pygame.display.flip()
        clock.tick(60) # Main rendering thread runs unconstrained at 60 FPS

    # Deterministic Hardware Deallocation Hooks
    with shared_state.lock:
        shared_state.system_running = False
    camera_stream.release_hardware()
    pygame.mixer.quit()
    pygame.quit()

if __name__ == "__main__":
    run_application_engine()