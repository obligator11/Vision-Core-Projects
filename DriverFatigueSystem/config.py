import os

class SystemConfig:
    """Centralized configuration management registry for structural and systemic thresholds."""
    # Video Stream Settings
    CAMERA_INDEX: str | int = "test_videos/driver_test_clip.mp4"
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_TARGET: int = 30

    # MediaPipe Landmark Index Mappings (Face Mesh IDs)
    LEFT_EYE_LANDMARKS: list[int] = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_LANDMARKS: list[int] = [33, 160, 158, 133, 153, 144]
    INNER_LIPS_LANDMARKS: list[int] = [78, 81, 13, 311, 308, 402, 14, 178]
    
    # 3D Head Pose Model Estimation Points (World coordinates approximation matches)
    NOSE_TIP_IDX: int = 1
    CHIN_IDX: int = 152
    LEFT_EYE_CORNER_IDX: int = 33
    RIGHT_EYE_CORNER_IDX: int = 263
    LEFT_MOUTH_CORNER_IDX: int = 61
    RIGHT_MOUTH_CORNER_IDX: int = 291

    # Algorithmic Detection Thresholds
    EYE_EAR_THRESHOLD: float = 0.21        # Below this value implies eye closure
    MOUTH_MAR_THRESHOLD: float = 0.50      # Above this value implies yawning
    GAZE_MAX_VARIANCE: float = 0.04         # Permissible horizontal pupil variance before distraction flag
    
    # Temporal Durations (Seconds)
    MAX_EYE_CLOSURE_DURATION: float = 1.5   # Continuous seconds before triggering DANGER status
    MAX_YAWN_DURATION: float = 3.0          # Continuous seconds before tracking fatigue accumulation
    DISTRACTION_DURATION_LIMIT: float = 2.0 # Max duration looking away before risk amplification

    # UI Visual Color Palette (BGR for OpenCV Compliance)
    COLOR_SAFE: tuple[int, int, int] = (0, 255, 0)      # Green
    COLOR_WARNING: tuple[int, int, int] = (0, 255, 255)  # Yellow
    COLOR_DANGER: tuple[int, int, int] = (0, 0, 255)     # Red
    COLOR_TEXT: tuple[int, int, int] = (255, 255, 255)   # White
    COLOR_MESH: tuple[int, int, int] = (220, 220, 220)   # Light Gray

    # Sound System Infrastructure
    AUDIO_DIRECTORY: str = "audio"
    ALERT_SOUND_FILE: str = os.path.join("audio", "alert.wav")
    WARNING_ALERT_INTERVAL: float = 1.5    # Pulsing delay loop for warnings