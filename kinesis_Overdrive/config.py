"""
Project Kinesis: NEXT — Central configuration.

Every tunable constant lives here so the rest of the codebase never
hardcodes magic numbers. Change difficulty curves, colors, timings,
or file paths in exactly one place.
"""

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LEADERBOARD_PATH = os.path.join(DATA_DIR, "leaderboard.json")

HIT_SOUND_PATH = os.path.join(ASSETS_DIR, "hit.wav")
START_SOUND_PATH = os.path.join(ASSETS_DIR, "start.wav")
LEVELUP_SOUND_PATH = os.path.join(ASSETS_DIR, "levelup.wav")
MISS_SOUND_PATH = os.path.join(ASSETS_DIR, "miss.wav")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Window / rendering
# ---------------------------------------------------------------------------
WINDOW_NAME = "Project Kinesis: NEXT"
TARGET_FPS = 60
DIM_BACKGROUND_ALPHA = 0.55  # how much of the raw camera feed shows through

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GOLD = (0, 215, 255)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_LEFT_TRAIL = (255, 200, 0)
COLOR_RIGHT_TRAIL = (0, 165, 255)
COLOR_GREEN = (80, 220, 100)
COLOR_ORANGE = (0, 140, 255)


# ---------------------------------------------------------------------------
# Gesture recognition
# ---------------------------------------------------------------------------
PALM_HOLD_SECONDS = 1.2       # how long an open palm must be held to confirm
FIST_HOLD_SECONDS = 1.0       # how long a fist must be held to pause/resume
GESTURE_MIN_DETECTION_CONF = 0.7
GESTURE_MIN_TRACKING_CONF = 0.6


# ---------------------------------------------------------------------------
# Round structure
# ---------------------------------------------------------------------------
ROUND_SECONDS = 60
COUNTDOWN_SECONDS = 3
COMBO_IDLE_RESET_SECONDS = 2.5   # miss/no-hit window before combo resets
LEADERBOARD_SIZE = 10


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Difficulty:
    name: str
    target_radius: int
    target_lifetime: float     # seconds before an unhit target expires (miss)
    spawn_interval: float      # seconds between new targets appearing
    max_active_targets: int
    drift: bool                 # whether targets slowly move
    drift_speed: float
    score_multiplier: float


DIFFICULTIES = {
    "EASY": Difficulty(
        name="EASY", target_radius=55, target_lifetime=3.2,
        spawn_interval=1.1, max_active_targets=1, drift=False,
        drift_speed=0.0, score_multiplier=0.8,
    ),
    "NORMAL": Difficulty(
        name="NORMAL", target_radius=42, target_lifetime=2.3,
        spawn_interval=0.85, max_active_targets=2, drift=False,
        drift_speed=0.0, score_multiplier=1.0,
    ),
    "INSANE": Difficulty(
        name="INSANE", target_radius=30, target_lifetime=1.5,
        spawn_interval=0.55, max_active_targets=3, drift=True,
        drift_speed=1.6, score_multiplier=1.6,
    ),
}
DEFAULT_DIFFICULTY = "NORMAL"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
BASE_SCORE_PER_MPH = 8.5
COMBO_MAX_MULTIPLIER = 4.0
COMBO_STEP = 0.25          # multiplier gained per consecutive hit
