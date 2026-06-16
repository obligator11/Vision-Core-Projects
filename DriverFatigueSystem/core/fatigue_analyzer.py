import time
import numpy as np
from config import SystemConfig
from utils.math_utils import MathUtils

class FatigueAnalyzer:
    """State engine performing runtime metric tracking analysis with baseline calibration."""
    
    def __init__(self) -> None:
        # Clocks and tracking states
        self.eye_closure_start_time: float | None = None
        self.yawn_start_time: float | None = None
        self.distraction_start_time: float | None = None
        
        self.fatigue_score: float = 0.0
        self.attention_score: float = 100.0
        self.risk_status: str = "SAFE"
        
        # Head Pose Calibration State Variables
        self.is_calibrated: bool = False
        self.calibration_frames_collected: int = 0
        self.pitch_baseline: float = 0.0
        self.yaw_baseline: float = 0.0
        self.roll_baseline: float = 0.0
        
        # Temporal tracking arrays
        self.pitch_buffer = []
        self.yaw_buffer = []
        self.roll_buffer = []

    def analyze(self, landmarks_contract, img_w: int, img_h: int) -> tuple[float, float, str]:
        """
        Processes facial landmarks tracking features against calibrated baseline parameters.
        """
        current_time = time.time()
        
        # Compute facial metrics
        ear_left = MathUtils.calculate_ear(landmarks_contract.left_eye)
        ear_right = MathUtils.calculate_ear(landmarks_contract.right_eye)
        current_ear = (ear_left + ear_right) / 2.0
        current_mar = MathUtils.calculate_mar(landmarks_contract.inner_lips)
        
        # Raw 3D head pose estimation
        pitch, yaw, roll = MathUtils.estimate_head_pose(landmarks_contract.pose_3d_points, img_w, img_h)
        
        # Handle 30-frame initial baseline calibration window
        if not self.is_calibrated:
            self.pitch_buffer.append(pitch)
            self.yaw_buffer.append(yaw)
            self.roll_buffer.append(roll)
            self.calibration_frames_collected += 1
            
            if self.calibration_frames_collected >= 30:
                self.pitch_baseline = float(np.mean(self.pitch_buffer))
                self.yaw_baseline = float(np.mean(self.yaw_buffer))
                self.roll_baseline = float(np.mean(self.roll_buffer))
                self.is_calibrated = True
                print(f"[INFO] Head pose baseline calibrated! Pitch Base: {self.pitch_baseline:.2f}, Yaw Base: {self.yaw_baseline:.2f}")
            
            # Return baseline safe metrics while calibration completes
            return 0.0, 100.0, "SAFE"

        # Apply calibration offsets to get normalized true relative values
        relative_pitch = pitch - self.pitch_baseline
        relative_yaw = yaw - self.yaw_baseline
        relative_roll = roll - self.roll_baseline

        # 1. Evaluate Eye Closures
        if current_ear < SystemConfig.EYE_EAR_THRESHOLD:
            if self.eye_closure_start_time is None:
                self.eye_closure_start_time = current_time
            closure_duration = current_time - self.eye_closure_start_time
        else:
            self.eye_closure_start_time = None
            closure_duration = 0.0

        # 2. Evaluate Yawning
        if current_mar > SystemConfig.MOUTH_MAR_THRESHOLD:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
            yawn_duration = current_time - self.yawn_start_time
        else:
            self.yawn_start_time = None
            yawn_duration = 0.0

        # 3. Evaluate Attention State (Using calibrated relative angles)
        is_distracted = abs(relative_yaw) > 20.0 or abs(relative_pitch) > 15.0 or abs(relative_roll) > 15.0
        
        if is_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            distraction_duration = current_time - self.distraction_start_time
        else:
            self.distraction_start_time = None
            distraction_duration = 0.0

        # 4. Calculate Fatigue Score
        target_fatigue = 0.0
        if closure_duration > 0:
            target_fatigue += min((closure_duration / SystemConfig.MAX_EYE_CLOSURE_DURATION) * 70.0, 70.0)
        if yawn_duration > 0:
            target_fatigue += min((yawn_duration / SystemConfig.MAX_YAWN_DURATION) * 30.0, 30.0)
            
        if closure_duration == 0.0 and yawn_duration == 0.0:
            self.fatigue_score = max(self.fatigue_score - 1.5, 0.0)
        else:
            self.fatigue_score = min(self.fatigue_score + 2.0, 100.0)
            
        self.fatigue_score = max(self.fatigue_score, target_fatigue)

        # 5. Calculate Attention Score (Fixed decay/recovery logic bounds)
        if distraction_duration > 0:
            # Gradually drain attention towards 0 instead of dropping instantly to 40
            penalty = (distraction_duration / SystemConfig.DISTRACTION_DURATION_LIMIT) * 100.0
            self.attention_score = max(100.0 - penalty, 0.0)
        else:
            # Smoothly recover attention score when looking back at the road
            self.attention_score = min(self.attention_score + 3.0, 100.0)

        # 6. Final Risk Categorization Matrix
        if self.fatigue_score >= 75.0 or self.attention_score <= 40.0 or closure_duration > SystemConfig.MAX_EYE_CLOSURE_DURATION:
            self.risk_status = "DANGER"
        elif self.fatigue_score >= 40.0 or self.attention_score <= 70.0:
            self.risk_status = "WARNING"
        else:
            self.risk_status = "SAFE"

        return float(self.fatigue_score), float(self.attention_score), self.risk_status