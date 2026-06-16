"""
SafetyAI - Armored Assessment Logic Engine
Maintains cross-frame memory to catch falls even if MediaPipe tracking breaks on impact.
"""
import time

class SafetyEngine:
    def __init__(self):
        # Operational Risk Config Matrix thresholds
        self.THRESHOLD_WARN_INACTIVE = 8.0   # Seconds before warning escalation
        self.THRESHOLD_DANGER_INACTIVE = 15.0 # Seconds before full alarm triggers
        
        # Internal Structural State Machine Flags
        self.absence_start_time = None
        self.inactivity_start_time = None
        self.last_fall_triggered_time = None
        self.is_fallen = False
        
        # Cross-Frame Velocity Memory (Crucial for broken tracking safety nets)
        self.last_known_velocity_y = 0.0
        self.last_known_aspect_ratio = 1.0
        
        # Core Output Telemetry Metrics
        self.safety_score = 100
        self.risk_level = "SAFE"
        self.current_anomaly = "Operational Nominal"
        
    def evaluate_states(self, keypoints: dict, motion_metrics: dict, frame_w: int, frame_h: int):
        """
        Executes a deterministic state check against worker telemetry.
        Armored against vanishing skeleton tracking on floor collapses.
        """
        current_time = time.time()
        
        # Extract immediate frame velocities from tracker memory if available
        velocity_y = motion_metrics.get('centroid_velocity_y', 0.0)
        
        # --- PROTECTION NET FOR BROKEN/MISSING SKELETONS ---
        if not keypoints:
            # If they were falling or flat right before tracking snapped, hold the fall state!
            if (self.last_known_velocity_y > (frame_h * 0.5) or self.last_known_aspect_ratio > 1.1) or self.is_fallen:
                self.is_fallen = True
                self.safety_score = 0
                self.risk_level = "DANGER"
                self.current_anomaly = "CRITICAL: Fall Impact Detected + Lost Body Tracking!"
                return self.safety_score, self.risk_level, self.current_anomaly
                
            # Otherwise, handle standard target absence outside the frame
            if self.absence_start_time is None:
                self.absence_start_time = current_time
            
            absence_duration = current_time - self.absence_start_time
            if absence_duration > 10.0:
                self.safety_score = 0
                self.risk_level = "DANGER"
                self.current_anomaly = "CRITICAL: Worker Missing Outside Window"
            elif absence_duration > 3.0:
                self.safety_score = 35
                self.risk_level = "WARNING"
                self.current_anomaly = "ALERT: Target Clearance Lost"
            else:
                self.risk_level = "SAFE"
                self.current_anomaly = "Worker Brief Disappearance"
                
            self.inactivity_start_time = None
            return self.safety_score, self.risk_level, self.current_anomaly

        # Target found: Reset absence timer
        self.absence_start_time = None
        
        try:
            nose = keypoints['nose']
            l_hip = keypoints['left_hip']
            r_hip = keypoints['right_hip']
            l_ank = keypoints['left_ankle']
            r_ank = keypoints['right_ankle']
        except KeyError:
            # Keep fallback memories updated even during partial occlusions
            return self.safety_score, self.risk_level, "Signal Obscured"

        # --- MATH LAYER 1: FALL ARREST GEOMETRY EXTRACTION ---
        all_x = [p[0] for p in keypoints.values()]
        all_y = [p[1] for p in keypoints.values()]
        bbox_w = max(all_x) - min(all_x)
        bbox_h = max(all_y) - min(all_y)
        aspect_ratio = float(bbox_w) / float(bbox_h) if bbox_h > 0 else 1.0
        
        # Save values into memory so the system remembers them if the body tracker drops frames next loop
        self.last_known_velocity_y = velocity_y
        self.last_known_aspect_ratio = aspect_ratio
        
        mid_ankle_y = (l_ank[1] + r_ank[1]) / 2.0
        vertical_span = abs(mid_ankle_y - nose[1])
        
        # Fall Assertion Rules: High downward velocity OR a collapsed horizontal aspect ratio profile
        if velocity_y > (frame_h * 0.35) or aspect_ratio > 1.15:
            self.is_fallen = True
            self.last_fall_triggered_time = current_time
            
        # Re-Stabilization Metric: Worker successfully stood back upright
        if aspect_ratio < 0.85 and vertical_span > (frame_h * 0.35):
            self.is_fallen = False

        # --- MATH LAYER 2: INACTIVITY TIMER ESCALATION ---
        is_moving = motion_metrics.get('is_moving', True)
        
        if not is_moving:
            if self.inactivity_start_time is None:
                self.inactivity_start_time = current_time
            inactivity_duration = current_time - self.inactivity_start_time
        else:
            self.inactivity_start_time = None
            inactivity_duration = 0.0

        # --- ARBITRATION SEVERITY EVALUATION MATRIX ---
        if self.is_fallen:
            self.safety_score = 5
            self.risk_level = "DANGER"
            self.current_anomaly = "CRITICAL METRIC: Horizontal Fall Collapse Confirmed"
            
        elif inactivity_duration > self.THRESHOLD_DANGER_INACTIVE:
            self.safety_score = 20
            self.risk_level = "DANGER"
            self.current_anomaly = f"CRITICAL: Static Immobility Lockout (> {int(inactivity_duration)}s)"
            
        elif inactivity_duration > self.THRESHOLD_WARN_INACTIVE:
            self.safety_score = 55
            self.risk_level = "WARNING"
            self.current_anomaly = f"WARNING: Prolonged Inactivity Active ({int(inactivity_duration)}s)"
            
        else:
            self.safety_score = 100
            self.risk_level = "SAFE"
            self.current_anomaly = "Operational Nominal"
            
        return self.safety_score, self.risk_level, self.current_anomaly

    def reset(self):
        """
        Clears all historical tracking parameters when video loops or restarts.
        """
        self.absence_start_time = None
        self.inactivity_start_time = None
        self.last_fall_triggered_time = None
        self.is_fallen = False
        self.last_known_velocity_y = 0.0
        self.last_known_aspect_ratio = 1.0
        self.safety_score = 100
        self.risk_level = "SAFE"
        self.current_anomaly = "Operational Nominal"