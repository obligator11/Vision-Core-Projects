from utils.math_helpers import OneEuroFilter
import time

class ConfidenceEngine:
    """Assembles independent analysis payloads into a balanced, filtered performance index."""
    def __init__(self):
        self.w_gaze = 0.35
        self.w_align = 0.20
        self.w_stable = 0.20
        self.w_speech = 0.25
        
        # Calibrated with higher min_cutoff for smooth tracking reactions
        self.filter = OneEuroFilter(t0=time.time(), x0=100.0, min_cutoff=1.8, beta=0.012)

    def process(self, face_data, posture_data, audio_data):
        if not face_data["face_detected"]:
            return {
                "confidence_score": 50,
                "eye_contact_score": 0,
                "posture_stability_score": 0,
                "speech_fluency_score": 0,
                "rating": "SUBJECT_NOT_FOUND"
            }

        gaze_val = face_data["eye_contact_ratio"] * 100.0
        align_val = posture_data["alignment_score"] if posture_data["pose_detected"] else 100.0
        stable_val = posture_data["stability_score"] if posture_data["pose_detected"] else 100.0

        if not audio_data.get("speech_detected", False):
            speech_fluency_score = 100.0
        else:
            fillers = audio_data.get("filler_count", 0)
            speech_fluency_score = max(20.0, 100.0 - (fillers * 5.0))

        # Sum weighted sub-component profiles
        computed_raw = (
            (gaze_val * self.w_gaze) + 
            (align_val * self.w_align) + 
            (stable_val * self.w_stable) +
            (speech_fluency_score * self.w_speech)
        )
        
        # Smooth processing signals to completely resolve max boundary ceiling bugs
        smoothed_score = int(self.filter.filter_signal(time.time(), computed_raw)[0])
        smoothed_score = max(0, min(100, smoothed_score))

        if smoothed_score >= 76:
            rating = "EXCELLENT_CONFIDENCE"
        elif smoothed_score >= 50:
            rating = "AVERAGE_CONFIDENCE"
        else:
            rating = "NERVOUS_BEHAVIOR"

        return {
            "confidence_score": smoothed_score,
            "eye_contact_score": int(gaze_val),
            "posture_stability_score": int((align_val + stable_val) / 2.0),
            "speech_fluency_score": int(speech_fluency_score),
            "rating": rating
        }