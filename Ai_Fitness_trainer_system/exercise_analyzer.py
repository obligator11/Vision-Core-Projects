

import time
from collections import deque
from utils import calculate_angle, distance


class ExerciseAnalyzer:
    def __init__(self):
        # rep counters / stage trackers per exercise
        self.counters = {
            "Squat": 0,
            "Push-up": 0,
            "Lunge": 0,
            "Jumping Jack": 0,
        }
        self.stage = {ex: None for ex in self.counters}  # "up" / "down"
        self.plank_seconds = 0.0
        self._plank_last_tick = None

        # form-quality tracking: list of booleans (True = good form) per rep event
        self.form_events = []

        # buffers for smoothing the auto-classifier's decision (avoid flicker)
        self._label_history = deque(maxlen=8)

        self.last_feedback = ""
        self.current_exercise = "Detecting..."

    # ------------------------------------------------------------------
    # Auto exercise classification
    # ------------------------------------------------------------------
    def classify_exercise(self, lm):
        """
        Heuristic whole-body classifier. Uses relative joint geometry
        (torso orientation, limb spread, stance width) rather than any
        single hard-coded exercise, so it generalizes across body types
        and camera distances.
        """
        try:
            l_sh, r_sh = lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"]
            l_hip, r_hip = lm["LEFT_HIP"], lm["RIGHT_HIP"]
            l_knee, r_knee = lm["LEFT_KNEE"], lm["RIGHT_KNEE"]
            l_ank, r_ank = lm["LEFT_ANKLE"], lm["RIGHT_ANKLE"]
            l_wr, r_wr = lm["LEFT_WRIST"], lm["RIGHT_WRIST"]
            l_el, r_el = lm["LEFT_ELBOW"], lm["RIGHT_ELBOW"]
        except KeyError:
            return self.current_exercise  # not enough landmarks yet

        shoulder_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
        hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        ankle_mid = ((l_ank[0] + r_ank[0]) / 2, (l_ank[1] + r_ank[1]) / 2)

        # Robust orientation / pose-shape signals (no need for torso angle math):
        body_height = distance(shoulder_mid, ankle_mid) + 1e-6
        torso_is_horizontal = abs(shoulder_mid[1] - hip_mid[1]) < 0.35 * body_height

        stance_width = distance(l_ank, r_ank)
        shoulder_width = distance(l_sh, r_sh) + 1e-6
        wide_stance = stance_width > 1.6 * shoulder_width

        arms_raised = (l_wr[1] < l_sh[1]) and (r_wr[1] < r_sh[1])

        knee_gap = distance(l_knee, r_knee)
        ankle_forward_offset = abs(l_ank[0] - r_ank[0])
        staggered_stance = ankle_forward_offset > 1.3 * shoulder_width and not wide_stance

        elbow_bent_low = (l_wr[1] > l_sh[1]) and (r_wr[1] > r_sh[1]) and torso_is_horizontal

        label = self.current_exercise

        if torso_is_horizontal and elbow_bent_low:
            label = "Push-up"
        elif torso_is_horizontal:
            label = "Plank"
        elif wide_stance and arms_raised:
            label = "Jumping Jack"
        elif staggered_stance:
            label = "Lunge"
        else:
            label = "Squat"

        self._label_history.append(label)
        # majority vote over recent frames to avoid flicker
        stable_label = max(set(self._label_history), key=self._label_history.count)
        self.current_exercise = stable_label
        return stable_label

    # ------------------------------------------------------------------
    # Per-exercise analysis (rep counting + form check)
    # ------------------------------------------------------------------
    def analyze(self, lm, exercise=None):
        """
        Run the appropriate rep-counter/form-checker for the given (or
        auto-detected) exercise. Returns a dict with rendering info.
        """
        if not lm:
            return {"exercise": self.current_exercise, "feedback": "No person detected", "angle": None}

        ex = exercise or self.classify_exercise(lm)

        if ex == "Squat":
            return self._analyze_squat(lm)
        elif ex == "Push-up":
            return self._analyze_pushup(lm)
        elif ex == "Lunge":
            return self._analyze_lunge(lm)
        elif ex == "Jumping Jack":
            return self._analyze_jumping_jack(lm)
        elif ex == "Plank":
            return self._analyze_plank(lm)
        else:
            return {"exercise": ex, "feedback": "", "angle": None}

    def _record_form(self, good):
        self.form_events.append(bool(good))

    def _analyze_squat(self, lm):
        hip, knee, ankle = lm["LEFT_HIP"], lm["LEFT_KNEE"], lm["LEFT_ANKLE"]
        shoulder = lm["LEFT_SHOULDER"]
        knee_angle = calculate_angle(hip[:2], knee[:2], ankle[:2])
        back_angle = calculate_angle(shoulder[:2], hip[:2], knee[:2])

        feedback = ""
        good_form = True

        # knee valgus check: knee shouldn't collapse inward past ankle x-position
        if knee[0] < ankle[0] - 25:
            feedback = "Push your knee outward, don't let it cave in"
            good_form = False

        if knee_angle < 100 and back_angle < 140:
            feedback = feedback or "Keep your chest up, avoid leaning too far forward"
            good_form = False

        stage = self.stage["Squat"]
        if knee_angle > 160:
            self.stage["Squat"] = "up"
        if knee_angle < 95 and stage == "up":
            self.stage["Squat"] = "down"
            self.counters["Squat"] += 1
            self._record_form(good_form)
            if not feedback:
                feedback = "Good depth!"

        return {"exercise": "Squat", "feedback": feedback, "angle": knee_angle,
                "reps": self.counters["Squat"]}

    def _analyze_pushup(self, lm):
        shoulder, elbow, wrist = lm["LEFT_SHOULDER"], lm["LEFT_ELBOW"], lm["LEFT_WRIST"]
        hip, ankle = lm["LEFT_HIP"], lm["LEFT_ANKLE"]
        elbow_angle = calculate_angle(shoulder[:2], elbow[:2], wrist[:2])
        body_line_angle = calculate_angle(shoulder[:2], hip[:2], ankle[:2])

        feedback = ""
        good_form = True
        if body_line_angle < 155:
            feedback = "Keep your hips up, don't let them sag"
            good_form = False

        stage = self.stage["Push-up"]
        if elbow_angle > 160:
            self.stage["Push-up"] = "up"
        if elbow_angle < 90 and stage == "up":
            self.stage["Push-up"] = "down"
            self.counters["Push-up"] += 1
            self._record_form(good_form)
            if not feedback:
                feedback = "Full range, nice rep!"

        return {"exercise": "Push-up", "feedback": feedback, "angle": elbow_angle,
                "reps": self.counters["Push-up"]}

    def _analyze_lunge(self, lm):
        hip, knee, ankle = lm["LEFT_HIP"], lm["LEFT_KNEE"], lm["LEFT_ANKLE"]
        front_knee_angle = calculate_angle(hip[:2], knee[:2], ankle[:2])

        feedback = ""
        good_form = True
        if knee[0] > ankle[0] + 40:
            feedback = "Don't let your front knee pass your toes"
            good_form = False

        stage = self.stage["Lunge"]
        if front_knee_angle > 160:
            self.stage["Lunge"] = "up"
        if front_knee_angle < 100 and stage == "up":
            self.stage["Lunge"] = "down"
            self.counters["Lunge"] += 1
            self._record_form(good_form)
            if not feedback:
                feedback = "Solid lunge!"

        return {"exercise": "Lunge", "feedback": feedback, "angle": front_knee_angle,
                "reps": self.counters["Lunge"]}

    def _analyze_jumping_jack(self, lm):
        l_wr, r_wr = lm["LEFT_WRIST"], lm["RIGHT_WRIST"]
        l_sh, r_sh = lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"]
        l_ank, r_ank = lm["LEFT_ANKLE"], lm["RIGHT_ANKLE"]

        arms_up = l_wr[1] < l_sh[1] and r_wr[1] < r_sh[1]
        legs_apart = distance(l_ank, r_ank) > 1.5 * distance(l_sh, r_sh)
        open_pose = arms_up and legs_apart

        feedback = ""
        stage = self.stage["Jumping Jack"]
        if not open_pose:
            self.stage["Jumping Jack"] = "closed"
        if open_pose and stage == "closed":
            self.stage["Jumping Jack"] = "open"
            self.counters["Jumping Jack"] += 1
            self._record_form(True)
            feedback = "Nice and full extension!"

        return {"exercise": "Jumping Jack", "feedback": feedback, "angle": None,
                "reps": self.counters["Jumping Jack"]}

    def _analyze_plank(self, lm):
        shoulder, hip, ankle = lm["LEFT_SHOULDER"], lm["LEFT_HIP"], lm["LEFT_ANKLE"]
        body_line_angle = calculate_angle(shoulder[:2], hip[:2], ankle[:2])

        feedback = ""
        good_form = True
        if body_line_angle < 160:
            feedback = "Straighten your body, hips too low/high"
            good_form = False
        else:
            feedback = "Great plank form, hold it!"

        now = time.time()
        if self._plank_last_tick is not None and good_form:
            self.plank_seconds += now - self._plank_last_tick
        self._plank_last_tick = now
        self._record_form(good_form)

        return {"exercise": "Plank", "feedback": feedback, "angle": body_line_angle,
                "hold_seconds": round(self.plank_seconds, 1)}

    # ------------------------------------------------------------------
    def form_score(self):
        """Percentage of tracked rep/frame events performed with good form."""
        if not self.form_events:
            return 100.0
        return round(100.0 * sum(self.form_events) / len(self.form_events), 1)

    def total_reps(self):
        return sum(self.counters.values())