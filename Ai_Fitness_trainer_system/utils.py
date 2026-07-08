

import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b, formed by points a-b-c.
    a, b, c are (x, y) tuples/lists (pixel or normalized coordinates).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    )
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def distance(a, b):
    """Euclidean distance between two 2D points."""
    a = np.array(a)
    b = np.array(b)
    return float(np.linalg.norm(a - b))


def smooth(value, history, window=5):
    """
    Simple moving-average smoother to reduce landmark jitter.
    'history' is a list that gets mutated in place.
    """
    history.append(value)
    if len(history) > window:
        history.pop(0)
    return sum(history) / len(history)


# Very rough MET-based calorie estimates (kcal per rep or per second),
# calibrated for an "average" 70kg adult. These are approximations
# meant to give the user a directional sense of effort, not medical-grade data.
CALORIES_PER_REP = {
    "Squat": 0.32,
    "Push-up": 0.29,
    "Lunge": 0.30,
    "Jumping Jack": 0.20,
}

CALORIES_PER_SECOND_HOLD = {
    "Plank": 0.045,
}