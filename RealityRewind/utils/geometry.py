import numpy as np

def extrapolate_reverse_vector(point: np.ndarray, velocity: np.ndarray, delta_t: float) -> tuple:
    """
    Uses vector kinematics equations to calculate linear backward path points 
    and constructs time arrows aligned with movement headings.
    """
    # Reverse direction vector mapping: x_past = x_now - (v * dt)
    past_point = point - (velocity * delta_t)
    
    # Extract structural head angle
    angle = np.arctan2(-velocity[1], -velocity[0])
    return past_point.astype(np.int32), angle