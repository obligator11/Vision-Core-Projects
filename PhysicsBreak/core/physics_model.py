import numpy as np

class PhysicsModel:
    """
    Applies discrete numerical methods (backward finite differences) to evaluate metrics 
    and models continuous integration vectors under uniform gravitational acceleration.
    """
    def __init__(self, pixel_gravity: float = 650.0):
        # Gravity acting straight down along video Y-axis (pixels/sec^2)
        self.gravity = np.array([0.0, pixel_gravity], dtype=float)
        
    def calculate_kinematics(self, history: list) -> tuple:
        """
        Evaluates discrete time velocity and acceleration metrics across frame points.
        Returns: (velocity_vector, acceleration_vector, delta_t)
        """
        if len(history) < 2:
            return np.zeros(2), np.zeros(2), 0.0
            
        pos_curr, t_curr = history[-1]
        pos_prev, t_prev = history[-2]
        
        dt = t_curr - t_prev
        if dt <= 0:
            return np.zeros(2), np.zeros(2), 0.0
            
        # Velocity via first-order backward difference
        velocity = (pos_curr - pos_prev) / dt
        
        # Acceleration via second-order difference
        acceleration = np.zeros(2)
        if len(history) >= 3:
            pos_old, t_old = history[-3]
            dt_old = t_prev - t_old
            if dt_old > 0:
                vel_prev = (pos_prev - pos_old) / dt_old
                acceleration = (velocity - vel_prev) / dt
                
        return velocity, acceleration, dt

    def predict_next_state(self, position: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculates forward kinematics extrapolation: x_hat = x + v*dt + 0.5*g*dt^2
        """
        if dt <= 0:
            dt = 0.033 # Standard 30 FPS fallback frame delta
        return position + (velocity * dt) + (0.5 * self.gravity * (dt ** 2))