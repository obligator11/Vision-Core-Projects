import numpy as np

class AnomalyEngine:
    """
    Compares observed trajectories against expected values using algebraic error spaces.
    """
    def __init__(self, noise_floor: float = 20.0, anomaly_limit: float = 55.0, critical_limit: float = 110.0):
        self.noise_floor = noise_floor
        self.anomaly_limit = anomaly_limit
        self.critical_limit = critical_limit
        
    def evaluate_deviation(self, actual: np.ndarray, predicted: np.ndarray) -> tuple:
        """
        Measures the absolute spatial offset magnitude of tracking observations.
        Returns: (deviation_vector, magnitude_scalar, status_flag)
        """
        deviation_vector = actual - predicted
        magnitude = np.linalg.norm(deviation_vector)
        
        if magnitude < self.noise_floor:
            status = "NOMINAL"
        elif magnitude < self.anomaly_limit:
            status = "DEVIATION"
        elif magnitude < self.critical_limit:
            status = "ANOMALY"
        else:
            status = "CRITICAL"
            
        return deviation_vector, magnitude, status