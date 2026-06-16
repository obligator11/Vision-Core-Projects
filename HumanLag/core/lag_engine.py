import numpy as np
from collections import deque
from typing import Tuple

class LagCalculationEngine:
    """
    Computes spatial deviations using Euclidean distance checks and models
    approximate temporal reaction delays across a historical moving window.
    """
    def __init__(self, calculation_window: int = 50):
        self.calculation_window = calculation_window
        self.error_history = deque(maxlen=calculation_window)
        self.lag_history_ms = deque(maxlen=calculation_window)

    def calculate_metrics(self, actual_pos: np.ndarray, predicted_pos: np.ndarray) -> Tuple[float, float]:
        # Spatial Distance Error evaluation via Euclidean vector norms
        spatial_error = float(np.linalg.norm(actual_pos - predicted_pos))
        self.error_history.append(spatial_error)
        
        # Estimate response time delay based on spatial variance vs velocity shifts
        if len(self.error_history) > 5:
            mean_error = np.mean(self.error_history)
            error_velocity = np.abs(spatial_error - mean_error) + 1e-3
            # Translate spatial distance error into latency metrics
            estimated_lag = float((spatial_error / error_velocity) * 8.0)
        else:
            estimated_lag = 0.0
            
        # Establish stable telemetry ceilings (0 ms to 450 ms range boundaries)
        estimated_lag = np.clip(estimated_lag, 0.0, 450.0)
        self.lag_history_ms.append(estimated_lag)
        
        return spatial_error, estimated_lag

    def get_lag_history(self) -> deque:
        return self.lag_history_ms