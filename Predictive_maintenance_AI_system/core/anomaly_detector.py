import collections 
import numpy as np 
import config 

class AnomalyDetecter:
    def __init__(self, window_size=config.ANOMALY_WINDOW_SIZE):
        self.history = collections.deque(maxlen=window_size)

    def evaluate_value(self, continous_value: float) -> tuple[bool, float]:

        if len(self.history) < 30:
            self.history.append(continous_value)
            return False, 0.0

    history_array = np.array(self.history)
    mean_val = np.mean(history_array)
    std_val = np.std(history_array)
    
    if std_val < 0.01:
        std_val = 0.01
    
    dynamic_threshold = mean_val + (config.ANOMALY_SIGMA_MULTIPLIER * std_val)
    self.history.append(continous_value)
    
    is_anomalous = continous_value > dynamic_threshold
    
    anomaly_score = (continous_value - mean_val) / (config.ANOMALY_SIGMA_MULTIPLIER * std_val)

    anomaly_score = max(0.0, anomaly_score)

    self.history.append(continous_value)

    return is_anomalous, anomaly_score
        