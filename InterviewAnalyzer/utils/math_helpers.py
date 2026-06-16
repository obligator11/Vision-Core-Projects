import numpy as np

class OneEuroFilter:
    """Implements a low-latency signal smoothing filter to eliminate 
    high-frequency jitter from neural network landmark coordinate outputs."""
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.atleast_1d(x0).astype(float)
        self.dx_prev = np.atleast_1d(dx0).astype(float)
        self.t_prev = float(t0)

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter_signal(self, t, x):
        x = np.atleast_1d(x).astype(float)
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        d_x = (x - self.x_prev) / dt
        clr = self._alpha(self.d_cutoff, dt)
        dx_hat = clr * d_x + (1.0 - clr) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat