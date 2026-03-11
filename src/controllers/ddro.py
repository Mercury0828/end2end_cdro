from __future__ import annotations

from collections import deque
import numpy as np
from .base import BaseController
from src.optimization.robust_layer import ScalarRobustLayer


class DDROController(BaseController):
    name = "ddro"

    def __init__(self, alpha, beta, gamma, u_max, lambda_penalty, rho_min=0.2, rho_max=2.0, adapt_gain=2.5, window=8):
        self.layer = ScalarRobustLayer(alpha, beta, gamma, u_max, lambda_penalty, xi_nominal=0.2)
        self.rho_min, self.rho_max = rho_min, rho_max
        self.adapt_gain = adapt_gain
        self.window = window
        self.residuals = deque(maxlen=window)
        self.rho = rho_min

    def reset(self):
        self.residuals.clear()
        self.rho = self.rho_min

    def observe(self, xi, z):
        pred = 0.2 if z < 0.5 else np.exp(0.5)
        self.residuals.append(abs(xi - pred))
        m = np.mean(self.residuals) if self.residuals else 0.0
        self.rho = float(np.clip(self.rho_min + self.adapt_gain * m, self.rho_min, self.rho_max))

    def act(self, x, z, tamb, xi_hat=None):
        out = self.layer.act(x, tamb, rho=self.rho)
        return out.u, self.rho, out
