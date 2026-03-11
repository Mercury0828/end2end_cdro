from __future__ import annotations

from .base import BaseController
from src.optimization.robust_layer import ScalarRobustLayer


class StaticCDROController(BaseController):
    name = "static_cdro"

    def __init__(self, alpha, beta, gamma, u_max, lambda_penalty, rho_fixed=0.9):
        self.layer = ScalarRobustLayer(alpha, beta, gamma, u_max, lambda_penalty, xi_nominal=0.2)
        self.rho_fixed = rho_fixed

    def reset(self):
        return None

    def act(self, x, z, tamb, xi_hat=None):
        out = self.layer.act(x, tamb, rho=self.rho_fixed)
        return out.u, self.rho_fixed, out
