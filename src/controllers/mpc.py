from __future__ import annotations

from .base import BaseController
from src.optimization.robust_layer import ScalarRobustLayer


class MPCController(BaseController):
    name = "mpc"

    def __init__(self, alpha, beta, gamma, u_max, lambda_penalty):
        self.layer = ScalarRobustLayer(alpha, beta, gamma, u_max, lambda_penalty, xi_nominal=0.2)

    def reset(self):
        return None

    def act(self, x, z, tamb, xi_hat=None):
        xi = 0.2 if xi_hat is None else float(xi_hat)
        self.layer.xi_nominal = xi
        out = self.layer.act(x, tamb, rho=0.0)
        return out.u, 0.0, out
