"""Pure Python plant for single-zone first-order thermal dynamics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlantParams:
    alpha: float = 0.08
    beta: float = 0.35
    gamma: float = 0.6
    x0: float = -1.5
    dt: float = 1.0


class PythonThermalEnv:
    def __init__(self, params: PlantParams):
        self.params = params
        self.x = params.x0

    def reset(self, x0: float | None = None):
        self.x = self.params.x0 if x0 is None else float(x0)
        return self.x

    def step(self, action: float, Tamb: float, xi: float, eps: float = 0.0):
        p = self.params
        u = min(max(float(action), 0.0), 10.0)
        x_next = self.x + p.alpha * (Tamb - self.x) - p.beta * u + p.gamma * xi + eps
        self.x = x_next
        return x_next

    def get_state(self):
        return self.x
