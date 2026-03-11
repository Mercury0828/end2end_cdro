"""Practical convex robust layer for scalar thermal system."""
from __future__ import annotations

from dataclasses import dataclass
import cvxpy as cp
from .cvx_utils import solve_prob


@dataclass
class RobustResult:
    u: float
    status: str
    solve_time: float | None


class ScalarRobustLayer:
    def __init__(self, alpha: float, beta: float, gamma: float, u_max: float, lambda_penalty: float, xi_nominal: float = 0.2):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.u_max, self.lambda_penalty = u_max, lambda_penalty
        self.xi_nominal = xi_nominal

        self.x = cp.Parameter(name="x")
        self.tamb = cp.Parameter(name="tamb")
        self.rho = cp.Parameter(nonneg=True, name="rho")

        self.u = cp.Variable(name="u")
        self.s = cp.Variable(nonneg=True, name="s")

        xi_worst = self.xi_nominal + self.rho
        x_next = self.x + self.alpha * (self.tamb - self.x) - self.beta * self.u + self.gamma * xi_worst

        constraints = [self.u >= 0, self.u <= self.u_max, self.s >= x_next]
        objective = cp.Minimize(self.u + self.lambda_penalty * self.s)
        self.problem = cp.Problem(objective, constraints)

    def act(self, x: float, tamb: float, rho: float) -> RobustResult:
        self.x.value = x
        self.tamb.value = tamb
        self.rho.value = max(float(rho), 0.0)
        status, solve_time = solve_prob(self.problem)
        u = float(self.u.value) if self.u.value is not None else self.u_max
        return RobustResult(u=u, status=status, solve_time=solve_time)
