"""Small CVXPY helpers."""
from __future__ import annotations

import cvxpy as cp


def solve_prob(prob: cp.Problem, solver: str = "ECOS"):
    try:
        prob.solve(solver=solver, warm_start=True)
    except Exception:
        prob.solve(solver="SCS", warm_start=True)
    return prob.status, getattr(prob.solver_stats, "solve_time", None)
