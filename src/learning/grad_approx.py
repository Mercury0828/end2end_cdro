from __future__ import annotations


def finite_diff_grad(cost_fn, rho: float, eps: float = 1e-1) -> float:
    c1 = cost_fn(rho + eps)
    c0 = cost_fn(max(0.0, rho - eps))
    return (c1 - c0) / (2 * eps)
