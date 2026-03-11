"""Working first-pass trainer for E2E-CDRO with pragmatic target shaping."""
from __future__ import annotations

import numpy as np
from src.controllers.e2e_cdro import E2ECDROController


def train_e2e(controller: E2ECDROController, scenarios: list[dict], epochs: int = 20):
    losses = []
    for _ in range(epochs):
        obs, target = [], []
        for sc in scenarios:
            z = sc["z"]
            # proactive target: low in calm, high in burst
            rho_star = np.where(z > 0.5, 1.4, 0.25)
            x_proxy = -1.0 + 0.1 * np.cumsum(sc["xi"] - sc["xi"].mean())
            for xt, zt, rt in zip(x_proxy, z, rho_star):
                obs.append([xt, zt])
                target.append(rt)
        losses.append(controller.update_supervised(obs, target))
    return losses
