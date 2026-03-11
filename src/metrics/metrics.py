from __future__ import annotations

import numpy as np


def compute_episode_metrics(df, safety_threshold=0.0):
    x = df["x"].to_numpy()
    u = df["u"].to_numpy()
    rho = df["rho"].to_numpy()
    z = df["z"].to_numpy()
    return {
        "total_cost": float(df["cost"].sum()),
        "avg_energy": float(np.mean(u)),
        "violation_rate": float(np.mean(x > safety_threshold)),
        "max_overshoot": float(np.max(np.maximum(0.0, x - safety_threshold))),
        "adaptability_score": float(np.mean(rho[z > 0.5]) - np.mean(rho[z <= 0.5])),
        "mean_rho": float(np.mean(rho)),
        "std_rho": float(np.std(rho)),
    }
