"""Scenario generation for the main liquid-cooling-oriented thermal benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

Family = Literal[
    "calm_burst_recovery",
    "burst_train",
    "drifting_hotspot",
    "broad_plateau",
    "combined_stress",
]


@dataclass
class ThermalScenarioConfig:
    T: int = 420
    grid_nx: int = 8
    grid_ny: int = 8
    base_power: float = 1.2
    burst_power: float = 5.0
    noise_std: float = 0.12
    ambient_base: float = 24.0
    ambient_amp: float = 2.0
    coolant_base: float = 20.0
    coolant_amp: float = 1.5


def _gaussian_blob(nx: int, ny: int, cx: float, cy: float, sigma: float = 1.1) -> np.ndarray:
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    return np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma)))


def generate_thermal_trajectory(cfg: ThermalScenarioConfig, rng: np.random.Generator, family: Family) -> dict:
    T, nx, ny = cfg.T, cfg.grid_nx, cfg.grid_ny
    t = np.arange(T)
    q = np.full((T, nx, ny), cfg.base_power, dtype=float)
    z = np.zeros(T, dtype=float)

    if family == "calm_burst_recovery":
        s, e = T // 3, (2 * T) // 3
        z[s:e] = 1.0
        blob = _gaussian_blob(nx, ny, nx * 0.65, ny * 0.4)
        q[s:e] += cfg.burst_power * blob
    elif family == "burst_train":
        width = max(16, T // 18)
        centers = [T // 6, T // 3, T // 2, int(0.72 * T), int(0.86 * T)]
        for c in centers:
            s, e = max(0, c - width // 2), min(T, c + width // 2)
            z[s:e] = 1.0
            blob = _gaussian_blob(nx, ny, rng.uniform(1.0, nx - 2.0), rng.uniform(1.0, ny - 2.0), sigma=0.9)
            q[s:e] += 0.9 * cfg.burst_power * blob
    elif family == "drifting_hotspot":
        z[:] = 0.8
        for k in range(T):
            cx = 1.0 + (nx - 2.0) * (k / max(1, T - 1))
            cy = 1.0 + (ny - 2.0) * (0.5 + 0.4 * np.sin(2 * np.pi * k / T))
            q[k] += 0.75 * cfg.burst_power * _gaussian_blob(nx, ny, cx, cy, sigma=1.0)
    elif family == "broad_plateau":
        s, e = T // 4, 3 * T // 4
        z[s:e] = 0.7
        q[s:e] += 0.45 * cfg.burst_power
    elif family == "combined_stress":
        z[:] = 0.6
        for k in range(T):
            amp = 0.3 * cfg.burst_power + 0.8 * cfg.burst_power * (0.5 + 0.5 * np.sin(2 * np.pi * k / 55.0))
            cx = nx * (0.2 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * k / 140.0)))
            cy = ny * (0.2 + 0.6 * (0.5 + 0.5 * np.cos(2 * np.pi * k / 97.0)))
            q[k] += amp * _gaussian_blob(nx, ny, cx, cy, sigma=0.9)
            if k % 50 < 12:
                z[k] = 1.0
    else:
        raise ValueError(f"Unknown family: {family}")

    q += rng.normal(0.0, cfg.noise_std, size=q.shape)
    q = np.clip(q, 0.0, None)

    context = z + rng.normal(0.0, 0.15, size=T)
    context += 0.04 * (q.sum(axis=(1, 2)) - q.mean()) / max(1e-6, q.std())
    context = np.clip(context, 0.0, 1.5)

    tamb = cfg.ambient_base + cfg.ambient_amp * np.sin(2 * np.pi * t / (0.9 * T)) + rng.normal(0.0, 0.25, size=T)
    tin = cfg.coolant_base + cfg.coolant_amp * np.sin(2 * np.pi * t / (0.7 * T) + 0.8) + rng.normal(0.0, 0.2, size=T)

    return {"t": t, "family": family, "context": context, "q_map": q, "Tamb": tamb, "Tin": tin}


def build_dataset(cfg: ThermalScenarioConfig, families: list[Family], n_per_family: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for fam in families:
        for _ in range(n_per_family):
            out.append(generate_thermal_trajectory(cfg, rng, fam))
    rng.shuffle(out)
    return out
