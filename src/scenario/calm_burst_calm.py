"""Calm-Burst-Calm scenario generation aligned with synthetic CDRO mechanism tests."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ScenarioConfig:
    T: int = 100
    calm_end: int = 40
    burst_end: int = 60
    xi_calm_mean: float = 0.2
    xi_calm_std: float = 0.05
    xi_burst_lognorm_mean: float = 0.0
    xi_burst_lognorm_sigma: float = 1.0
    tamb_base: float = 30.0
    tamb_amp: float = 4.0
    tamb_period: float = 100.0


def generate_trajectory(cfg: ScenarioConfig, rng: np.random.Generator):
    t = np.arange(cfg.T)
    z = np.zeros(cfg.T)
    z[cfg.calm_end:cfg.burst_end] = 1.0

    xi = np.empty(cfg.T)
    calm_mask = t < cfg.calm_end
    burst_mask = (t >= cfg.calm_end) & (t < cfg.burst_end)
    rec_mask = t >= cfg.burst_end

    xi[calm_mask] = rng.normal(cfg.xi_calm_mean, cfg.xi_calm_std, calm_mask.sum())
    xi[burst_mask] = rng.lognormal(cfg.xi_burst_lognorm_mean, cfg.xi_burst_lognorm_sigma, burst_mask.sum())
    xi[rec_mask] = rng.normal(cfg.xi_calm_mean, cfg.xi_calm_std, rec_mask.sum())

    tamb = cfg.tamb_base + cfg.tamb_amp * np.sin(2 * np.pi * t / cfg.tamb_period)
    return {"t": t, "z": z, "xi": xi, "Tamb": tamb}


def generate_dataset(cfg: ScenarioConfig, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return [generate_trajectory(cfg, rng) for _ in range(n)]
