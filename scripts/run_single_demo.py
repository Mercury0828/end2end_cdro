#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np

from src.utils.config import load_yaml
from src.scenario.thermal_workload import ThermalScenarioConfig, build_dataset
from src.plant.plant_factory import make_plant
from src.controllers.main_suite import (
    SurrogateModel,
    PIDController,
    DeterministicMPCController,
    RobustMPCController,
    NonContextualDROController,
    ContextualDROController,
)


def rolling_forecast(arr: np.ndarray, t: int, H: int) -> np.ndarray:
    end = min(len(arr), t + H)
    fc = arr[t:end]
    if len(fc) < H:
        fc = np.concatenate([fc, np.full(H - len(fc), fc[-1])])
    return fc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="contextual_dro", choices=["pid", "deterministic_mpc", "robust_mpc", "non_contextual_dro", "contextual_dro"])
    args = ap.parse_args()

    cfg = load_yaml("configs/base.yaml")
    env = make_plant(cfg["plant"])
    sc = build_dataset(ThermalScenarioConfig(**cfg["scenario"]), ["combined_stress"], 1, seed=cfg["seed"])[0]
    m = SurrogateModel(**cfg["surrogate"])

    ctrls = {
        "pid": PIDController(target_hotspot=cfg["safety"]["hotspot_threshold"]),
        "deterministic_mpc": DeterministicMPCController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"]),
        "robust_mpc": RobustMPCController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"], rho_uncertainty=cfg["controller"]["robust_rho"]),
        "non_contextual_dro": NonContextualDROController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"], rho_dro=cfg["controller"]["dro_rho"]),
        "contextual_dro": ContextualDROController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"], rho_min=cfg["controller"]["ctx_rho_min"], rho_max=cfg["controller"]["ctx_rho_max"], rho_gain=cfg["controller"]["ctx_rho_gain"]),
    }
    ctrl = ctrls[args.controller]

    env.reset(); ctrl.reset()
    power_total = sc["q_map"].sum(axis=(1, 2))
    hs = []
    for t in sc["t"]:
        st = env.get_state()
        p_fc = rolling_forecast(power_total, t, cfg["controller"]["horizon"])
        tin_fc = rolling_forecast(sc["Tin"], t, cfg["controller"]["horizon"])
        tamb_fc = rolling_forecast(sc["Tamb"], t, cfg["controller"]["horizon"])
        u, rho, _ = ctrl.act(st, float(sc["context"][t]), p_fc, tamb_fc, tin_fc)
        ns = env.step(u, float(sc["Tamb"][t]), float(sc["Tin"][t]), sc["q_map"][t])
        hs.append(ns["hotspot"])
    print(args.controller, {"peak_hotspot": float(np.max(hs)), "mean_hotspot": float(np.mean(hs))})
