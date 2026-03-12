#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.utils.io import ensure_dir
from src.utils.seeds import set_seed
from src.scenario.thermal_workload import ThermalScenarioConfig, build_dataset
from src.plant.plant_factory import make_plant
from src.controllers.main_suite import (
    SurrogateModel,
    PIDController,
    DeterministicMPCController,
    RobustMPCController,
    NonContextualDROController,
    ContextualDROController,
    tune_pid_on_validation,
)


def rolling_forecast(arr: np.ndarray, t: int, H: int) -> np.ndarray:
    end = min(len(arr), t + H)
    fc = arr[t:end]
    if len(fc) < H:
        fc = np.concatenate([fc, np.full(H - len(fc), fc[-1])])
    return fc


def compute_metrics(ep_df: pd.DataFrame, threshold: float) -> dict:
    x = ep_df["hotspot"].to_numpy()
    u = ep_df["u"].to_numpy()
    rho = ep_df["rho"].to_numpy()
    return {
        "total_cost": float(ep_df["cost"].sum()),
        "energy_usage": float(np.sum(u)),
        "violation_rate": float(np.mean(x > threshold)),
        "max_overshoot": float(np.max(np.maximum(0.0, x - threshold))),
        "time_above_threshold": float(np.sum(x > threshold)),
        "peak_hotspot_temperature": float(np.max(x)),
        "recovery_time_after_burst": float((ep_df["phase"] == "recovery").to_numpy().argmax() if np.any(ep_df["phase"] == "recovery") else 0),
        "control_smoothness": float(np.mean(np.abs(np.diff(u)))) if len(u) > 1 else 0.0,
        "avg_rho": float(np.mean(rho)),
        "rho_std": float(np.std(rho)),
        "avg_chip_temp": float(np.mean(ep_df["chip_avg"])),
    }


def run_episode(controller, scenario, env, cfg):
    env.reset()
    controller.reset()
    H = cfg["controller"]["horizon"]
    rows = []
    solve_times = []

    for t in scenario["t"]:
        st = env.get_state()
        power_total = scenario["q_map"].sum(axis=(1, 2))
        p_fc = rolling_forecast(power_total, t, H)
        tin_fc = rolling_forecast(scenario["Tin"], t, H)
        tamb_fc = rolling_forecast(scenario["Tamb"], t, H)
        u, rho, info = controller.act(st, float(scenario["context"][t]), p_fc, tamb_fc, tin_fc)
        ns = env.step(u, float(scenario["Tamb"][t]), float(scenario["Tin"][t]), scenario["q_map"][t])
        penalty = max(0.0, ns["hotspot"] - cfg["safety"]["hotspot_threshold"])
        cost = cfg["safety"]["energy_weight"] * u + cfg["safety"]["violation_weight"] * penalty
        solve_times.append(0.0 if info is None else float(info.get("solve_time", 0.0)))

        T = len(scenario["t"])
        phase = "calm" if t < T // 3 else ("burst" if t < (2 * T) // 3 else "recovery")
        rows.append({
            "t": int(t), "family": scenario["family"], "phase": phase, "context": float(scenario["context"][t]),
            "power_total": float(power_total[t]), "u": float(u), "rho": float(rho),
            "hotspot": float(ns["hotspot"]), "chip_avg": float(ns["chip_avg"]), "t_plate": float(ns["t_plate"]),
            "t_coolant": float(ns["t_coolant"]), "Tamb": float(scenario["Tamb"][t]), "Tin": float(scenario["Tin"][t]),
            "cost": float(cost), "solve_time": solve_times[-1], "peak_tile_idx": int(np.argmax(ns["chip"])),
            "chip_flat": json.dumps(ns["chip"].ravel().tolist()),
        })
    ep_df = pd.DataFrame(rows)
    metrics = compute_metrics(ep_df, cfg["safety"]["hotspot_threshold"])
    metrics["solver_runtime_mean"] = float(np.mean(solve_times))
    return ep_df, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.base)
    set_seed(cfg["seed"])
    out_dir = Path(args.out or cfg["evaluation"]["output_dir"])
    ensure_dir(out_dir)

    scfg = ThermalScenarioConfig(**cfg["scenario"]) 
    val = build_dataset(scfg, cfg["evaluation"]["id_families"], cfg["evaluation"]["n_val_per_family"], seed=cfg["seed"] + 1)
    test_id = build_dataset(scfg, cfg["evaluation"]["id_families"], cfg["evaluation"]["n_test_per_family"], seed=cfg["seed"] + 2)
    test_ood = build_dataset(scfg, cfg["evaluation"]["ood_families"], cfg["evaluation"]["n_test_per_family"], seed=cfg["seed"] + 3)

    env_factory = lambda: make_plant(cfg["plant"])
    m = SurrogateModel(**cfg["surrogate"])

    pid = PIDController(target_hotspot=cfg["safety"]["hotspot_threshold"])
    tune = tune_pid_on_validation(pid, val, env_factory)

    controllers = {
        "pid": pid,
        "deterministic_mpc": DeterministicMPCController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"], w_slack=cfg["controller"]["w_slack"], rho_fixed=0.0),
        "robust_mpc": RobustMPCController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"], rho_uncertainty=cfg["controller"]["robust_rho"], w_slack=cfg["controller"]["w_slack"]),
        "non_contextual_dro": NonContextualDROController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"], rho_dro=cfg["controller"]["dro_rho"], w_slack=cfg["controller"]["w_slack"]),
        "contextual_dro": ContextualDROController(m, horizon=cfg["controller"]["horizon"], target_hotspot=cfg["safety"]["hotspot_threshold"], rho_min=cfg["controller"]["ctx_rho_min"], rho_max=cfg["controller"]["ctx_rho_max"], rho_gain=cfg["controller"]["ctx_rho_gain"], w_slack=cfg["controller"]["w_slack"]),
    }

    all_step, all_ep = [], []
    for seed in cfg["evaluation"]["seeds"]:
        set_seed(seed)
        for split, dataset in [("id", test_id), ("ood", test_ood)]:
            for c_name, controller in controllers.items():
                for ep, sc in enumerate(dataset):
                    env = env_factory()
                    step_df, met = run_episode(controller, sc, env, cfg)
                    step_df["controller"] = c_name
                    step_df["split"] = split
                    step_df["seed"] = seed
                    step_df["episode"] = ep
                    all_step.append(step_df)
                    all_ep.append({"controller": c_name, "split": split, "seed": seed, "episode": ep, **met})

    step_df = pd.concat(all_step, ignore_index=True)
    ep_df = pd.DataFrame(all_ep)
    summary = ep_df.groupby(["controller", "split"]).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    step_df.to_csv(out_dir / "results_per_step.csv", index=False)
    ep_df.to_csv(out_dir / "episode_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "pid_tuning.json").write_text(json.dumps(tune, indent=2))
    print(summary[["controller", "split", "total_cost_mean", "violation_rate_mean", "peak_hotspot_temperature_mean"]])
