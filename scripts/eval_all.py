#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from src.utils.config import load_yaml, merge_dicts
from src.utils.seeds import set_seed
from src.utils.io import ensure_dir
from src.scenario.calm_burst_calm import ScenarioConfig, generate_dataset
from src.plant.plant_factory import make_plant
from src.controllers.mpc import MPCController
from src.controllers.static_cdro import StaticCDROController
from src.controllers.ddro import DDROController
from src.controllers.e2e_cdro import E2ECDROController
from src.controllers.ppo_controller import PPOController
from src.runner import rollout
from src.metrics.summarize import summarize_runs


def build_controllers(cfg):
    p = cfg["plant"]
    lam = cfg["lambda_penalty"]
    return {
        "mpc": MPCController(p["alpha"], p["beta"], p["gamma"], 10.0, lam),
        "static_cdro": StaticCDROController(p["alpha"], p["beta"], p["gamma"], 10.0, lam, rho_fixed=0.9),
        "ddro": DDROController(p["alpha"], p["beta"], p["gamma"], 10.0, lam),
        "e2e_cdro": E2ECDROController(p["alpha"], p["beta"], p["gamma"], 10.0, lam),
        "ppo": PPOController(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--scenario", default="configs/scenario.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = merge_dicts(load_yaml(args.base), {"scenario": load_yaml(args.scenario)})
    set_seed(cfg["seed"])

    out_dir = Path(args.out or cfg["evaluation"]["output_dir"])
    ensure_dir(out_dir)

    scfg = ScenarioConfig(**cfg["scenario"])
    test_set = generate_dataset(scfg, cfg["evaluation"]["n_test"], seed=cfg["seed"] + 2)

    controllers = build_controllers(cfg)
    all_steps, ep_rows = [], []

    for name, ctrl in controllers.items():
        for ep, sc in enumerate(test_set):
            env = make_plant(cfg["plant"])
            df, m = rollout(env, ctrl, sc, lambda_penalty=cfg["lambda_penalty"], safety_threshold=cfg["safety_threshold"])
            df["controller"] = name
            df["episode"] = ep
            all_steps.append(df)
            ep_rows.append({"controller": name, "episode": ep, **m})

    step_df = pd.concat(all_steps, ignore_index=True)
    ep_df = pd.DataFrame(ep_rows)
    summary = summarize_runs(ep_df)

    step_df.to_csv(out_dir / cfg["evaluation"]["results_csv"], index=False)
    summary.to_csv(out_dir / cfg["evaluation"]["summary_csv"], index=False)
    print(summary)
