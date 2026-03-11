#!/usr/bin/env python
from __future__ import annotations

import argparse
from src.utils.config import load_yaml
from src.scenario.calm_burst_calm import ScenarioConfig, generate_dataset
from src.plant.plant_factory import make_plant
from src.controllers.mpc import MPCController
from src.controllers.static_cdro import StaticCDROController
from src.controllers.ddro import DDROController
from src.controllers.e2e_cdro import E2ECDROController
from src.runner import rollout

MAP = {
    "mpc": MPCController,
    "static_cdro": StaticCDROController,
    "ddro": DDROController,
    "e2e_cdro": E2ECDROController,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="mpc", choices=list(MAP.keys()))
    args = ap.parse_args()

    base = load_yaml("configs/base.yaml")
    scfg = ScenarioConfig(**load_yaml("configs/scenario.yaml"))
    sc = generate_dataset(scfg, 1, seed=base["seed"])[0]
    env = make_plant(base["plant"])
    p = base["plant"]
    ctrl = MAP[args.controller](p["alpha"], p["beta"], p["gamma"], 10.0, base["lambda_penalty"])
    _, m = rollout(env, ctrl, sc, base["lambda_penalty"], base["safety_threshold"])
    print(args.controller, m)
