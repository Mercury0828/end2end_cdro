#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd

from src.utils.config import load_yaml
from src.scenario.calm_burst_calm import ScenarioConfig, generate_dataset
from src.plotting.plots import plot_scenario, plot_mechanism, plot_outcome

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/default_run/results_per_step.csv")
    ap.add_argument("--outdir", default="outputs/default_run")
    args = ap.parse_args()

    base = load_yaml("configs/base.yaml")
    scfg = ScenarioConfig(**load_yaml("configs/scenario.yaml"))
    sc = generate_dataset(scfg, 1, seed=base["seed"] + 99)[0]
    plot_scenario(sc, f"{args.outdir}/scenario.png")

    df = pd.read_csv(args.results)
    first_ep = df[df["episode"] == 0]
    plot_mechanism(first_ep[first_ep["controller"].isin(["e2e_cdro", "static_cdro", "ddro"])], f"{args.outdir}/mechanism.png")
    plot_outcome(first_ep[first_ep["controller"].isin(["e2e_cdro", "static_cdro", "mpc"])], f"{args.outdir}/outcome.png", base["safety_threshold"])
    print("Plots saved.")
