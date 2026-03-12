#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

from src.plotting.plots import (
    plot_system_schematic,
    plot_thermal_schematic,
    plot_timeseries_panel,
    plot_chip_snapshots,
    plot_spacetime,
    plot_coolant_flow,
    plot_summary,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/main_run/results_per_step.csv")
    ap.add_argument("--summary", default="outputs/main_run/summary.csv")
    ap.add_argument("--outdir", default="outputs/main_run/figures")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)
    if Path(args.summary).exists():
        try:
            summary = pd.read_csv(args.summary)
        except EmptyDataError:
            summary = pd.DataFrame()
    else:
        summary = pd.DataFrame()

    if df.empty:
        raise RuntimeError("Results CSV is empty; run eval_all longer or use a checkpointed output.")

    plot_system_schematic(str(out / "01_system_schematic.png"))
    plot_thermal_schematic(str(out / "02_thermal_plant_schematic.png"))
    fam = "combined_stress" if (df["family"] == "combined_stress").any() else str(df["family"].iloc[0])
    plot_timeseries_panel(df, str(out / "03_timeseries_id_timeseries.png"), scenario_family=fam, split="id")
    plot_chip_snapshots(df[df["split"] == "id"], str(out / "04_chip_snapshots.png"), methods=["pid", "deterministic_mpc", "robust_mpc", "non_contextual_dro", "contextual_dro"])
    plot_spacetime(df[df["split"] == "id"], str(out / "05_spacetime_contextual_dro.png"), method="contextual_dro")
    plot_coolant_flow(df[df["split"] == "id"], str(out / "06_coolant_flow_contextual_dro.png"), method="contextual_dro")
    plot_summary(summary, str(out / "07_summary_bars.png"))

    print(f"Saved figures to {out}")
