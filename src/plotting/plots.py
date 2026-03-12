from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10})

PALETTE = {
    "pid": "#1f77b4",
    "deterministic_mpc": "#ff7f0e",
    "robust_mpc": "#2ca02c",
    "non_contextual_dro": "#9467bd",
    "contextual_dro": "#d62728",
}


def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _pick_row_at_or_before(d: pd.DataFrame, t: int) -> pd.Series | None:
    if d.empty:
        return None
    exact = d[d["t"] == t]
    if not exact.empty:
        return exact.iloc[0]
    le = d[d["t"] <= t]
    if not le.empty:
        return le.iloc[-1]
    return d.iloc[0]


def plot_system_schematic(out_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = {
        "scenario": (0.02, 0.3, 0.2, 0.4, "Scenario Generator\n(workload + context)"),
        "controller": (0.3, 0.3, 0.2, 0.4, "Controller\nPID/MPC/RMPC/DRO/CDRO"),
        "plant": (0.58, 0.3, 0.2, 0.4, "Thermal Plant\nchip + cold plate + coolant"),
        "metrics": (0.82, 0.3, 0.16, 0.4, "Metrics & Figures")
    }
    for _, (x, y, w, h, label) in boxes.items():
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#f4f6f8", edgecolor="#2c3e50", lw=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", weight="bold")
    for x0, x1 in [(0.22, 0.3), (0.5, 0.58), (0.78, 0.82)]:
        ax.annotate("", (x1, 0.5), (x0, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="#34495e"))
    ax.text(0.63, 0.2, "Measured hotspot, plate, coolant", color="#34495e")
    ax.text(0.33, 0.76, "Control: pump speed", color="#34495e")
    _save(fig, out_path)


def plot_thermal_schematic(out_path: str, nx=8, ny=8):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    chip = np.linspace(0.2, 1.0, nx * ny).reshape(nx, ny)
    ax.imshow(chip, extent=(0.05, 0.55, 0.2, 0.85), cmap="inferno", alpha=0.9)
    ax.add_patch(plt.Rectangle((0.04, 0.18), 0.52, 0.7, fill=False, lw=2, color="black"))
    ax.text(0.3, 0.9, "Chip tile array", ha="center", weight="bold")
    ax.add_patch(plt.Rectangle((0.6, 0.25), 0.15, 0.55, facecolor="#9bd3e6", edgecolor="black"))
    ax.text(0.675, 0.84, "Cold plate", ha="center")
    ax.annotate("Inlet", (0.78, 0.75), (0.92, 0.75), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("Outlet", (0.92, 0.3), (0.78, 0.3), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.82, 0.53, "Coolant path", rotation=90, ha="center")
    ax.annotate("Pump speed u", (0.78, 0.53), (0.93, 0.53), arrowprops=dict(arrowstyle="->", lw=2, color="tab:blue"), color="tab:blue")
    _save(fig, out_path)


def plot_timeseries_panel(df: pd.DataFrame, out_path: str, scenario_family: str, split: str):
    sub = df[(df["family"] == scenario_family) & (df["split"] == split)]
    if sub.empty:
        sub = df[df["split"] == split]
    if sub.empty:
        return
    ep0 = int(sub["episode"].min())
    sub = sub[sub["episode"] == ep0]

    fig, axs = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    for ctrl, g in sub.groupby("controller"):
        color = PALETTE.get(ctrl)
        axs[0].plot(g["t"], g["context"], color=color, alpha=0.9, label=ctrl)
        axs[1].plot(g["t"], g["power_total"], color=color, alpha=0.9)
        axs[2].plot(g["t"], g["u"], color=color, alpha=0.9)
        axs[3].plot(g["t"], g["hotspot"], color=color, alpha=0.9)
        axs[4].plot(g["t"], g["rho"], color=color, alpha=0.9)
    axs[0].set_ylabel("Context")
    axs[1].set_ylabel("Total power")
    axs[2].set_ylabel("Pump u")
    axs[3].set_ylabel("Hotspot (°C)")
    axs[4].set_ylabel("rho")
    axs[4].set_xlabel("time step")
    axs[3].axhline(75.0, ls="--", color="black", lw=1.2, label="safety threshold")
    axs[0].legend(ncol=3, loc="upper right")
    fig.suptitle(f"Controller comparison ({scenario_family}, {split}, ep={ep0})")
    _save(fig, out_path)


def _chip_from_row(row: pd.Series, nx: int, ny: int) -> np.ndarray:
    return np.array(json.loads(row["chip_flat"])).reshape(nx, ny)


def plot_chip_snapshots(df: pd.DataFrame, out_path: str, methods: list[str], nx=8, ny=8):
    if df.empty:
        return
    available_methods = [m for m in methods if (df["controller"] == m).any()]
    if not available_methods:
        return

    ref = df[(df["controller"] == available_methods[0])]
    ep0 = int(ref["episode"].min())
    ref = ref[ref["episode"] == ep0]
    times = [int(ref["t"].quantile(q)) for q in [0.2, 0.45, 0.6, 0.85]]

    fig, axs = plt.subplots(len(available_methods), len(times), figsize=(14, max(3, 2 * len(available_methods))))
    axs = np.atleast_2d(axs)
    vmin, vmax = df["chip_avg"].min(), df["hotspot"].max()
    im = None
    for i, m in enumerate(available_methods):
        d = df[(df["controller"] == m) & (df["episode"] == ep0)].sort_values("t")
        for j, t in enumerate(times):
            row = _pick_row_at_or_before(d, t)
            ax = axs[i, j]
            if row is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_axis_off()
                continue
            im = ax.imshow(_chip_from_row(row, nx, ny), cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t~{t}")
            if j == 0:
                ax.set_ylabel(m)
    if im is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label="Temperature (°C)")
    _save(fig, out_path)


def plot_spacetime(df: pd.DataFrame, out_path: str, method="contextual_dro"):
    d = df[(df["controller"] == method)]
    if d.empty:
        any_methods = list(df["controller"].unique()) if not df.empty else []
        if not any_methods:
            return
        method = any_methods[0]
        d = df[df["controller"] == method]
    ep0 = int(d["episode"].min())
    d = d[d["episode"] == ep0]
    if d.empty:
        return
    mat = np.stack([np.array(json.loads(v)) for v in d["chip_flat"]], axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat.T, aspect="auto", cmap="inferno", origin="lower")
    ax.set_xlabel("time step")
    ax.set_ylabel("tile index")
    ax.set_title(f"Space-time chip temperature evolution ({method})")
    fig.colorbar(im, ax=ax, label="Temperature (°C)")
    _save(fig, out_path)


def plot_coolant_flow(df: pd.DataFrame, out_path: str, method="contextual_dro"):
    d = df[(df["controller"] == method)]
    if d.empty:
        any_methods = list(df["controller"].unique()) if not df.empty else []
        if not any_methods:
            return
        method = any_methods[0]
        d = df[df["controller"] == method]
    ep0 = int(d["episode"].min())
    d = d[d["episode"] == ep0]
    if d.empty:
        return
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(d["t"], d["t_plate"], label="cold-plate", color="#1f77b4", lw=2)
    axs[0].plot(d["t"], d["t_coolant"], label="coolant", color="#17becf", lw=2)
    axs[0].plot(d["t"], d["hotspot"], label="hotspot", color="#d62728", alpha=0.7)
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("Temperature (°C)")
    axs[1].fill_between(d["t"], 0, d["u"], color="#9ecae1", alpha=0.8, label="pump speed")
    axs[1].set_ylabel("Pump u")
    axs[1].set_xlabel("time step")
    axs[1].legend()
    _save(fig, out_path)


def plot_summary(summary_df: pd.DataFrame, out_path: str):
    if summary_df.empty:
        return
    metrics = ["violation_rate_mean", "peak_hotspot_temperature_mean", "energy_usage_mean", "control_smoothness_mean"]
    id_df = summary_df[summary_df["split"] == "id"]
    if id_df.empty:
        id_df = summary_df
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()
    for i, met in enumerate(metrics):
        ax = axs[i]
        if met in id_df.columns:
            ax.bar(id_df["controller"], id_df[met], color=[PALETTE.get(c, "gray") for c in id_df["controller"]])
        ax.set_title(met.replace("_mean", ""))
        ax.tick_params(axis="x", rotation=25)
    _save(fig, out_path)
