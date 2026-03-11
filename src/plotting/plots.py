from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_scenario(sc: dict, out_path: str):
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(sc["t"], sc["z"], label="z")
    axs[0].set_ylabel("z")
    axs[1].plot(sc["t"], sc["xi"], label="xi", color="tab:orange")
    axs[1].set_ylabel("xi")
    axs[2].plot(sc["t"], sc["Tamb"], label="Tamb", color="tab:green")
    axs[2].set_ylabel("Tamb")
    axs[2].set_xlabel("t")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mechanism(df, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, g in df.groupby("controller"):
        ax.plot(g["t"], g["rho"], label=name)
    ax.set_title("Radius adaptation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_outcome(df, out_path: str, safety_threshold=0.0):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for name, g in df.groupby("controller"):
        axs[0].plot(g["t"], g["x"], label=name)
        axs[1].plot(g["t"], g["u"], label=name)
    axs[0].axhline(safety_threshold, color="red", linestyle="--")
    axs[0].set_ylabel("x")
    axs[1].set_ylabel("u")
    axs[1].set_xlabel("t")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
