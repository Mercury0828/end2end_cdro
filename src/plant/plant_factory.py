"""Factory selecting FMU-backed plant when possible, fallback to Python plant."""
from __future__ import annotations

from pathlib import Path
from .python_env import PlantParams, PythonThermalEnv
from .fmu_env import FMUConfig, FMUThermalEnv


def make_plant(plant_cfg: dict):
    mode = plant_cfg.get("mode", "python")
    if mode == "fmu" and Path(plant_cfg.get("fmu_path", "")).exists():
        try:
            return FMUThermalEnv(FMUConfig(fmu_path=plant_cfg["fmu_path"], step_size=plant_cfg.get("dt", 1.0)))
        except Exception:
            pass
    return PythonThermalEnv(PlantParams(
        alpha=plant_cfg.get("alpha", 0.08),
        beta=plant_cfg.get("beta", 0.35),
        gamma=plant_cfg.get("gamma", 0.6),
        x0=plant_cfg.get("x0", -1.5),
        dt=plant_cfg.get("dt", 1.0),
    ))
