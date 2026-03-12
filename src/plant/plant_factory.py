"""Factory selecting plant backend; retains scalar/FMU compatibility."""
from __future__ import annotations

from pathlib import Path
from .python_env import PlantParams, PythonThermalEnv
from .grid_cooling_env import GridCoolingParams, GridCoolingEnv
from .fmu_env import FMUConfig, FMUThermalEnv


def _make_scalar(plant_cfg: dict):
    return PythonThermalEnv(PlantParams(
        alpha=plant_cfg.get("alpha", 0.08),
        beta=plant_cfg.get("beta", 0.35),
        gamma=plant_cfg.get("gamma", 0.6),
        x0=plant_cfg.get("x0", -1.5),
        dt=plant_cfg.get("dt", 1.0),
    ))


def make_plant(plant_cfg: dict):
    mode = plant_cfg.get("mode", "python")
    plant_kind = plant_cfg.get("plant_kind", "grid")

    if mode == "fmu":
        if Path(plant_cfg.get("fmu_path", "")).exists():
            try:
                return FMUThermalEnv(FMUConfig(fmu_path=plant_cfg["fmu_path"], step_size=plant_cfg.get("dt", 1.0)))
            except Exception:
                pass
        # preserve legacy expectation: FMU fallback defaults to scalar Python plant
        if "plant_kind" not in plant_cfg:
            return _make_scalar(plant_cfg)

    if plant_kind == "scalar":
        return _make_scalar(plant_cfg)

    return GridCoolingEnv(GridCoolingParams(
        nx=plant_cfg.get("nx", 8),
        ny=plant_cfg.get("ny", 8),
        dt=plant_cfg.get("dt", 1.0),
        c_chip=plant_cfg.get("c_chip", 8.0),
        c_plate=plant_cfg.get("c_plate", 18.0),
        c_coolant=plant_cfg.get("c_coolant", 30.0),
        k_diff=plant_cfg.get("k_diff", 0.18),
        k_chip_plate=plant_cfg.get("k_chip_plate", 0.2),
        k_plate_coolant=plant_cfg.get("k_plate_coolant", 0.32),
        k_ambient=plant_cfg.get("k_ambient", 0.02),
        pump_gain=plant_cfg.get("pump_gain", 0.55),
        pump_bias=plant_cfg.get("pump_bias", 0.25),
        pump_min=plant_cfg.get("pump_min", 0.0),
        pump_max=plant_cfg.get("pump_max", 1.0),
        pump_slew=plant_cfg.get("pump_slew", 0.04),
        init_chip_temp=plant_cfg.get("init_chip_temp", 34.0),
        init_plate_temp=plant_cfg.get("init_plate_temp", 30.0),
        init_coolant_temp=plant_cfg.get("init_coolant_temp", 24.0),
    ))
