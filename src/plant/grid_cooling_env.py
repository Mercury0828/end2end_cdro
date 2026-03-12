"""Main showcase plant: chip thermal grid + cold plate + coolant surrogate."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GridCoolingParams:
    nx: int = 8
    ny: int = 8
    dt: float = 1.0
    c_chip: float = 8.0
    c_plate: float = 18.0
    c_coolant: float = 30.0
    k_diff: float = 0.18
    k_chip_plate: float = 0.2
    k_plate_coolant: float = 0.32
    k_ambient: float = 0.02
    pump_gain: float = 0.55
    pump_bias: float = 0.25
    pump_min: float = 0.0
    pump_max: float = 1.0
    pump_slew: float = 0.04
    init_chip_temp: float = 34.0
    init_plate_temp: float = 30.0
    init_coolant_temp: float = 24.0


class GridCoolingEnv:
    def __init__(self, params: GridCoolingParams):
        self.p = params
        self.reset()

    def reset(self):
        self.chip = np.full((self.p.nx, self.p.ny), self.p.init_chip_temp, dtype=float)
        self.t_plate = float(self.p.init_plate_temp)
        self.t_coolant = float(self.p.init_coolant_temp)
        self.u = float(self.p.pump_bias)
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "chip": self.chip.copy(),
            "hotspot": float(self.chip.max()),
            "chip_avg": float(self.chip.mean()),
            "t_plate": float(self.t_plate),
            "t_coolant": float(self.t_coolant),
            "u_prev": float(self.u),
        }

    def _laplacian(self, x: np.ndarray) -> np.ndarray:
        up = np.roll(x, -1, axis=0)
        dn = np.roll(x, 1, axis=0)
        lf = np.roll(x, -1, axis=1)
        rt = np.roll(x, 1, axis=1)
        return up + dn + lf + rt - 4.0 * x

    def step(self, action: float, Tamb: float, Tin: float, q_map: np.ndarray) -> dict:
        p = self.p
        u_target = np.clip(float(action), p.pump_min, p.pump_max)
        du = np.clip(u_target - self.u, -p.pump_slew, p.pump_slew)
        self.u = float(np.clip(self.u + du, p.pump_min, p.pump_max))

        lap = self._laplacian(self.chip)
        diffusion = p.k_diff * lap
        chip_plate = -p.k_chip_plate * (self.chip - self.t_plate)
        ambient = -p.k_ambient * (self.chip - float(Tamb))
        power = np.asarray(q_map, dtype=float)

        dchip = (diffusion + chip_plate + ambient + power) / p.c_chip
        self.chip = self.chip + p.dt * dchip

        heat_from_chip = p.k_chip_plate * (self.chip.mean() - self.t_plate)
        heat_to_coolant = p.k_plate_coolant * (self.t_plate - self.t_coolant) * (0.5 + self.u)
        self.t_plate += p.dt * (heat_from_chip - heat_to_coolant) / p.c_plate

        coolant_mix = (self.t_plate - self.t_coolant) * (p.pump_bias + p.pump_gain * self.u)
        self.t_coolant += p.dt * (coolant_mix - (self.t_coolant - float(Tin))) / p.c_coolant

        return self.get_state()
