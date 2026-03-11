"""FMPy-backed runtime wrapper with Python-environment-compatible interface."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FMUConfig:
    fmu_path: str
    stop_time: float = 100.0
    step_size: float = 1.0


class FMUThermalEnv:
    def __init__(self, cfg: FMUConfig):
        try:
            from fmpy import read_model_description, extract
            from fmpy.fmi2 import FMU2Slave
        except Exception as exc:
            raise RuntimeError("FMPy unavailable. Install fmpy to use FMU mode.") from exc

        self._extract = extract
        self._read_model_description = read_model_description
        self._FMU2Slave = FMU2Slave
        self.cfg = cfg
        self.time = 0.0
        self._init_fmu()

    def _init_fmu(self):
        model_description = self._read_model_description(self.cfg.fmu_path)
        unzipdir = self._extract(self.cfg.fmu_path)
        self.model_variables = {v.name: v for v in model_description.modelVariables}
        self.fmu = self._FMU2Slave(
            guid=model_description.guid,
            unzipDirectory=unzipdir,
            modelIdentifier=model_description.coSimulation.modelIdentifier,
            instanceName='inst1',
        )
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def _vr(self, name: str) -> int:
        return self.model_variables[name].valueReference

    def reset(self, x0: float | None = None):
        self.fmu.reset()
        self.time = 0.0
        self.fmu.setupExperiment(startTime=0.0)
        self.fmu.enterInitializationMode()
        if x0 is not None and "x" in self.model_variables:
            self.fmu.setReal([self._vr("x")], [float(x0)])
        self.fmu.exitInitializationMode()
        return self.get_state()

    def step(self, action: float, Tamb: float, xi: float, eps: float = 0.0):
        if "u" in self.model_variables:
            self.fmu.setReal([self._vr("u")], [float(action)])
        if "Tamb" in self.model_variables:
            self.fmu.setReal([self._vr("Tamb")], [float(Tamb)])
        if "xi" in self.model_variables:
            self.fmu.setReal([self._vr("xi")], [float(xi)])
        self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.cfg.step_size)
        self.time += self.cfg.step_size
        return self.get_state()

    def get_state(self):
        return self.fmu.getReal([self._vr("x")])[0]
