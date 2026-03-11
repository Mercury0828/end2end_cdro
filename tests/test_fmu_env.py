from pathlib import Path
import pytest
from src.plant.plant_factory import make_plant


def test_factory_fallback_without_fmu():
    env = make_plant({"mode": "fmu", "fmu_path": "artifacts/missing.fmu"})
    assert env.__class__.__name__ == "PythonThermalEnv"

@pytest.mark.skipif(not Path('artifacts/SingleZoneThermal.fmu').exists(), reason='FMU not exported in CI/local env')
def test_fmu_if_available():
    env = make_plant({"mode": "fmu", "fmu_path": "artifacts/SingleZoneThermal.fmu", "dt": 1.0})
    env.reset()
    x = env.step(1.0, 30.0, 0.2)
    assert x is not None
