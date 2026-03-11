from src.plant.python_env import PythonThermalEnv, PlantParams


def test_python_env_step_runs():
    env = PythonThermalEnv(PlantParams())
    env.reset()
    x = env.step(1.0, 30.0, 0.2)
    assert isinstance(x, float)
