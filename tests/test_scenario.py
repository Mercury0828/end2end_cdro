from src.scenario.calm_burst_calm import ScenarioConfig, generate_trajectory
import numpy as np


def test_calm_burst_context_mask():
    sc = generate_trajectory(ScenarioConfig(), np.random.default_rng(0))
    assert sc["z"][:40].sum() == 0
    assert sc["z"][40:60].sum() == 20
