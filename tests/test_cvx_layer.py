from src.optimization.robust_layer import ScalarRobustLayer


def test_robust_layer_act():
    layer = ScalarRobustLayer(0.08, 0.35, 0.6, 10.0, 100.0)
    out = layer.act(-1.0, 30.0, 0.5)
    assert 0.0 <= out.u <= 10.0
