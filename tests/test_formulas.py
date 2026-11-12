from multi_context_memory.formulas import compute_boundary_pressure, compute_recall_trigger, compute_write_pressure, decay_value, feature_similarity


def test_pressure_formulas_follow_weighted_ordering() -> None:
    weights = {"a": 0.5, "b": 0.5}
    low = {"a": 0.2, "b": 0.2}
    high = {"a": 0.8, "b": 0.9}
    assert compute_boundary_pressure(high, weights) > compute_boundary_pressure(low, weights)
    assert compute_write_pressure(high, weights) > compute_write_pressure(low, weights)
    assert compute_recall_trigger(high, weights) > compute_recall_trigger(low, weights)


def test_feature_similarity_prefers_overlap() -> None:
    close = feature_similarity({"bang": 1.0, "metallic": 0.8}, {"bang": 0.9, "metallic": 0.7})
    far = feature_similarity({"bang": 1.0, "metallic": 0.8}, {"water": 1.0, "soft": 0.4})
    assert close > far


def test_decay_value_is_gradual() -> None:
    first = decay_value(1.0, 0.1, 1.0)
    second = decay_value(first, 0.1, 1.0)
    assert 0.0 < second < first < 1.0
