import numpy as np

from amrl_fast.statistics import discrete_mutual_information, paired_effect_size


def test_discrete_mi_handles_joint_controller_actions():
    outputs = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.8, 0.7, 0.9], [0.7, 0.8, 0.8]]
    deltas = [0.1, 0.2, 1.0, 0.9]
    assert discrete_mutual_information(outputs, deltas) >= 0.0


def test_paired_statistics_use_seed_aggregates():
    p_value, effect = paired_effect_size([0.2, 0.3, 0.4], [0.25, 0.35, 0.45])
    assert np.isfinite(p_value)
    assert np.isfinite(effect)
