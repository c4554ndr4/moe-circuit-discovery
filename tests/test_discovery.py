import numpy as np
import pytest
import torch

from moe_circuits.discovery import ActivationStats, matched_random


def test_response_mask_route_weight_and_equal_example_weight():
    stats = ActivationStats((1, 2, 3))
    stats.begin(1, 3)
    stats.observe(0, 0, torch.tensor([0, 1, 2]),
                  torch.tensor([[999., 999., 999.], [4., -8., 0.], [4., -8., 0.]]),
                  torch.tensor([[1.], [.5], [.5]]))
    stats.finish(1)
    stats.begin(0, 1)
    stats.observe(0, 0, torch.tensor([0]), torch.tensor([[2., -4., 0.]]), torch.ones(1, 1))
    stats.finish(1)
    stats.begin(0, 1)
    stats.observe(0, 0, torch.tensor([0]), torch.zeros(1, 3), torch.ones(1, 1))
    stats.finish(0)
    result = stats.rank(2)
    assert result[0]["neuron"] == 1 and result[0]["signed_delta"] == -4
    assert result[1]["signed_delta"] == 2
    assert stats.rank(2, mode="positive")[0]["neuron"] == 0
    assert stats.counts == [1, 2]
    np.testing.assert_array_equal(stats.positive_example_hits[0, 0], [2, 2, 2])


def test_missing_capture_fails():
    stats = ActivationStats((2, 2, 3))
    stats.begin(0, 1)
    with pytest.raises(ValueError, match="Missing"):
        stats.finish(1)


def test_random_control_matches_experts_excludes_targets():
    targets = [{"layer": 0, "expert": 1, "neuron": n} for n in (1, 2)]
    control = matched_random(targets, (2, 3, 8), 4)
    assert len(control) == 2
    assert all(row["layer"] == 0 and row["expert"] == 1 and row["neuron"] not in (1, 2) for row in control)
    assert control == matched_random(targets, (2, 3, 8), 4)
