"""RPPO input shape handling tests."""

from __future__ import annotations

import torch

from RPPO_pipeline.RPPO_input import RPPOInput


def _obs(batch_shape):
    return {
        "grid_obs": torch.zeros(*batch_shape, 18, 21, 21),
        "self_obs": torch.zeros(*batch_shape, 27),
        "prev_action_obs": torch.zeros(*batch_shape, 5),
        "tank_type_obs": torch.zeros(*batch_shape, dtype=torch.long),
    }


def test_rppo_input_preserves_recurrent_time_dimension():
    module = RPPOInput()

    features = module(_obs((2, 3)))

    assert features.shape == (2, 3, 312)


def test_rppo_input_keeps_batched_non_recurrent_shape():
    module = RPPOInput()

    features = module(_obs((4,)))

    assert features.shape == (4, 312)
