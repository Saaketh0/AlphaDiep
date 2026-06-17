"""Shared tensor helpers for league weight serialization."""

from __future__ import annotations

import torch


def cpu_state_dict(state_dict: dict) -> dict:
    """Detach every tensor in *state_dict* to CPU for safetensors / Redis storage."""
    return {name: to_cpu_tensor(value) for name, value in state_dict.items()}


def to_cpu_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return torch.as_tensor(value).cpu()
    except ImportError:
        pass
    return value
