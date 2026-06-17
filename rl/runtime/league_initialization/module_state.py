"""RLModule weight helpers for league bootstrap and ghost loading."""

from __future__ import annotations

from .tensor_utils import cpu_state_dict


def get_module_state(algorithm, module_id: str) -> dict:
    """Return a CPU state dict for a policy RLModule."""
    module = algorithm.get_module(module_id)
    if hasattr(module, "get_state"):
        state = module.get_state()
        return cpu_state_dict(state)
    return cpu_state_dict(module.state_dict())


def set_module_state(algorithm, module_id: str, state_dict: dict) -> None:
    """Load a state dict into a policy RLModule."""
    module = algorithm.get_module(module_id)
    if hasattr(module, "set_state"):
        module.set_state(state_dict)
        return
    module.load_state_dict(state_dict)


def sync_module_weights(algorithm) -> None:
    """Push learner module weights to env runners."""
    env_runner_group = getattr(algorithm, "env_runner_group", None)
    if env_runner_group is not None:
        env_runner_group.sync_weights()
        return

    workers = getattr(algorithm, "workers", None)
    if workers is not None:
        workers.sync_weights()

