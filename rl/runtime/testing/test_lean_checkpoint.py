"""Tests for lean RLlib checkpoint pruning and main-only optimizer registration."""

from __future__ import annotations

from league_initialization.constants import GHOST_POLICIES, MAIN_POLICIES
from lean_checkpoint import (
    DiepPPOTorchLearner,
    ghost_rl_module_not_components,
    lean_checkpoint_not_components,
)
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_OPTIMIZER,
    COMPONENT_RL_MODULE,
)


def test_ghost_rl_module_not_components_covers_all_ghosts():
    paths = ghost_rl_module_not_components()
    assert len(paths) == len(GHOST_POLICIES)
    for ghost_id in GHOST_POLICIES:
        expected = (
            f"{COMPONENT_LEARNER_GROUP}/{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}/{ghost_id}"
        )
        assert expected in paths


def test_lean_checkpoint_excludes_optimizer_and_ghosts():
    """``DiepPPO.save_checkpoint`` must skip ghost modules *and* the optimizer state."""
    paths = lean_checkpoint_not_components()
    optimizer_path = f"{COMPONENT_LEARNER_GROUP}/{COMPONENT_LEARNER}/{COMPONENT_OPTIMIZER}"

    assert optimizer_path in paths
    for ghost_id in GHOST_POLICIES:
        ghost_path = (
            f"{COMPONENT_LEARNER_GROUP}/{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}/{ghost_id}"
        )
        assert ghost_path in paths
    # Mains stay in the checkpoint so resume can restore weights.
    for main_id in MAIN_POLICIES:
        main_path = (
            f"{COMPONENT_LEARNER_GROUP}/{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}/{main_id}"
        )
        assert main_path not in paths


class _NoopTimer:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_save_checkpoint_passes_lean_not_components_to_get_state(tmp_path, monkeypatch):
    """End-to-end: ``DiepPPO.save_checkpoint`` threads ``lean_checkpoint_not_components`` into ``get_state``."""
    from lean_checkpoint import DiepPPO

    captured: dict[str, object] = {}

    class _FakeConfig:
        enable_rl_module_and_learner = True
        _use_msgpack_checkpoints = False

    algo = DiepPPO.__new__(DiepPPO)
    algo.config = _FakeConfig()
    algo._metrics_save_checkpoint_time = "noop_metric"

    def fake_get_state(*, not_components):
        captured["not_components"] = list(not_components)
        return {"sentinel": True}

    def fake_save_to_path(checkpoint_dir, *, state, use_msgpack):
        captured["state"] = state
        captured["dir"] = str(checkpoint_dir)
        captured["use_msgpack"] = use_msgpack

    algo.get_state = fake_get_state
    algo.save_to_path = fake_save_to_path
    monkeypatch.setattr(
        "lean_checkpoint.TimerAndPrometheusLogger",
        lambda *a, **k: _NoopTimer(),
    )

    DiepPPO.save_checkpoint(algo, str(tmp_path))

    assert captured["not_components"] == lean_checkpoint_not_components()
    assert captured["dir"] == str(tmp_path)
    assert captured["use_msgpack"] is False


class _StubModule:
    def __init__(self, module_ids):
        self._module_ids = list(module_ids)

    def keys(self):
        return list(self._module_ids)

    def __getitem__(self, module_id):
        # Returned sentinel only needs to be passed to rl_module_is_compatible; the
        # learner stub treats anything as compatible.
        return module_id


class _StubConfig:
    def __init__(self, policies_to_train):
        self.policies_to_train = list(policies_to_train)

    def get_config_for_module(self, module_id):
        return self


def test_configure_optimizers_only_for_main_policies():
    """DiepPPOTorchLearner skips ghost modules when registering optimizers."""

    learner = DiepPPOTorchLearner.__new__(DiepPPOTorchLearner)
    learner._module = _StubModule(MAIN_POLICIES + GHOST_POLICIES)
    learner.config = _StubConfig(MAIN_POLICIES)

    configured: list[str] = []

    def fake_rl_module_is_compatible(_module):
        return True

    def fake_configure_optimizers_for_module(*, module_id, config):
        configured.append(module_id)

    learner.rl_module_is_compatible = fake_rl_module_is_compatible
    learner.configure_optimizers_for_module = fake_configure_optimizers_for_module

    DiepPPOTorchLearner.configure_optimizers(learner)

    assert configured == list(MAIN_POLICIES)
    assert not any(module_id in GHOST_POLICIES for module_id in configured)
