"""Lean RLlib checkpoints: exclude ghost RLModules and skip ghost optimizers.

Ghosts are reloaded from the league (Redis/SSD) on init via ``LeagueBootstrapCallback``;
they do not need to be duplicated in Tune checkpoints and never receive gradients.
"""

from __future__ import annotations

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_OPTIMIZER,
    COMPONENT_RL_MODULE,
)
from ray.rllib.core.learner.learner import Learner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics.ray_metrics import TimerAndPrometheusLogger
from ray.tune.trainable import Trainable

from league_initialization.constants import GHOST_POLICIES


def ghost_rl_module_not_components() -> list[str]:
    """RLlib ``not_components`` paths for all ghost policy RLModules."""
    prefix = f"{COMPONENT_LEARNER_GROUP}/{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}"
    return [f"{prefix}/{ghost_id}" for ghost_id in GHOST_POLICIES]


def lean_checkpoint_not_components() -> list[str]:
    """All RLlib components excluded from ``DiepPPO.save_checkpoint``.

    Ghost RLModules are reloaded from the league track on resume. The optimizer
    component is dropped too: Adam reinitializes on restore, trading a brief
    post-resume noise for ~74 MB of disk per checkpoint (4 main optimizers).
    """
    paths = list(ghost_rl_module_not_components())
    paths.append(f"{COMPONENT_LEARNER_GROUP}/{COMPONENT_LEARNER}/{COMPONENT_OPTIMIZER}")
    return paths


class DiepPPOTorchLearner(PPOTorchLearner):
    """PPOTorchLearner that only registers optimizers for trainable modules.

    Mirrors ``Learner.configure_optimizers`` but skips any module whose ID is not in
    ``policies_to_train`` (queried via ``should_module_be_updated``). Ghosts never get
    an Adam state allocated, saving ~310 MB of optimizer RAM for our 16 ghost policies.
    """

    @override(Learner)
    def configure_optimizers(self) -> None:
        for module_id in self.module.keys():
            if not self.should_module_be_updated(module_id):
                continue
            if self.rl_module_is_compatible(self.module[module_id]):
                config = self.config.get_config_for_module(module_id)
                self.configure_optimizers_for_module(module_id=module_id, config=config)


class DiepPPO(PPO):
    """PPO with league-aware checkpoint pruning for ghost policies."""

    @override(Trainable)
    def save_checkpoint(self, checkpoint_dir: str) -> None:
        with TimerAndPrometheusLogger(self._metrics_save_checkpoint_time):
            if not self.config.enable_rl_module_and_learner:
                return super().save_checkpoint(checkpoint_dir)

            self.save_to_path(
                checkpoint_dir,
                state=self.get_state(not_components=lean_checkpoint_not_components()),
                use_msgpack=self.config._use_msgpack_checkpoints,
            )


__all__ = [
    "DiepPPO",
    "DiepPPOTorchLearner",
    "ghost_rl_module_not_components",
    "lean_checkpoint_not_components",
]
