"""Canonical build/config/runtime helpers for Diep RLlib training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import CheckpointConfig, RunConfig
from ray.tune.registry import register_env

from DiepModelConfig import DiepCatalog, DiepConfig, DiepPolicy
from lean_checkpoint import DiepPPO, DiepPPOTorchLearner
from league_initialization import GHOST_POLICIES, MAIN_POLICIES, LeagueBootstrapCallback, policy_id_for_agent
from league_initialization.disk_store import hydrate_redis_from_disk
from league_initialization.paths import DIEPCUSTOM_ROOT, RLLIB_CHECKPOINT_DIR
from model_store import RedisModelStore
from rl.observability.config import ObservabilityConfig
from rl.observability.logging.rllib_callbacks import DiepRLlibObservabilityCallback
from rl.observability.logging.wandb_tune import create_wandb_logger_callback
from rewards import training_env_config
from training_metadata import TrainingRunMetadata

ENV_NAME = "diepcustom_headless"
TUNE_RUN_NAME = "rl_run"
CHECKPOINT_FREQUENCY = 10
KEEP_PER_TRIAL = 10
TRAINING_ITERATIONS = 2000
DIEP_ENV_CONFIG = training_env_config()
EVAL_ENV_CONFIG_ENV_VAR = "DIEP_EVAL_ENV_CONFIG_JSON"


def _eval_env_config_from_env(default: dict[str, Any]) -> dict[str, Any]:
    raw = os.environ.get(EVAL_ENV_CONFIG_ENV_VAR, "")
    if not raw:
        return dict(default)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)



class TrainingCallbacks(LeagueBootstrapCallback, DiepRLlibObservabilityCallback):
    """Combine ghost-league maintenance with lightweight Diep observability."""

    def __init__(self):
        LeagueBootstrapCallback.__init__(self)
        DiepRLlibObservabilityCallback.__init__(
            self,
            config=ObservabilityConfig.from_env(eval_env_config=_eval_env_config_from_env(DIEP_ENV_CONFIG)),
        )

    def on_train_result(self, *, algorithm, result=None, **kwargs):
        LeagueBootstrapCallback.on_train_result(self, algorithm=algorithm, result=result, **kwargs)
        DiepRLlibObservabilityCallback.on_train_result(self, algorithm=algorithm, result=result, **kwargs)

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return policy_id_for_agent(agent_id)


def register_training_env() -> None:
    from rl.env import DiepCustomParallelEnv

    register_env(ENV_NAME, lambda cfg: ParallelPettingZooEnv(DiepCustomParallelEnv(**cfg)))


def build_rllib_config(
    *,
    env_config: dict[str, Any] | None = None,
    num_env_runners: int | None = None,
    num_envs_per_env_runner: int | None = None,
    num_learners: int | None = None,
    num_gpus_per_learner: int | None = None,
    num_gpus: int | None = None,
    rollout_fragment_length: int | str | None = None,
    train_batch_size: int | None = None,
    minibatch_size: int | None = None,
    num_epochs: int | None = None,
) -> PPOConfig:
    """Build the canonical PPO config; small overrides keep smoke tests cheap."""
    from resource_compute import compute_resource, get_num_envs_per_env_runner

    resolved_env_config = dict(env_config or DIEP_ENV_CONFIG)
    compute_resources = compute_resource()
    resolved_num_env_runners = compute_resources[0] if num_env_runners is None else num_env_runners
    resolved_num_envs = (
        get_num_envs_per_env_runner(compute_resources)
        if num_envs_per_env_runner is None
        else num_envs_per_env_runner
    )
    resolved_num_learners = compute_resources[1] if num_learners is None else num_learners
    resolved_num_gpus_per_learner = compute_resources[2] if num_gpus_per_learner is None else num_gpus_per_learner
    resolved_num_gpus = compute_resources[3] if num_gpus is None else num_gpus

    spec = RLModuleSpec(
        module_class=DiepPolicy,
        model_config=DiepConfig,
        catalog_class=DiepCatalog,
    )
    config = (
        PPOConfig()
        .environment(ENV_NAME, env_config=resolved_env_config)
        .framework(framework="torch")
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
            policies=set(MAIN_POLICIES + GHOST_POLICIES),
            policies_to_train=MAIN_POLICIES,
        )
        .env_runners(
            num_env_runners=resolved_num_env_runners,
            num_cpus_per_env_runner=1,
            num_envs_per_env_runner=resolved_num_envs,
            rollout_fragment_length=rollout_fragment_length,
        )
        .learners(
            num_learners=resolved_num_learners,
            num_gpus_per_learner=resolved_num_gpus_per_learner,
            learner_class=DiepPPOTorchLearner,
        )
        .resources(num_gpus=resolved_num_gpus)
        .rl_module(rl_module_spec=spec)
        .callbacks(TrainingCallbacks)
    )
    training_kwargs = {
        key: value
        for key, value in {
            "train_batch_size": train_batch_size,
            "minibatch_size": minibatch_size,
            "num_epochs": num_epochs,
        }.items()
        if value is not None
    }
    if training_kwargs:
        config = config.training(**training_kwargs)
    return config


def create_tuner(
    *,
    config: PPOConfig,
    run_name: str = TUNE_RUN_NAME,
    metadata: TrainingRunMetadata | None = None,
    stop_iterations: int = TRAINING_ITERATIONS,
    checkpoint_frequency: int = CHECKPOINT_FREQUENCY,
    num_to_keep: int = KEEP_PER_TRIAL,
    checkpoint_at_end: bool = True,
    enable_wandb: bool = True,
):
    callbacks = []
    if enable_wandb:
        obs_overrides: dict[str, Any] = {"wandb_group": run_name, "eval_env_config": config.env_config}
        if metadata is not None:
            obs_overrides.update(
                {
                    "run_id": metadata.run_id,
                    "wandb_tags": (f"run_id:{metadata.run_id}",),
                }
            )
        obs_config = ObservabilityConfig.from_env(**obs_overrides)
        callbacks.append(create_wandb_logger_callback(obs_config))
    return tune.Tuner(
        DiepPPO,
        param_space=config,
        run_config=RunConfig(
            name=run_name,
            storage_path=str(RLLIB_CHECKPOINT_DIR),
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=checkpoint_frequency,
                num_to_keep=num_to_keep,
                checkpoint_at_end=checkpoint_at_end,
            ),
            stop={"training_iteration": stop_iterations},
            callbacks=callbacks,
        ),
    )


def experiment_path(run_name: str = TUNE_RUN_NAME) -> Path:
    return RLLIB_CHECKPOINT_DIR / run_name


def start_redis(action: str = "start") -> int:
    script = Path(__file__).resolve().parent / "start_redis.sh"
    return subprocess.run([str(script), action], check=True).returncode


def ensure_league_available(store: RedisModelStore | None = None) -> RedisModelStore:
    league_store = store or RedisModelStore(snapshot_every=0)
    if not league_store.has_league_keys():
        hydrate_redis_from_disk(league_store)
    if not league_store.has_league_keys():
        raise RuntimeError("League is empty; run `python rl/runtime/train.py seed` first.")
    league_store.warm_iteration_cache()
    return league_store


def build_cpp_headless() -> None:
    subprocess.run(["npm", "run", "test:cpp"], cwd=DIEPCUSTOM_ROOT / "ts-server", check=True)


def ensure_repo_on_path() -> None:
    root = str(DIEPCUSTOM_ROOT)
    testing = str(DIEPCUSTOM_ROOT / "rl" / "runtime")
    for path in (root, testing):
        if path not in sys.path:
            sys.path.insert(0, path)


__all__ = [
    "CHECKPOINT_FREQUENCY",
    "DIEP_ENV_CONFIG",
    "ENV_NAME",
    "KEEP_PER_TRIAL",
    "TRAINING_ITERATIONS",
    "TUNE_RUN_NAME",
    "build_cpp_headless",
    "build_rllib_config",
    "create_tuner",
    "ensure_league_available",
    "experiment_path",
    "register_training_env",
    "start_redis",
]
