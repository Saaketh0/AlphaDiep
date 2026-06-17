"""Fresh RLlib PPO training entry point.

Prefer the canonical CLI for day-to-day use:

    python rl/runtime/train.py train

This module remains importable for tests and backward-compatible direct runs.
"""

from __future__ import annotations

import json
import os

import ray

from league_initialization.paths import RLLIB_CHECKPOINT_DIR
from model_store import RedisModelStore
from training_metadata import TrainingRunMetadata, write_rllib_and_league_metadata
from training_runtime import (
    CHECKPOINT_FREQUENCY,
    DIEP_ENV_CONFIG,
    KEEP_PER_TRIAL,
    TRAINING_ITERATIONS,
    TUNE_RUN_NAME,
    build_rllib_config,
    create_tuner,
    ensure_league_available,
    experiment_path,
    register_training_env,
)

TRAINING_ENV_CONFIG = DIEP_ENV_CONFIG


def run_training(
    *,
    env_config: dict | None = None,
    metadata: TrainingRunMetadata | None = None,
    run_name: str = TUNE_RUN_NAME,
    stop_iterations: int = TRAINING_ITERATIONS,
    checkpoint_frequency: int = CHECKPOINT_FREQUENCY,
    num_to_keep: int = KEEP_PER_TRIAL,
    smoke: bool = False,
):
    """Start a fresh PPO run after verifying the Redis/SSD league exists."""
    resolved_env_config = dict(env_config or TRAINING_ENV_CONFIG)
    register_training_env()
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    store = ensure_league_available(RedisModelStore(snapshot_every=0))
    resolved_metadata = metadata or TrainingRunMetadata.create(
        env_config=resolved_env_config,
        latest_league_iteration=store.latest_iteration(),
    )
    rllib_meta, _league_meta = write_rllib_and_league_metadata(
        resolved_metadata,
        experiment_path=experiment_path(run_name),
    )
    os.environ["DIEP_TRAINING_RUN_METADATA"] = str(rllib_meta)
    os.environ["DIEP_EVAL_ENV_CONFIG_JSON"] = json.dumps(resolved_env_config, sort_keys=True)
    os.environ["WANDB_RUN_ID"] = resolved_metadata.run_id

    config_kwargs = {}
    if smoke:
        config_kwargs = {
            "num_env_runners": 0,
            "num_envs_per_env_runner": 1,
            "num_learners": 0,
            "num_gpus_per_learner": 0,
            "num_gpus": 0,
            "rollout_fragment_length": 8,
            "train_batch_size": 32,
            "minibatch_size": 16,
            "num_epochs": 1,
        }
    config = build_rllib_config(env_config=resolved_env_config, **config_kwargs)
    tuner = create_tuner(
        config=config,
        run_name=run_name,
        metadata=resolved_metadata,
        stop_iterations=stop_iterations,
        checkpoint_frequency=checkpoint_frequency,
        num_to_keep=num_to_keep,
        checkpoint_at_end=True,
        enable_wandb=not smoke,
    )
    result_grid = tuner.fit()
    errors = []
    if hasattr(result_grid, "errors"):
        errors = list(result_grid.errors)
    else:
        for result in result_grid:
            error = getattr(result, "error", None)
            if error is not None:
                errors.append(error)
    if errors:
        raise RuntimeError(f"Training failed with {len(errors)} Tune trial error(s): {errors[0]}")
    return result_grid


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
