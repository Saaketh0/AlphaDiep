"""Resolve and safely resume Tune experiments for Diep training."""

from __future__ import annotations

import json
from pathlib import Path
import os

from league_initialization.paths import RLLIB_CHECKPOINT_DIR
from rewards import training_env_config
from training_metadata import validate_resume_metadata
from training_runtime import (
    CHECKPOINT_FREQUENCY,
    DIEP_ENV_CONFIG,
    KEEP_PER_TRIAL,
    TRAINING_ITERATIONS,
    TUNE_RUN_NAME,
    ensure_league_available,
    register_training_env,
)


def find_experiment_path() -> str:
    """Return the Tune experiment directory for ``Tuner.restore``."""
    path = RLLIB_CHECKPOINT_DIR / TUNE_RUN_NAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"No prior Tune experiment at {path}. "
            "Start a fresh run first, or pass --resume-path explicitly."
        )
    return str(path)


def resolve_experiment_path(explicit: str | None = None) -> str:
    """Use an explicit experiment path or discover the default ``rl_run`` dir."""
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"Resume path does not exist: {path}")
        return str(path)
    return find_experiment_path()


def validate_resume(
    *,
    resume_path: str | None = None,
    env_config: dict | None = None,
    allow_unsafe_resume: bool = False,
):
    experiment = resolve_experiment_path(resume_path)
    resolved_env_config = dict(env_config or DIEP_ENV_CONFIG)
    ensure_league_available()
    return validate_resume_metadata(
        experiment_path=experiment,
        env_config=resolved_env_config,
        allow_unsafe_resume=allow_unsafe_resume,
    )


def resume_training(
    *,
    resume_path: str | None = None,
    env_config: dict | None = None,
    allow_unsafe_resume: bool = False,
):
    """Resume a prior Tune PPO run after fail-closed provenance validation."""
    import ray
    from ray import tune

    from lean_checkpoint import DiepPPO

    experiment = resolve_experiment_path(resume_path)
    resolved_env_config = dict(env_config or DIEP_ENV_CONFIG)
    metadata = validate_resume(
        resume_path=experiment,
        env_config=resolved_env_config,
        allow_unsafe_resume=allow_unsafe_resume,
    )
    if metadata is not None:
        metadata_path = Path(experiment) / "run_metadata.json"
        os.environ["DIEP_TRAINING_RUN_METADATA"] = str(metadata_path)
        os.environ["DIEP_EVAL_ENV_CONFIG_JSON"] = json.dumps(resolved_env_config, sort_keys=True)
        os.environ["WANDB_RUN_ID"] = metadata.run_id
        os.environ["WANDB_RESUME"] = "must"
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    register_training_env()
    tuner = tune.Tuner.restore(experiment, trainable=DiepPPO)
    return tuner.fit()


def main() -> None:
    """CLI-compatible resume entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Resume Diep RLlib PPO training")
    parser.add_argument("--resume-path", default=None, help="Optional Tune experiment directory")
    parser.add_argument(
        "--allow-unsafe-resume",
        action="store_true",
        help="Development-only escape hatch: bypass provenance checks with a warning.",
    )
    args = parser.parse_args()
    resume_training(resume_path=args.resume_path, allow_unsafe_resume=args.allow_unsafe_resume)


if __name__ == "__main__":
    main()
