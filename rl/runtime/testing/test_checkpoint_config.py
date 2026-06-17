"""Smoke tests for RLlib checkpoint and resume configuration."""

from __future__ import annotations

from league_initialization.paths import RLLIB_CHECKPOINT_DIR, TRAINING_DATA_ROOT
from resume_from_checkpoint import (
    CHECKPOINT_FREQUENCY,
    DIEP_ENV_CONFIG,
    KEEP_PER_TRIAL,
    TUNE_RUN_NAME,
    find_experiment_path,
    resolve_experiment_path,
)


def test_rllib_checkpoint_dir_resolves_under_diepcustom_training_data():
    assert RLLIB_CHECKPOINT_DIR == TRAINING_DATA_ROOT / "RLlib"
    assert RLLIB_CHECKPOINT_DIR.name == "RLlib"


def test_checkpoint_frequency_and_retention():
    assert CHECKPOINT_FREQUENCY == 10
    assert KEEP_PER_TRIAL == 10


def test_env_config_uses_fast_reward_path():
    assert DIEP_ENV_CONFIG["include_snapshot_info"] is False
    assert DIEP_ENV_CONFIG["fast_reward_state"] is True


def test_find_experiment_path_requires_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "resume_from_checkpoint.RLLIB_CHECKPOINT_DIR",
        tmp_path,
    )
    try:
        find_experiment_path()
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

    experiment = tmp_path / TUNE_RUN_NAME
    experiment.mkdir()
    assert find_experiment_path() == str(experiment)
    assert resolve_experiment_path(str(experiment)) == str(experiment)
