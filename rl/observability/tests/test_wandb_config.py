"""Tests for W&B offline-first observability defaults."""

from __future__ import annotations

import importlib

import pytest

from rl.observability.config import ObservabilityConfig
from rl.observability.logging.wandb_tune import wandb_logger_kwargs


def _expected_display_name(run_id: str) -> str:
    from datetime import datetime

    return f"run-{run_id}-{datetime.now().strftime('%m-%d')}"


# Verifies W&B logging defaults are local/offline and checkpoint-safe.
def test_wandb_logger_kwargs_default_offline():
    config = ObservabilityConfig(run_id="test-run")
    kwargs = wandb_logger_kwargs(config)
    assert kwargs["project"] == "diepcustom-headless-rl"
    assert kwargs["group"] == "ppo-training"
    assert kwargs["mode"] == "offline"
    assert kwargs["name"] == _expected_display_name("test-run")
    assert kwargs["id"] == "test-run"
    assert kwargs["resume"] == "allow"
    assert "run_id:test-run" in kwargs["tags"]
    assert kwargs["log_config"] is True
    assert kwargs["upload_checkpoints"] is False


# Verifies environment overrides used by training are honored.
def test_observability_config_from_env(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("WANDB_TAGS", "alpha,beta")
    monkeypatch.setenv("DIEP_VIDEO_INTERVAL", "1")
    monkeypatch.setenv("DIEP_VIDEO_FPS", "12")
    config = ObservabilityConfig.from_env(run_id="env-run")
    assert config.wandb_mode == "disabled"
    assert config.wandb_resume == "must"
    assert config.wandb_tags == ("alpha", "beta")
    assert config.video_agent == "agent_1"
    assert config.video_interval_iterations == 1
    assert config.video_fps == 12


# Eval video is off by default during training; opt-in via DIEP_VIDEO_ENABLED.
def test_video_disabled_by_default_from_env(monkeypatch):
    monkeypatch.delenv("DIEP_VIDEO_ENABLED", raising=False)
    config = ObservabilityConfig.from_env(run_id="video-default")
    assert config.video_enabled is False


def test_video_enabled_when_env_var_set(monkeypatch):
    monkeypatch.setenv("DIEP_VIDEO_ENABLED", "true")
    config = ObservabilityConfig.from_env(run_id="video-on")
    assert config.video_enabled is True


# Explicit constructor opt-in still works (e.g. eval tooling, write_eval_video).
def test_video_enabled_default_via_direct_construction():
    config = ObservabilityConfig(run_id="direct")
    assert config.video_enabled is True


def test_default_observability_paths_live_under_training_data_wandb():
    config = ObservabilityConfig(run_id="path-test")
    assert config.runs_root == config.runs_root.parents[0] / "W&B"
    assert config.runs_root.name == "W&B"
    assert config.runs_root.parent.name == "training_data"
    assert config.run_dir == config.runs_root / "path-test"
    assert config.eval_iteration_dir(500) == config.run_dir / "eval" / "500"


def test_wandb_logger_kwargs_store_offline_runs_under_wandb_root():
    config = ObservabilityConfig(run_id="wandb-dir-test")
    kwargs = wandb_logger_kwargs(config)
    assert kwargs["name"] == _expected_display_name("wandb-dir-test")
    assert kwargs["id"] == "wandb-dir-test"
    assert kwargs["resume"] == "allow"
    assert kwargs["dir"] == str(config.runs_root)
    assert config.upload_checkpoints is False


def test_wandb_logger_kwargs_use_metadata_run_id_for_identity_and_group():
    config = ObservabilityConfig(
        run_id="metadata-run",
        wandb_group="rl_run",
        wandb_tags=("extra", "run_id:metadata-run"),
    )
    kwargs = wandb_logger_kwargs(config)
    assert kwargs["name"] == _expected_display_name("metadata-run")
    assert kwargs["id"] == "metadata-run"
    assert kwargs["group"] == "rl_run"
    assert kwargs["tags"].count("run_id:metadata-run") == 1
    assert "extra" in kwargs["tags"]


def test_legacy_custom_wandb_modules_are_removed():
    for module in (
        "observability.logging.wandb_logger",
        "observability.logging.diep_metrics_callback",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
