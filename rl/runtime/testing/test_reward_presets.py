"""Tests for file-driven RL reward presets."""

from __future__ import annotations

import json

import pytest

from rewards import BASIC_REWARD_CONFIG, DEFAULT_INFO_LOG_AGENTS, load_reward_config, training_env_config
from training_metadata import reward_config_hash, training_env_config_hash


EXPECTED_BASIC_REWARD_CONFIG = {
    "raw": 0.0,
    "score_delta": 1.0,
    "health_delta": 0.0,
    "damage_taken": -0.01,
    "enemy_kills": 2.0,
    "farm_kills": 0.05,
    "level_delta": 0.02,
    "level_milestone": 0.5,
    "edge_proximity": -0.01,
    "movement_speed": 0.005,
    "retreat": 0.03,
    "aim_accuracy": 0.05,
    "enemy_damage_dealt": 0.02,
    "alive": 0.0,
    "death": -1.0,
    "truncation": 0.0,
    "step": -0.001,
}


def test_preset_name_resolves_to_basic_defaults():
    assert load_reward_config("basic") == EXPECTED_BASIC_REWARD_CONFIG
    assert BASIC_REWARD_CONFIG == EXPECTED_BASIC_REWARD_CONFIG


def test_explicit_json_path_resolves_correctly(tmp_path):
    preset = tmp_path / "aggressive.json"
    preset.write_text(json.dumps({"score_delta": 3, "death": -2, "step": -0.01}), encoding="utf-8")

    loaded = load_reward_config(preset)

    assert loaded["score_delta"] == 3.0
    assert loaded["death"] == -2.0
    assert loaded["step"] == -0.01
    assert loaded["raw"] == 0.0


def test_unknown_reward_field_fails_clearly(tmp_path):
    preset = tmp_path / "bad.json"
    preset.write_text(json.dumps({"score_delta": 1.0, "distance_to_center": 0.5}), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown reward config fields"):
        load_reward_config(preset)


def test_reward_value_change_is_tracked_without_changing_structural_env_hash(tmp_path):
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps({**EXPECTED_BASIC_REWARD_CONFIG, "death": -2.0}), encoding="utf-8")

    basic_env = training_env_config(reward_config=load_reward_config("basic"))
    tuned_env = training_env_config(reward_config=load_reward_config(tuned))

    assert training_env_config_hash(basic_env) == training_env_config_hash(tuned_env)
    assert reward_config_hash(basic_env) != reward_config_hash(tuned_env)


def test_training_env_config_logs_reward_components_for_all_four_main_agents_by_default():
    assert training_env_config()["info_log_agents"] == DEFAULT_INFO_LOG_AGENTS


def test_train_parser_exposes_reward_config_on_smoke_help(capsys):
    from train import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["smoke", "--help"])

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "--reward-config" in captured.out


def test_resume_command_passes_selected_reward_config(monkeypatch, tmp_path):
    import argparse
    import train

    preset = tmp_path / "resume_reward.json"
    preset.write_text(json.dumps({"score_delta": 4.0, "death": -3.0}), encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(train, "start_redis", lambda action: 0)

    def fake_resume_training(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(train, "resume_training", fake_resume_training)

    assert train.command_resume(
        argparse.Namespace(
            resume_path="training_data/RLlib/rl_run",
            allow_unsafe_resume=False,
            reward_config=str(preset),
        )
    ) == 0

    env_config = captured["env_config"]
    assert env_config["reward_config"]["score_delta"] == 4.0
    assert env_config["reward_config"]["death"] == -3.0


def test_smoke_command_reaches_store_validation_without_name_error(monkeypatch, tmp_path):
    import argparse
    import train

    events: list[str] = []

    monkeypatch.setattr(train, "start_redis", lambda action: events.append(f"redis:{action}") or 0)
    monkeypatch.setattr(train, "build_cpp_headless", lambda: events.append("build"))
    monkeypatch.setattr(train, "seed_league_cache", lambda **kwargs: events.append("seed") or {})
    monkeypatch.setattr(train, "run_training", lambda **kwargs: events.append("train") or None)
    monkeypatch.setattr(train, "validate_resume_metadata", lambda **kwargs: events.append("validate") or None)
    monkeypatch.setattr(train, "experiment_path", lambda name: f"/tmp/{name}")

    class FakeStore:
        classes = ("A", "B", "C", "D")

        def __init__(self, snapshot_every=0):
            assert snapshot_every == 0

        def latest_iteration(self):
            return 2

    monkeypatch.setattr(train, "RedisModelStore", FakeStore)
    for char_class in FakeStore.classes:
        (tmp_path / char_class).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(train, "LEAGUE_EXPORT_DIR", tmp_path)

    assert train.command_smoke(
        argparse.Namespace(
            max_ticks=64,
            seed_count=2,
            reward_config="basic",
        )
    ) == 0
    assert events == ["redis:start", "build", "seed", "train", "validate"]
