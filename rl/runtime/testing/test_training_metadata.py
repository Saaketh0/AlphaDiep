"""Tests for fail-closed training resume provenance."""

from __future__ import annotations

import json

from training_metadata import (
    REWARD_CONFIG_WARNING,
    ResumeProvenanceError,
    TrainingRunMetadata,
    UNSAFE_RESUME_WARNING,
    league_metadata_path,
    policy_layout_hash,
    read_metadata,
    rllib_metadata_path,
    training_env_config_hash,
    validate_resume_metadata,
    write_metadata,
    write_rllib_and_league_metadata,
)


def _env_config():
    return {"agents": 20, "max_ticks": 64, "nested": {"b": 2, "a": 1}}


def _reward_env_config():
    return {**_env_config(), "reward_config": {"score_delta": 1.0, "death": -1.0}}


def test_metadata_hashing_is_stable_for_key_order():
    left = {"b": 2, "a": {"d": 4, "c": 3}}
    right = {"a": {"c": 3, "d": 4}, "b": 2}

    assert training_env_config_hash(left) == training_env_config_hash(right)
    assert len(policy_layout_hash()) == 64


def test_missing_league_metadata_fails_resume(tmp_path):
    metadata = TrainingRunMetadata.create(env_config=_env_config(), latest_league_iteration=1)
    experiment = tmp_path / "RLlib" / "rl_run"
    write_metadata(rllib_metadata_path(experiment), metadata)

    try:
        validate_resume_metadata(experiment_path=experiment, league_dir=tmp_path / "redis", env_config=_env_config())
    except ResumeProvenanceError as exc:
        assert "Missing provenance metadata" in str(exc)
    else:
        raise AssertionError("resume validation should fail without league metadata")


def test_mismatched_run_metadata_fails_resume(tmp_path):
    env_config = _env_config()
    rllib = TrainingRunMetadata.create(env_config=env_config, latest_league_iteration=1, run_id="run-a", league_id="league-a")
    league = TrainingRunMetadata.create(env_config=env_config, latest_league_iteration=1, run_id="run-b", league_id="league-a")
    experiment = tmp_path / "RLlib" / "rl_run"
    league_dir = tmp_path / "redis"
    write_metadata(rllib_metadata_path(experiment), rllib)
    write_metadata(league_metadata_path(league_dir), league)

    try:
        validate_resume_metadata(experiment_path=experiment, league_dir=league_dir, env_config=env_config)
    except ResumeProvenanceError as exc:
        assert "run_id" in str(exc)
    else:
        raise AssertionError("resume validation should fail on run mismatch")


def test_mismatched_config_fails_resume(tmp_path):
    env_config = _env_config()
    metadata = TrainingRunMetadata.create(env_config=env_config, latest_league_iteration=1)
    experiment = tmp_path / "RLlib" / "rl_run"
    league_dir = tmp_path / "redis"
    write_rllib_and_league_metadata(metadata, experiment_path=experiment, league_dir=league_dir)

    changed = dict(env_config)
    changed["max_ticks"] = 65
    try:
        validate_resume_metadata(experiment_path=experiment, league_dir=league_dir, env_config=changed)
    except ResumeProvenanceError as exc:
        assert "training_env_config_hash" in str(exc)
    else:
        raise AssertionError("resume validation should fail on config mismatch")


def test_reward_value_change_warns_but_allows_resume(tmp_path, capsys):
    env_config = _reward_env_config()
    metadata = TrainingRunMetadata.create(env_config=env_config, latest_league_iteration=1)
    experiment = tmp_path / "RLlib" / "rl_run"
    league_dir = tmp_path / "redis"
    write_rllib_and_league_metadata(metadata, experiment_path=experiment, league_dir=league_dir)

    changed = {**env_config, "reward_config": {**env_config["reward_config"], "death": -2.0}}

    assert validate_resume_metadata(experiment_path=experiment, league_dir=league_dir, env_config=changed) == metadata
    captured = capsys.readouterr()
    assert REWARD_CONFIG_WARNING in captured.out


def test_reward_field_change_fails_resume(tmp_path):
    env_config = _reward_env_config()
    metadata = TrainingRunMetadata.create(env_config=env_config, latest_league_iteration=1)
    experiment = tmp_path / "RLlib" / "rl_run"
    league_dir = tmp_path / "redis"
    write_rllib_and_league_metadata(metadata, experiment_path=experiment, league_dir=league_dir)

    changed = {**env_config, "reward_config": {"score_delta": 1.0, "death": -1.0, "step": -0.001}}
    try:
        validate_resume_metadata(experiment_path=experiment, league_dir=league_dir, env_config=changed)
    except ResumeProvenanceError as exc:
        assert "reward_config_fields_hash" in str(exc)
    else:
        raise AssertionError("resume validation should fail on reward field mismatch")


def test_legacy_metadata_without_reward_hashes_remains_readable(tmp_path):
    env_config = _reward_env_config()
    metadata = TrainingRunMetadata.create(env_config=env_config, latest_league_iteration=1)
    path = rllib_metadata_path(tmp_path / "RLlib" / "rl_run")
    write_metadata(path, metadata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("reward_config_hash")
    payload.pop("reward_config_fields_hash")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = read_metadata(path)

    assert loaded.reward_config_hash == ""
    assert loaded.reward_config_fields_hash == ""


def test_matching_metadata_allows_resume_validation(tmp_path):
    env_config = _env_config()
    metadata = TrainingRunMetadata.create(env_config=env_config, latest_league_iteration=2)
    experiment = tmp_path / "RLlib" / "rl_run"
    league_dir = tmp_path / "redis"
    write_rllib_and_league_metadata(metadata, experiment_path=experiment, league_dir=league_dir)

    assert validate_resume_metadata(experiment_path=experiment, league_dir=league_dir, env_config=env_config) == metadata


def test_unsafe_override_is_explicit_and_detectable(tmp_path, capsys):
    result = validate_resume_metadata(
        experiment_path=tmp_path / "missing",
        league_dir=tmp_path / "redis",
        env_config=_env_config(),
        allow_unsafe_resume=True,
    )

    captured = capsys.readouterr()
    assert result is None
    assert UNSAFE_RESUME_WARNING in captured.out
