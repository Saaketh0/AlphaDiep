"""Tests for cache-first behavior of ``get_num_envs_per_env_runner``."""

from __future__ import annotations

import json

from resource_compute import env_runner_capacity


def _write_cache(path, num_envs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"num_envs_per_env_runner": num_envs}) + "\n")


def test_get_num_envs_uses_cache_when_present(tmp_path, monkeypatch):
    cache_file = tmp_path / "env_runner_capacity.json"
    _write_cache(cache_file, 8)
    monkeypatch.setattr(env_runner_capacity, "CAPACITY_FILE", cache_file)
    monkeypatch.delenv(env_runner_capacity.REPROBE_ENV_VAR, raising=False)

    def _fail_benchmark(*args, **kwargs):
        raise AssertionError("benchmark should not run when cache exists")

    monkeypatch.setattr(env_runner_capacity, "benchmark_and_save", _fail_benchmark)
    monkeypatch.setattr(env_runner_capacity, "_register_diep_env", lambda: None)

    assert env_runner_capacity.get_num_envs_per_env_runner((1, 1, 0, 0)) == 8


def test_get_num_envs_probes_when_cache_missing(tmp_path, monkeypatch):
    cache_file = tmp_path / "env_runner_capacity.json"
    monkeypatch.setattr(env_runner_capacity, "CAPACITY_FILE", cache_file)
    monkeypatch.delenv(env_runner_capacity.REPROBE_ENV_VAR, raising=False)

    calls = {"count": 0}

    def _fake_benchmark(_build_config, **_kwargs):
        calls["count"] += 1
        return {"num_envs_per_env_runner": 2}

    monkeypatch.setattr(env_runner_capacity, "benchmark_and_save", _fake_benchmark)
    monkeypatch.setattr(env_runner_capacity, "_register_diep_env", lambda: None)

    assert env_runner_capacity.get_num_envs_per_env_runner((1, 1, 0, 0)) == 2
    assert calls["count"] == 1


def test_reprobe_env_var_forces_benchmark(tmp_path, monkeypatch):
    cache_file = tmp_path / "env_runner_capacity.json"
    _write_cache(cache_file, 8)
    monkeypatch.setattr(env_runner_capacity, "CAPACITY_FILE", cache_file)
    monkeypatch.setenv(env_runner_capacity.REPROBE_ENV_VAR, "1")

    calls = {"count": 0}

    def _fake_benchmark(_build_config, **_kwargs):
        calls["count"] += 1
        return {"num_envs_per_env_runner": 4}

    monkeypatch.setattr(env_runner_capacity, "benchmark_and_save", _fake_benchmark)
    monkeypatch.setattr(env_runner_capacity, "_register_diep_env", lambda: None)

    assert env_runner_capacity.get_num_envs_per_env_runner((1, 1, 0, 0)) == 4
    assert calls["count"] == 1


def test_force_probe_arg_overrides_cache(tmp_path, monkeypatch):
    cache_file = tmp_path / "env_runner_capacity.json"
    _write_cache(cache_file, 8)
    monkeypatch.setattr(env_runner_capacity, "CAPACITY_FILE", cache_file)
    monkeypatch.delenv(env_runner_capacity.REPROBE_ENV_VAR, raising=False)

    calls = {"count": 0}

    def _fake_benchmark(_build_config, **_kwargs):
        calls["count"] += 1
        return {"num_envs_per_env_runner": 16}

    monkeypatch.setattr(env_runner_capacity, "benchmark_and_save", _fake_benchmark)
    monkeypatch.setattr(env_runner_capacity, "_register_diep_env", lambda: None)

    assert (
        env_runner_capacity.get_num_envs_per_env_runner((1, 1, 0, 0), force_probe=True)
        == 16
    )
    assert calls["count"] == 1
