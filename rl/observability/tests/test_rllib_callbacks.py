"""Tests for Ray 2.55-native RLlib observability callbacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.observability.config import ObservabilityConfig
from rl.observability.core.stats_bridge import EPISODE_STATS_FIELDS
from rl.observability.logging import rllib_callbacks
from rl.observability.logging.rllib_callbacks import DiepRLlibObservabilityCallback
from rl.observability.video.eval_video import EvalVideoResult


class FakeEpisode:
    def __init__(self) -> None:
        self.custom_data: dict = {}
        self.total_reward = 12.5
        self.id_ = "fake-episode"
        self.get_infos_calls: list[tuple[tuple, dict]] = []

    def get_infos(self, *args, **kwargs):
        self.get_infos_calls.append((args, kwargs))
        if args != (-1,):
            raise AssertionError("callback must request only the latest info index")
        return {
            "agent_0": {"reward_components": {"score_delta": 2.0, "step": 1.0}},
            "agent_1": {"reward_components": {"score_delta": 3.0, "step": 1.0}},
            "agent_2": {"reward_components": {"score_delta": 4.0, "step": 1.0}},
            "agent_3": {"reward_components": {"score_delta": 5.0, "step": 1.0}},
        }


class FakeMetricsLogger:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def log_value(self, key, value, **kwargs) -> None:
        self.values[key] = value


class FakeSim:
    def episode_stats_array(self):
        values = {field: 0.0 for field in EPISODE_STATS_FIELDS}
        values.update(
            {
                "lifetime_steps": 7.0,
                "score_total": 100.0,
                "score_from_farming": 25.0,
                "score_from_pvp": 75.0,
                "damage_dealt": 9.0,
                "enemy_damage_dealt": 4.0,
                "damage_taken": 3.0,
                "shots_fired": 10.0,
                "shots_hit": 4.0,
                "enemy_kills": 2.0,
                "farm_kills": 5.0,
                "level_reached": 8.0,
            }
        )
        rows = []
        for index in range(4):
            agent_values = dict(values)
            agent_values["score_total"] = 100.0 + index
            agent_values["score_from_farming"] = 25.0 + index
            agent_values["score_from_pvp"] = 75.0 + index
            rows.append([agent_values[field] for field in EPISODE_STATS_FIELDS])
        return np.asarray(rows, dtype=np.float64)


class FakeEnv:
    def __init__(self) -> None:
        self._sim = FakeSim()

    def episode_stats_array(self):
        return self._sim.episode_stats_array()


def test_rllib_callback_uses_custom_data_latest_infos_and_metrics_logger(tmp_path: Path):
    config = ObservabilityConfig(run_id="callback-test", runs_root=tmp_path)
    callback = DiepRLlibObservabilityCallback(config=config)
    episode = FakeEpisode()

    callback.on_episode_step(episode=episode)
    callback.on_episode_step(episode=episode)

    assert "diep_reward_sums" in episode.custom_data
    assert episode.custom_data["diep_step_counts"]["agent_0"] == 2
    assert episode.custom_data["diep_step_counts"]["agent_3"] == 2
    assert all(call[0] == (-1,) for call in episode.get_infos_calls)

    metrics = FakeMetricsLogger()
    callback.on_episode_end(episode=episode, metrics_logger=metrics, env=FakeEnv(), env_index=0)

    assert metrics.values["main/agent_0/reward/score_delta_sum"] == 4.0
    assert metrics.values["main/agent_3/reward/score_delta_sum"] == 10.0
    assert "main/agent_0/reward/score_delta_mean" not in metrics.values
    assert "reward_normalized/score_delta_sum" not in metrics.values
    assert metrics.values["main/agent_0/game/score_total"] == 100.0
    assert metrics.values["main/agent_3/game/score_total"] == 103.0
    assert metrics.values["main/agent_0/game/hit_rate"] == 0.4


def test_episode_step_sample_interval_preserves_mean(tmp_path: Path):
    """P8: with interval=5 over 10 steps we accumulate twice but mean stays unbiased."""
    config = ObservabilityConfig(
        run_id="step-sampling",
        runs_root=tmp_path,
        stats_log_agents=("agent_0",),
        episode_step_sample_interval=5,
    )
    callback = DiepRLlibObservabilityCallback(config=config)
    episode = FakeEpisode()

    for _ in range(10):
        callback.on_episode_step(episode=episode)

    # Two sampled accumulations at step indices 0 and 5, each adding score_delta=2.0.
    assert episode.custom_data["diep_reward_sums"]["agent_0"]["score_delta"] == 4.0
    # ``step_counts`` is incremented by the interval per sample so the episode-end
    # mean divides by 10 environment steps, matching the original per-step behavior.
    assert episode.custom_data["diep_step_counts"]["agent_0"] == 10

    metrics = FakeMetricsLogger()
    callback.on_episode_end(episode=episode, metrics_logger=metrics, env=FakeEnv(), env_index=0)
    # Per-step mean: 4.0 sum / 10 effective steps = 0.4 (same as 2.0/step actual sampled value).
    assert metrics.values["main/agent_0/reward/score_delta_sum"] == 4.0
    assert "main/agent_0/reward/score_delta_mean" not in metrics.values


def test_episode_step_sample_interval_default_one_matches_legacy(tmp_path: Path):
    """Interval=1 must behave exactly like the pre-P8 per-step accumulation."""
    config = ObservabilityConfig(
        run_id="step-sampling-default",
        runs_root=tmp_path,
        stats_log_agents=("agent_0",),
    )
    assert config.episode_step_sample_interval == 1
    callback = DiepRLlibObservabilityCallback(config=config)
    episode = FakeEpisode()

    callback.on_episode_step(episode=episode)
    callback.on_episode_step(episode=episode)

    assert episode.custom_data["diep_step_counts"]["agent_0"] == 2
    assert episode.custom_data["diep_reward_sums"]["agent_0"]["score_delta"] == 4.0


def test_train_result_video_metadata_and_config_propagation(monkeypatch, tmp_path: Path):
    eval_env_config = {"agents": 20, "max_ticks": 123, "seed": 9}
    config = ObservabilityConfig(
        run_id="video-test",
        runs_root=tmp_path,
        eval_env_config=eval_env_config,
        video_interval_iterations=1,
    )
    callback = DiepRLlibObservabilityCallback(config=config)
    expected_path = config.eval_iteration_dir(1) / "eval.mp4"

    def fake_maybe_write_eval_video(algorithm, observed_config, *, iteration):
        assert observed_config.eval_env_config == eval_env_config
        assert iteration == 1
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_bytes(b"fake-mp4")
        return EvalVideoResult(expected_path, elapsed_seconds=0.25, used_policy_fallback=True)

    monkeypatch.setattr(rllib_callbacks, "maybe_write_eval_video", fake_maybe_write_eval_video)
    result = {"training_iteration": 1}
    metrics = FakeMetricsLogger()

    callback.on_train_result(algorithm=object(), result=result, metrics_logger=metrics)

    assert result["gameplay/training_video_path"] == str(expected_path)
    assert result["gameplay/training_video_elapsed_seconds"] == 0.25
    assert result["gameplay/training_video_policy_fallback"] is True
    assert metrics.values["gameplay/training_video_elapsed_seconds"] == 0.25
    assert metrics.values["gameplay/training_video_policy_fallback"] == 1


def test_train_result_appends_four_main_agent_tables(tmp_path: Path):
    config = ObservabilityConfig(run_id="table-test", runs_root=tmp_path, video_interval_iterations=0)
    callback = DiepRLlibObservabilityCallback(config=config)
    episode = FakeEpisode()
    callback.on_episode_step(episode=episode)
    callback.on_episode_end(episode=episode, metrics_logger=FakeMetricsLogger(), env=FakeEnv(), env_index=0)

    result = {"training_iteration": 7}
    callback.on_train_result(algorithm=object(), result=result, metrics_logger=FakeMetricsLogger())

    for agent in ("agent_0", "agent_1", "agent_2", "agent_3"):
        table = result[f"main/{agent}/iteration_table"]
        assert "iteration" in table.columns
        assert "reward/score_delta_sum" in table.columns
        assert "reward/score_delta_mean" not in table.columns
        assert "game/score_from_farming" in table.columns
        assert len(table.data) == 1
        assert table.data[0][table.columns.index("iteration")] == 7
