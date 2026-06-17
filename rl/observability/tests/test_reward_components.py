"""Reward component logging for the supported RLlib observability path."""

from __future__ import annotations

from pathlib import Path

from rl.observability.config import ObservabilityConfig
from rl.observability.tests.test_rllib_callbacks import FakeEnv, FakeEpisode, FakeMetricsLogger
from rl.observability.logging.rllib_callbacks import DiepRLlibObservabilityCallback


def test_rllib_callback_logs_raw_reward_sums_without_normalized_or_means(tmp_path: Path):
    config = ObservabilityConfig(run_id="reward-components", runs_root=tmp_path)
    callback = DiepRLlibObservabilityCallback(config=config)
    episode = FakeEpisode()

    callback.on_episode_step(episode=episode)
    callback.on_episode_step(episode=episode)
    metrics = FakeMetricsLogger()
    callback.on_episode_end(episode=episode, metrics_logger=metrics, env=FakeEnv(), env_index=0)

    assert metrics.values["main/agent_0/reward/score_delta_sum"] == 4.0
    assert metrics.values["main/agent_1/reward/score_delta_sum"] == 6.0
    assert metrics.values["main/agent_2/reward/score_delta_sum"] == 8.0
    assert metrics.values["main/agent_3/reward/score_delta_sum"] == 10.0
    assert all("reward_normalized" not in key for key in metrics.values)
    assert all(not key.endswith("_mean") for key in metrics.values)


def test_rllib_iteration_tables_include_raw_reward_sums_only(tmp_path: Path):
    config = ObservabilityConfig(run_id="reward-table", runs_root=tmp_path, video_interval_iterations=0)
    callback = DiepRLlibObservabilityCallback(config=config)
    episode = FakeEpisode()

    callback.on_episode_step(episode=episode)
    callback.on_episode_end(episode=episode, metrics_logger=FakeMetricsLogger(), env=FakeEnv(), env_index=0)
    result = {"training_iteration": 3}
    callback.on_train_result(algorithm=object(), result=result, metrics_logger=FakeMetricsLogger())

    table = result["main/agent_0/iteration_table"]
    assert "reward/score_delta_sum" in table.columns
    assert "reward/score_delta_mean" not in table.columns
    assert all("normalized" not in column for column in table.columns)
