"""RLlib callbacks that publish Diep main-tank metrics, W&B tables, and training videos."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Mapping

from rl.env.rewards import REWARD_FIELDS

from rl.observability.config import ObservabilityConfig
from rl.observability.core.stats_bridge import EpisodeStatsSummary
from rl.observability.video.eval_video import maybe_write_eval_video

try:  # RLlib is an optional dependency for observability unit tests.
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except ImportError:  # pragma: no cover - used only when Ray is not installed.
    class DefaultCallbacks:  # type: ignore[no-redef]
        """Fallback base so this module remains importable without Ray."""

        pass

logger = logging.getLogger(__name__)

MAIN_TANK_AGENTS = ("agent_0", "agent_1", "agent_2", "agent_3")
GAME_METRIC_KEYS = (
    "score_total",
    "score_from_farming",
    "score_from_pvp",
    "damage_dealt",
    "enemy_damage_dealt",
    "damage_taken",
    "enemy_kills",
    "farm_kills",
    "shots_fired",
    "shots_hit",
    "hit_rate",
    "death_count",
    "death_cause",
    "level_reached",
)


def _empty_reward_totals() -> dict[str, float]:
    """Build the per-agent accumulator used while an RLlib episode is in flight."""
    return {field: 0.0 for field in REWARD_FIELDS}


def _episode_custom_data(episode: Any) -> dict[str, Any]:
    """Return Ray 2.55's mutable per-episode custom data mapping."""
    data = getattr(episode, "custom_data", None)
    if data is None:
        data = {}
        setattr(episode, "custom_data", data)
    return data


def _latest_infos(episode: Any) -> dict[str, dict[str, Any]]:
    """Read only the latest info dictionaries from Ray 2.55's episode API."""
    get_infos = getattr(episode, "get_infos", None)
    if not callable(get_infos):
        return {}

    call_attempts = (
        lambda: get_infos(-1, env_steps=True, return_list=False),
        lambda: get_infos(-1, return_list=False),
        lambda: get_infos(-1),
    )
    values: Any = None
    for call in call_attempts:
        try:
            values = call()
            break
        except TypeError:
            continue

    if isinstance(values, Mapping):
        if all(isinstance(value, Mapping) for value in values.values()):
            return {str(agent): dict(info or {}) for agent, info in values.items()}
        return {"agent_0": dict(values)}
    return {}


def _unwrap_env(env: Any) -> Any:
    """Unwrap common RLlib/PettingZoo wrappers to find the Diep parallel environment."""
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_sim") and hasattr(current, "episode_stats_array"):
            return current
        for attr in ("par_env", "pettingzoo_env", "env", "unwrapped"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    return current


def _env_from_kwargs(kwargs: Mapping[str, Any]) -> Any:
    """Extract the first concrete environment from RLlib callback keyword arguments."""
    for key in ("env", "base_env", "env_runner", "worker"):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        if key == "base_env" and hasattr(candidate, "get_sub_environments"):
            envs = candidate.get_sub_environments()
            if envs:
                return _unwrap_env(envs[0])
        for attr in ("env", "base_env"):
            nested = getattr(candidate, attr, None)
            if nested is not None:
                return _unwrap_env(nested)
        return _unwrap_env(candidate)
    return None


def _log_metric(metrics_logger: Any, key: str, value: Any) -> None:
    """Emit one metric through RLlib's Ray 2.55 MetricsLogger."""
    if metrics_logger is None:
        return
    log_value = getattr(metrics_logger, "log_value", None)
    if not callable(log_value):
        return
    log_value(key, value)


def _log_reward_sum_metrics(metrics_logger: Any, prefix: str, sums: Mapping[str, float]) -> None:
    """Emit raw reward component sums only; normalized/mean metrics are intentionally omitted."""
    for field in REWARD_FIELDS:
        _log_metric(metrics_logger, f"{prefix}/{field}_sum", float(sums.get(field, 0.0)))


def _game_metric_values(summary: EpisodeStatsSummary) -> dict[str, float | int]:
    return {key: getattr(summary, key) for key in GAME_METRIC_KEYS}


def _log_game_metrics(metrics_logger: Any, prefix: str, summary: EpisodeStatsSummary) -> None:
    """Emit episode stat fields from the native stats row through RLlib metrics logging."""
    for key, value in _game_metric_values(summary).items():
        _log_metric(metrics_logger, f"{prefix}/game/{key}", value)


class _LocalTable:
    """Small W&B Table-compatible fallback for offline tests without wandb installed."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.data: list[list[Any]] = []

    def add_data(self, *values: Any) -> None:
        self.data.append(list(values))


class DiepRLlibObservabilityCallback(DefaultCallbacks):
    """Collect Diep metrics for RLlib/Tune and periodically emit training videos."""

    def __init__(self, config: ObservabilityConfig | None = None):
        super().__init__()
        self.config = config or ObservabilityConfig.from_env()
        self.config.ensure_directories()
        self._started_at = time.perf_counter()
        self._last_video_iteration = 0
        self._pending_iteration_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._iteration_tables: dict[str, Any] = {}

    @property
    def main_agents(self) -> tuple[str, ...]:
        agents = tuple(agent for agent in self.config.stats_log_agents if agent in MAIN_TANK_AGENTS)
        return agents or MAIN_TANK_AGENTS

    def on_episode_step(self, *, episode, **kwargs):  # noqa: D401 - RLlib callback signature.
        """Accumulate raw reward components from latest infos exposed by RLlib."""
        interval = max(1, int(self.config.episode_step_sample_interval))
        data = _episode_custom_data(episode)
        step_idx = int(data.get("diep_obs_step_idx", 0))
        data["diep_obs_step_idx"] = step_idx + 1
        if step_idx % interval != 0:
            return

        reward_sums = data.setdefault("diep_reward_sums", defaultdict(_empty_reward_totals))
        step_counts = data.setdefault("diep_step_counts", defaultdict(int))
        for agent, info in _latest_infos(episode).items():
            if agent not in self.main_agents:
                continue
            components = info.get("reward_components") or {}
            if components:
                for field in REWARD_FIELDS:
                    reward_sums[agent][field] += float(components.get(field, 0.0))
                # Count K steps for one sampled accumulation so table episode length
                # remains in environment-step units when sampling every K steps.
                step_counts[agent] += interval

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):  # noqa: D401 - RLlib callback signature.
        """Flush all four main tanks into RLlib metrics and pending W&B table rows."""
        data = _episode_custom_data(episode)
        env = _env_from_kwargs(kwargs)
        sim = getattr(env, "_sim", None)
        if sim is None:
            return
        try:
            rows = sim.episode_stats_array()
            total_reward = float(getattr(episode, "total_reward", 0.0) or 0.0)
            for agent in self.main_agents:
                agent_index = int(agent.split("_", 1)[1])
                reward_sums = dict(data.get("diep_reward_sums", {}).get(agent, _empty_reward_totals()))
                steps = int(data.get("diep_step_counts", {}).get(agent, 0))
                summary = EpisodeStatsSummary.from_row(
                    rows[agent_index],
                    episode_id=str(getattr(episode, "id_", getattr(episode, "episode_id", "episode"))),
                    controlled_agent=agent,
                    episode_length=steps,
                    total_reward=total_reward,
                )
                prefix = f"main/{agent}"
                _log_reward_sum_metrics(metrics_logger, f"{prefix}/reward", reward_sums)
                _log_game_metrics(metrics_logger, prefix, summary)
                row = {
                    "episode_length": steps,
                    "episode_reward": total_reward,
                    **{f"reward/{field}_sum": float(reward_sums.get(field, 0.0)) for field in REWARD_FIELDS},
                    **{f"game/{key}": value for key, value in _game_metric_values(summary).items()},
                }
                self._pending_iteration_rows[agent].append(row)
        except Exception:  # pragma: no cover - metrics must not break training.
            logger.exception("Failed to collect Diep episode stats")

    def _table_columns(self) -> list[str]:
        return [
            "iteration",
            "episode_length",
            "episode_reward",
            *[f"reward/{field}_sum" for field in REWARD_FIELDS],
            *[f"game/{key}" for key in GAME_METRIC_KEYS],
        ]

    def _table_for_agent(self, agent: str) -> Any:
        table = self._iteration_tables.get(agent)
        if table is not None:
            return table
        columns = self._table_columns()
        try:
            import wandb  # type: ignore

            table = wandb.Table(columns=columns)
        except Exception:  # pragma: no cover - tests and minimal installs use fallback.
            table = _LocalTable(columns)
        self._iteration_tables[agent] = table
        return table

    def _empty_iteration_row(self) -> dict[str, Any]:
        return {
            "episode_length": 0,
            "episode_reward": 0.0,
            **{f"reward/{field}_sum": 0.0 for field in REWARD_FIELDS},
            **{f"game/{key}": 0.0 for key in GAME_METRIC_KEYS},
        }

    def _append_iteration_tables(self, result: dict[str, Any], iteration: int) -> None:
        columns = self._table_columns()
        for agent in self.main_agents:
            pending = self._pending_iteration_rows.get(agent) or []
            row = pending[-1] if pending else self._empty_iteration_row()
            table = self._table_for_agent(agent)
            table.add_data(*[iteration if column == "iteration" else row.get(column, 0.0) for column in columns])
            result[f"main/{agent}/iteration_table"] = table
            pending.clear()

    def _log_checkpoint_metadata(self, result: dict[str, Any], iteration: int) -> None:
        checkpoint = result.get("checkpoint") or result.get("best_checkpoints")
        if checkpoint is None:
            return
        result["checkpoint/iteration"] = iteration
        result["checkpoint/path"] = str(checkpoint)
        result["checkpoint/uploaded_to_wandb"] = False

    def on_train_result(self, *, algorithm, metrics_logger=None, result=None, **kwargs):  # noqa: D401 - RLlib callback signature.
        """Append iteration W&B tables and run occasional training-sample videos."""
        if not isinstance(result, dict):
            return
        iteration = int(result.get("training_iteration") or 0)
        if iteration <= 0:
            return
        self._append_iteration_tables(result, iteration)
        self._log_checkpoint_metadata(result, iteration)
        if self.config.video_interval_iterations <= 0:
            return
        if iteration % self.config.video_interval_iterations != 0 or iteration == self._last_video_iteration:
            return
        self._last_video_iteration = iteration
        started_at = time.perf_counter()
        try:
            # The current renderer drives a policy sample from the training algorithm
            # and logs it as training video. It is intentionally not a separate eval
            # metric namespace and failures are non-fatal.
            video = maybe_write_eval_video(algorithm, self.config, iteration=iteration)
            elapsed = video.elapsed_seconds if video is not None else time.perf_counter() - started_at
            result["gameplay/training_video_elapsed_seconds"] = elapsed
            result["gameplay/training_video_policy_fallback"] = bool(video.used_policy_fallback) if video is not None else False
            if video is not None:
                result["gameplay/training_video_path"] = str(video.path)
                _log_metric(metrics_logger, "gameplay/training_video_elapsed_seconds", elapsed)
                _log_metric(metrics_logger, "gameplay/training_video_policy_fallback", int(video.used_policy_fallback))
                try:
                    import wandb  # type: ignore

                    result["gameplay/training_video"] = wandb.Video(str(video.path), fps=self.config.video_fps, format="mp4")
                except ImportError:
                    result["gameplay/training_video"] = str(video.path)
        except Exception:  # pragma: no cover - video failures should never stop learning.
            logger.exception("Failed to write Diep training video at iteration %s", iteration)
            result["gameplay/training_video_error"] = "failed"
            result["gameplay/training_video_elapsed_seconds"] = time.perf_counter() - started_at


__all__ = ["DiepRLlibObservabilityCallback", "MAIN_TANK_AGENTS"]
