"""Small helper for writing periodic Diep gameplay evaluation videos."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

import numpy as np

from rl.env import DiepCustomParallelEnv
from rl.observability.config import ObservabilityConfig
from rl.observability.core.stats_bridge import EpisodeStatsSummary
from rl.observability.video.render_overlay import overlay_frame
from rl.observability.video.render_world import DEFAULT_FRAME_SIZE, render_world_frame
from rl.observability.video.video_writer import FfmpegVideoWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalVideoResult:
    """Metadata produced by a synchronous eval-video render."""

    path: Path
    elapsed_seconds: float
    used_policy_fallback: bool


# Maps an agent id to the policy id used by the training script, with a safe fallback.
def _default_policy_mapping(agent_id: str) -> str:
    index = int(str(agent_id).split("_")[-1])
    if index < 4:
        return ["main_class_A", "main_class_B", "main_class_C", "main_class_D"][index]
    char_class = "ABCD"[index % 4]
    ghost_slot = (index // 4) % 4
    return f"class_{char_class}_ghost_{ghost_slot}"


# Computes one action from the current RLlib algorithm, falling back to random samples.
def _compute_action(
    algorithm: Any,
    env: DiepCustomParallelEnv,
    agent: str,
    observation: Mapping[str, Any],
    policy_mapping_fn: Callable[[str], str] | None = None,
    fallback_state: MutableMapping[str, bool] | None = None,
) -> Any:
    policy_id = policy_mapping_fn(agent) if policy_mapping_fn is not None else _default_policy_mapping(agent)
    compute = getattr(algorithm, "compute_single_action", None)
    if callable(compute):
        try:
            return compute(observation, policy_id=policy_id, explore=False)
        except Exception:
            logger.debug("Falling back to sampled eval action for %s", agent, exc_info=True)
    if fallback_state is not None:
        fallback_state["used"] = True
    return env.action_space(agent).sample()


# Builds an EpisodeStatsSummary for overlay text from the current simulator row.
def _summary_for_agent(env: DiepCustomParallelEnv, agent: str, episode_length: int, total_reward: float) -> EpisodeStatsSummary:
    agent_index = int(agent.split("_", 1)[1])
    return EpisodeStatsSummary.from_row(
        env._sim.episode_stats_array()[agent_index],
        episode_id="eval",
        controlled_agent=agent,
        episode_length=episode_length,
        total_reward=total_reward,
    )


# Writes a frame with grid rendering and gameplay overlay.
def _write_frame(writer: FfmpegVideoWriter, env: DiepCustomParallelEnv, obs: Mapping[str, Any], agent: str, step_reward: float, total_reward: float, length: int, trail: list[tuple[int, int]]) -> None:
    snapshot = env.snapshot()
    frame = render_world_frame(snapshot, followed_agent_id=int(agent.split("_", 1)[1]), frame_size=DEFAULT_FRAME_SIZE)
    stats = _summary_for_agent(env, agent, length, total_reward)
    overlaid = overlay_frame(
        frame,
        snapshot=snapshot,
        agent_id=int(agent.split("_", 1)[1]),
        prev_action_obs=obs.get("prev_action_obs"),
        step_reward=step_reward,
        total_reward=total_reward,
        stats=stats,
        trail=trail,
    )
    writer.write(overlaid)


# Runs one deterministic eval episode and writes it to eval.mp4.
def write_eval_video(
    algorithm: Any,
    config: ObservabilityConfig,
    *,
    iteration: int,
    env_config: Mapping[str, Any] | None = None,
    policy_mapping_fn: Callable[[str], str] | None = None,
    manual_testing: bool = False,
) -> EvalVideoResult:
    started_at = time.perf_counter()
    output_path = config.video_path(iteration, manual_testing=manual_testing)
    eval_config = dict(env_config or config.eval_env_config)
    eval_config.setdefault("include_snapshot_info", True)
    eval_config.setdefault("normalize_reward_components", True)
    env = DiepCustomParallelEnv(**eval_config)
    fallback_state: dict[str, bool] = {"used": False}
    agent = config.video_agent
    total_reward = 0.0
    length = 0
    trail: list[tuple[int, int]] = []
    try:
        observations, _infos = env.reset(seed=eval_config.get("seed", 1))
        first_frame = render_world_frame(env.snapshot(), followed_agent_id=int(agent.split("_", 1)[1]), frame_size=DEFAULT_FRAME_SIZE)
        with FfmpegVideoWriter(output_path, width=first_frame.shape[1], height=first_frame.shape[0], fps=config.video_fps) as writer:
            _write_frame(writer, env, observations[agent], agent, 0.0, total_reward, length, trail)
            while agent in observations and length < config.eval_max_steps:
                actions = {
                    current_agent: _compute_action(
                        algorithm,
                        env,
                        current_agent,
                        observation,
                        policy_mapping_fn,
                        fallback_state,
                    )
                    for current_agent, observation in observations.items()
                }
                observations, rewards, terminations, truncations, _infos = env.step(actions)
                step_reward = float(rewards.get(agent, 0.0))
                total_reward += step_reward
                length += 1
                if agent in observations:
                    _write_frame(writer, env, observations[agent], agent, step_reward, total_reward, length, trail)
                if bool(terminations.get(agent, False) or truncations.get(agent, False)):
                    break
    finally:
        env.close()
    return EvalVideoResult(output_path, time.perf_counter() - started_at, fallback_state["used"])


# Writes an eval video only when local W&B logging can consume it later.
def maybe_write_eval_video(algorithm: Any, config: ObservabilityConfig, *, iteration: int) -> EvalVideoResult | None:
    if not config.video_enabled:
        return None
    result = write_eval_video(algorithm, config, iteration=iteration)
    if not result.path.exists() or result.path.stat().st_size <= 0:
        raise RuntimeError(f"eval video was not created: {result.path}")
    return result


__all__ = ["EvalVideoResult", "maybe_write_eval_video", "write_eval_video"]
