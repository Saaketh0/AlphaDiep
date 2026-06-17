#!/usr/bin/env python3
"""Generate one Diep training-style video without training or mutating models.

This is intentionally separate from ``train.py``: it does not start Ray, build a
Tuner, run optimizers, write checkpoints, or update league/model state. It only
steps a local PettingZoo headless environment and renders the full-world video
camera used by observability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RL_TESTING_DIR = Path(__file__).resolve().parent
DIEPCUSTOM_ROOT = RL_TESTING_DIR.parent
for _path in (str(DIEPCUSTOM_ROOT), str(RL_TESTING_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rl.observability.config import ObservabilityConfig
from rl.observability.video.eval_video import write_eval_video
from rewards import DEFAULT_REWARD_PRESET, load_reward_config, training_env_config


class RandomActionAlgorithm:
    """No-model policy placeholder; renderer falls back to env.action_space.sample()."""

    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a single full-world training-style video without running training.",
    )
    parser.add_argument("--run-id", default="video-observation", help="Output run id under the W&B/video root")
    parser.add_argument("--runs-root", default=None, help="Output root (default: training_data/W&B)")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration folder number for output path")
    parser.add_argument("--video-agent", default="agent_1", help="Agent camera to follow (default: agent_1)")
    parser.add_argument("--seed", type=int, default=1, help="Environment reset seed")
    parser.add_argument("--scenario", default="training-ffa-easy", help="Headless scenario to render")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents in the local environment")
    parser.add_argument("--max-ticks", type=int, default=300, help="Environment episode tick limit")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum rendered steps")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    parser.add_argument(
        "--reward-config",
        default=DEFAULT_REWARD_PRESET,
        metavar="PRESET_OR_PATH",
        help="Reward preset name from rl/runtime/reward_presets/ or explicit JSON path",
    )
    return parser


def generate_video(args: argparse.Namespace) -> dict[str, Any]:
    env_config = training_env_config(
        agents=args.agents,
        max_ticks=args.max_ticks,
        scenario=args.scenario,
        seed=args.seed,
        reward_config=load_reward_config(args.reward_config),
    )
    runs_root = Path(args.runs_root) if args.runs_root is not None else ObservabilityConfig().runs_root
    config = ObservabilityConfig(
        run_id=args.run_id,
        runs_root=runs_root,
        video_enabled=True,
        video_agent=args.video_agent,
        video_interval_iterations=1,
        video_fps=args.fps,
        eval_max_steps=args.max_steps,
        eval_env_config=env_config,
    )
    result = write_eval_video(
        RandomActionAlgorithm(),
        config,
        iteration=args.iteration,
        env_config=env_config,
        manual_testing=True,
    )
    return {
        "path": str(result.path.resolve()),
        "relative_path": str(result.path),
        "elapsed_seconds": result.elapsed_seconds,
        "used_policy_fallback": result.used_policy_fallback,
        "video_agent": args.video_agent,
        "scenario": args.scenario,
        "agents": args.agents,
        "max_ticks": args.max_ticks,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "no_training": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = generate_video(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
