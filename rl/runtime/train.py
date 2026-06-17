#!/usr/bin/env python3
"""Canonical Diep training CLI.

Examples:
    python rl/runtime/train.py doctor
    python rl/runtime/train.py redis start
    python rl/runtime/train.py seed --count 50
    python rl/runtime/train.py smoke
    python rl/runtime/train.py train
    python rl/runtime/train.py resume --resume-path training_data/RLlib/rl_run
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

# Allow `python rl/runtime/train.py ...` from the repository root without asking
# users to set PYTHONPATH manually.
RUNTIME_DIR = Path(__file__).resolve().parent
DIEPCUSTOM_ROOT = RUNTIME_DIR.parents[1]
for _path in (str(DIEPCUSTOM_ROOT), str(RUNTIME_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from league_initialization.constants import GHOST_POLICIES
from league_initialization.paths import LEAGUE_EXPORT_DIR, REDIS_SERVER_DATA_DIR, RLLIB_CHECKPOINT_DIR
from league_initialization.seed_league_cache import seed_league_cache
from model_store import RedisModelStore
from ray_code import run_training
from resume_from_checkpoint import resume_training, validate_resume
from rewards import DEFAULT_REWARD_PRESET, load_reward_config, training_env_config
from training_metadata import validate_resume_metadata
from training_runtime import build_cpp_headless, experiment_path, start_redis

OPTIONAL_FFMPEG = "ffmpeg"
REQUIRED_MODULES = ("ray", "redis", "torch", "gymnasium", "pettingzoo", "safetensors", "zstandard")


def _module_status(name: str) -> str:
    return "ok" if importlib.util.find_spec(name) is not None else "missing"


def command_doctor(_args: argparse.Namespace) -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Repo: {DIEPCUSTOM_ROOT}")
    for module in REQUIRED_MODULES:
        print(f"Python module {module}: {_module_status(module)}")
    print(f"Docker: {'ok' if shutil.which('docker') else 'missing'}")
    print(f"ffmpeg (optional video): {'ok' if shutil.which(OPTIONAL_FFMPEG) else 'missing'}")
    try:
        from rl.env.headless import abi_version

        print(f"C++ headless library: ok (ABI {abi_version()})")
    except Exception as exc:  # noqa: BLE001 - doctor reports all dependency failures.
        print(f"C++ headless library: missing/error ({exc})")
    print(f"Redis data: {REDIS_SERVER_DATA_DIR}")
    print(f"League exports: {LEAGUE_EXPORT_DIR}")
    print(f"RLlib checkpoints: {RLLIB_CHECKPOINT_DIR}")
    return 0


def command_redis(args: argparse.Namespace) -> int:
    return start_redis(args.redis_action)


def command_seed(args: argparse.Namespace) -> int:
    env_config = training_env_config(reward_config=load_reward_config(args.reward_config))
    start_redis("start")
    result = seed_league_cache(count=args.count, write_provenance=True, env_config=env_config)
    per_class = {char_class: len(keys) for char_class, keys in result["written"].items()}
    print(f"Seeded Redis keys per class: {per_class}")
    print(f"Exported {len(result['exported'])} league files to {LEAGUE_EXPORT_DIR}")
    print(f"Wrote league metadata: {result['metadata_path']}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    env_config = training_env_config(reward_config=load_reward_config(args.reward_config))
    start_redis("start")
    run_training(env_config=env_config)
    return 0


def command_resume(args: argparse.Namespace) -> int:
    env_config = training_env_config(reward_config=load_reward_config(args.reward_config))
    start_redis("start")
    resume_training(
        resume_path=args.resume_path,
        env_config=env_config,
        allow_unsafe_resume=args.allow_unsafe_resume,
    )
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    os.environ.setdefault("DIEP_GHOST_REFRESH_INTERVAL", "1")
    env_config = training_env_config(
        max_ticks=args.max_ticks,
        reward_config=load_reward_config(args.reward_config),
    )
    start_redis("start")
    build_cpp_headless()
    # Seed a tiny deterministic league and then run one minimal training iteration.
    seed_league_cache(count=args.seed_count, write_provenance=False, env_config=env_config)
    run_training(
        env_config=env_config,
        run_name="training_smoke",
        stop_iterations=1,
        checkpoint_frequency=1,
        num_to_keep=1,
        smoke=True,
    )
    store = RedisModelStore(snapshot_every=0)
    latest = store.latest_iteration()
    missing_exports = [char_class for char_class in store.classes if not (LEAGUE_EXPORT_DIR / char_class).is_dir()]
    if latest < args.seed_count:
        raise RuntimeError(f"Smoke did not advance league iteration (latest={latest})")
    if missing_exports:
        raise RuntimeError(f"Missing SSD league exports for classes: {missing_exports}")
    validate_resume_metadata(
        experiment_path=experiment_path("training_smoke"),
        env_config=env_config,
    )
    print(
        "Training smoke passed: Redis/SSD league present, one PPO iteration completed, "
        f"{len(GHOST_POLICIES)} ghost policy slots configured."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Diep RL training CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Validate Python deps, C++ library, Ray, Redis/Docker availability")
    doctor.set_defaults(func=command_doctor)

    redis_parser = sub.add_parser("redis", help="Manage Docker Redis")
    redis_parser.add_argument("redis_action", choices=("start", "status", "stop"))
    redis_parser.set_defaults(func=command_redis)

    def add_reward_config_argument(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--reward-config",
            default=DEFAULT_REWARD_PRESET,
            metavar="PRESET_OR_PATH",
            help="Reward preset name from rl/runtime/reward_presets/ or explicit JSON path (default: basic)",
        )

    seed = sub.add_parser("seed", help="Start Redis and seed the ghost league")
    seed.add_argument("--count", type=int, default=50, help="Iterations to seed per class")
    add_reward_config_argument(seed)
    seed.set_defaults(func=command_seed)

    smoke = sub.add_parser("smoke", help="Run real-Redis end-to-end training smoke")
    smoke.add_argument("--max-ticks", type=int, default=64)
    smoke.add_argument("--seed-count", type=int, default=2)
    add_reward_config_argument(smoke)
    smoke.set_defaults(func=command_smoke)

    train = sub.add_parser("train", help="Start Redis, validate/seed league, and launch fresh training")
    add_reward_config_argument(train)
    train.set_defaults(func=command_train)

    resume = sub.add_parser("resume", help="Start Redis, validate metadata, and resume Tune experiment")
    resume.add_argument("--resume-path", default=None)
    resume.add_argument("--allow-unsafe-resume", action="store_true")
    add_reward_config_argument(resume)
    resume.set_defaults(func=command_resume)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:  # noqa: BLE001 - CLI should emit clear terminal errors.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
