"""Reward preset configuration for RLlib Diep training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl.env.rewards import REWARD_FIELDS, RewardConfig, make_reward_config

REWARD_PRESET_DIR = Path(__file__).resolve().parent / "reward_presets"
DEFAULT_REWARD_PRESET = "basic"
DEFAULT_INFO_LOG_AGENTS = ("agent_0", "agent_1", "agent_2", "agent_3")


def _reward_config_to_dict(config: RewardConfig) -> dict[str, float]:
    return {field: float(getattr(config, field)) for field in REWARD_FIELDS}


def _resolve_reward_config_path(name_or_path: str | Path = DEFAULT_REWARD_PRESET) -> Path:
    """Resolve a reward preset name or explicit JSON path to a file."""

    raw_value = str(name_or_path or DEFAULT_REWARD_PRESET)
    requested = Path(raw_value).expanduser()
    if requested.is_file():
        return requested

    is_simple_name = requested.parent == Path(".")
    if is_simple_name:
        preset_filename = requested.name if requested.suffix else f"{requested.name}.json"
        preset_path = REWARD_PRESET_DIR / preset_filename
        if preset_path.is_file():
            return preset_path
        raise FileNotFoundError(
            f"Reward preset not found: {raw_value!r} (looked for {preset_path})"
        )

    raise FileNotFoundError(f"Reward config file not found: {requested}")


def _read_reward_config_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid reward config JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Reward config JSON must be an object: {path}")
    return payload


def load_reward_config(name_or_path: str | Path = DEFAULT_REWARD_PRESET) -> dict[str, float]:
    """Load and validate a reward config preset by name or explicit JSON path."""

    path = _resolve_reward_config_path(name_or_path)
    return _reward_config_to_dict(make_reward_config(_read_reward_config_json(path)))


# Keep this public constant for backward compatibility with older scripts.
BASIC_REWARD_CONFIG = load_reward_config(DEFAULT_REWARD_PRESET)


def training_env_config(**overrides):
    """Return the default env config for long-running RL training."""

    config = {
        "agents": 20,
        "max_ticks": 2000,
        "scenario": "training-ffa-easy",
        "include_snapshot_info": False,
        "fast_reward_state": True,
        "normalize_reward_components": True,
        "info_log_agents": DEFAULT_INFO_LOG_AGENTS,
        "reward_config": BASIC_REWARD_CONFIG,
    }
    config.update(overrides)
    return config


__all__ = [
    "BASIC_REWARD_CONFIG",
    "DEFAULT_REWARD_PRESET",
    "DEFAULT_INFO_LOG_AGENTS",
    "REWARD_PRESET_DIR",
    "load_reward_config",
    "training_env_config",
]
