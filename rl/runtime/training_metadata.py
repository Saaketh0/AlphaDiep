"""Run/league provenance metadata for fail-closed RLlib resume.

RLlib checkpoints intentionally omit ghost policies; the matching league state
lives in Redis/SSD.  These helpers bind both artifact tracks together so resume
can reject stale or missing league state before Tune restores a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Avoid importing league_initialization at module import time: its callback imports this module.
DIEPCUSTOM_ROOT = Path(__file__).resolve().parents[2]
LEAGUE_EXPORT_DIR = DIEPCUSTOM_ROOT / "training_data" / "redis"
RLLIB_CHECKPOINT_DIR = DIEPCUSTOM_ROOT / "training_data" / "RLlib"

SCHEMA_VERSION = 1
RLLIB_METADATA_FILENAME = "run_metadata.json"
LEAGUE_METADATA_FILENAME = "league_metadata.json"
UNSAFE_RESUME_WARNING = (
    "WARNING: --allow-unsafe-resume bypasses run/league provenance checks. "
    "Use only for local development; ghosts may not match the RLlib checkpoint."
)
REWARD_CONFIG_WARNING = (
    "WARNING: reward_config values differ from the saved run metadata. "
    "Resume is allowed because reward weights do not affect checkpoint compatibility."
)


class ResumeProvenanceError(RuntimeError):
    """Raised when RLlib and league metadata are missing or incompatible."""


@dataclass(frozen=True)
class TrainingRunMetadata:
    schema_version: int
    run_id: str
    league_id: str
    training_env_config_hash: str
    policy_layout_hash: str
    latest_league_iteration: int
    timestamp: str
    reward_config_hash: str = ""
    reward_config_fields_hash: str = ""

    @classmethod
    def create(
        cls,
        *,
        env_config: dict[str, Any],
        latest_league_iteration: int,
        run_id: str | None = None,
        league_id: str | None = None,
    ) -> "TrainingRunMetadata":
        return cls(
            schema_version=SCHEMA_VERSION,
            run_id=run_id or new_run_id(),
            league_id=league_id or new_league_id(),
            training_env_config_hash=training_env_config_hash(env_config),
            policy_layout_hash=policy_layout_hash(),
            latest_league_iteration=int(latest_league_iteration),
            timestamp=utc_timestamp(),
            reward_config_hash=reward_config_hash(env_config),
            reward_config_fields_hash=reward_config_fields_hash(env_config),
        )

    def with_latest_iteration(self, latest_league_iteration: int) -> "TrainingRunMetadata":
        return TrainingRunMetadata(
            schema_version=self.schema_version,
            run_id=self.run_id,
            league_id=self.league_id,
            training_env_config_hash=self.training_env_config_hash,
            policy_layout_hash=self.policy_layout_hash,
            latest_league_iteration=int(latest_league_iteration),
            timestamp=utc_timestamp(),
            reward_config_hash=self.reward_config_hash,
            reward_config_fields_hash=self.reward_config_fields_hash,
        )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return f"run-{uuid.uuid4().hex}"


def new_league_id() -> str:
    return f"league-{uuid.uuid4().hex}"


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def _reward_config(env_config: dict[str, Any]) -> Any:
    return env_config.get("reward_config") or {}


def structural_training_env_config(env_config: dict[str, Any]) -> dict[str, Any]:
    """Return resume-critical env config without tunable reward scalar values."""
    return {key: value for key, value in env_config.items() if key != "reward_config"}


def training_env_config_hash(env_config: dict[str, Any]) -> str:
    return stable_hash(structural_training_env_config(env_config))


def reward_config_hash(env_config: dict[str, Any]) -> str:
    return stable_hash(_reward_config(env_config))


def reward_config_fields_hash(env_config: dict[str, Any]) -> str:
    reward_config = _reward_config(env_config)
    fields = sorted(reward_config) if isinstance(reward_config, dict) else []
    return stable_hash(fields)


def policy_layout_hash() -> str:
    from league_initialization.constants import CHAR_CLASSES, GHOST_POLICIES, MAIN_POLICIES

    layout = {
        "char_classes": CHAR_CLASSES,
        "main_policies": MAIN_POLICIES,
        "ghost_policies": GHOST_POLICIES,
        "ghost_policy_count": len(GHOST_POLICIES),
    }
    return stable_hash(layout)


def rllib_metadata_path(experiment_path: str | Path | None = None) -> Path:
    base = Path(experiment_path) if experiment_path is not None else RLLIB_CHECKPOINT_DIR / "rl_run"
    return base / RLLIB_METADATA_FILENAME


def league_metadata_path(league_dir: str | Path | None = None) -> Path:
    return (Path(league_dir) if league_dir is not None else LEAGUE_EXPORT_DIR) / LEAGUE_METADATA_FILENAME


def write_metadata(path: str | Path, metadata: TrainingRunMetadata) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_metadata(path: str | Path) -> TrainingRunMetadata:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResumeProvenanceError(f"Missing provenance metadata: {source}") from exc
    payload.setdefault("reward_config_hash", "")
    payload.setdefault("reward_config_fields_hash", "")
    missing = set(TrainingRunMetadata.__dataclass_fields__) - set(payload)
    if missing:
        raise ResumeProvenanceError(f"Incomplete provenance metadata at {source}: missing {sorted(missing)}")
    return TrainingRunMetadata(**{name: payload[name] for name in TrainingRunMetadata.__dataclass_fields__})


def write_rllib_and_league_metadata(
    metadata: TrainingRunMetadata,
    *,
    experiment_path: str | Path | None = None,
    league_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    return (
        write_metadata(rllib_metadata_path(experiment_path), metadata),
        write_metadata(league_metadata_path(league_dir), metadata),
    )


def validate_resume_metadata(
    *,
    experiment_path: str | Path,
    env_config: dict[str, Any],
    league_dir: str | Path | None = None,
    allow_unsafe_resume: bool = False,
) -> TrainingRunMetadata | None:
    """Validate RLlib/league provenance, failing closed unless explicitly bypassed."""
    if allow_unsafe_resume:
        print(UNSAFE_RESUME_WARNING, flush=True)
        return None

    rllib = read_metadata(rllib_metadata_path(experiment_path))
    league = read_metadata(league_metadata_path(league_dir))

    expected_env_hash = training_env_config_hash(env_config)
    legacy_expected_env_hash = stable_hash(env_config)
    expected_policy_hash = policy_layout_hash()
    expected_reward_hash = reward_config_hash(env_config)
    expected_reward_fields_hash = reward_config_fields_hash(env_config)
    mismatches: list[str] = []
    for field in (
        "schema_version",
        "run_id",
        "league_id",
        "training_env_config_hash",
        "policy_layout_hash",
        "reward_config_hash",
        "reward_config_fields_hash",
    ):
        if getattr(rllib, field) != getattr(league, field):
            mismatches.append(f"{field}: RLlib={getattr(rllib, field)!r}, league={getattr(league, field)!r}")
    if rllib.training_env_config_hash != expected_env_hash:
        is_legacy_metadata = not rllib.reward_config_fields_hash
        if not (is_legacy_metadata and rllib.training_env_config_hash == legacy_expected_env_hash):
            mismatches.append("training_env_config_hash does not match current structural training config")
    if rllib.policy_layout_hash != expected_policy_hash:
        mismatches.append("policy_layout_hash does not match current policy/class layout")
    if rllib.reward_config_fields_hash and rllib.reward_config_fields_hash != expected_reward_fields_hash:
        mismatches.append("reward_config_fields_hash does not match current reward field set")
    if league.latest_league_iteration < 0:
        mismatches.append("league latest iteration is missing/negative")

    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise ResumeProvenanceError(f"Unsafe resume refused; provenance mismatch:\n  - {detail}")
    if rllib.reward_config_hash and rllib.reward_config_hash != expected_reward_hash:
        print(REWARD_CONFIG_WARNING, flush=True)
    return rllib


def metadata_from_env(default: TrainingRunMetadata | None = None) -> TrainingRunMetadata | None:
    path = os.environ.get("DIEP_TRAINING_RUN_METADATA")
    if not path:
        return default
    return read_metadata(path)


__all__ = [
    "LEAGUE_METADATA_FILENAME",
    "RLLIB_METADATA_FILENAME",
    "REWARD_CONFIG_WARNING",
    "ResumeProvenanceError",
    "TrainingRunMetadata",
    "UNSAFE_RESUME_WARNING",
    "league_metadata_path",
    "metadata_from_env",
    "policy_layout_hash",
    "read_metadata",
    "reward_config_fields_hash",
    "reward_config_hash",
    "rllib_metadata_path",
    "structural_training_env_config",
    "training_env_config_hash",
    "validate_resume_metadata",
    "write_metadata",
    "write_rllib_and_league_metadata",
]
