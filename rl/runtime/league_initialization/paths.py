"""Canonical SSD paths for league weight persistence.

All training artifacts live under ``diepcustom/training_data/`` regardless of the
process working directory. Paths are resolved from this file's location so they
are stable whether code runs from ``diepcustom/`` or ``diepcustom/rl/runtime/``.
"""

from __future__ import annotations

from pathlib import Path

# .../diepcustom/rl/runtime/league_initialization/paths.py -> .../diepcustom
DIEPCUSTOM_ROOT = Path(__file__).resolve().parents[3]

TRAINING_DATA_ROOT = DIEPCUSTOM_ROOT / "training_data"

# League weight exports (safetensors per class/iteration); primary hydrate source.
LEAGUE_EXPORT_DIR = TRAINING_DATA_ROOT / "redis"

# Redis server data dir (Docker bind mount target for AOF/RDB).
REDIS_SERVER_DATA_DIR = TRAINING_DATA_ROOT / "redis-server"

# RLlib/Tune training checkpoints for resume (bulky; separate from league weights).
RLLIB_CHECKPOINT_DIR = TRAINING_DATA_ROOT / "RLlib"
