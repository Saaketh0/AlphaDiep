"""Compatibility shim; use rl.env.pettingzoo_env for training code."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.env.pettingzoo_env import *  # noqa: F401,F403
