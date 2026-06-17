"""Pytest import path bootstrap for rl/runtime tests run from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "rl" / "runtime"
for path in (ROOT, RUNTIME):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
