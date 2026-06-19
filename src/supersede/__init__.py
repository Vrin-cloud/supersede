"""Supersede: an RL environment for training agents to use current, not stale, facts.

Public surface is intentionally small. The reusable temporal core
(:mod:`supersede.temporal`) is independent of the environment and can be
imported on its own.
"""

from __future__ import annotations

__version__ = "0.0.1"

from supersede.models import Fact
from supersede.temporal import (
    FactConflict,
    batch_detect_conflicts,
    detect_conflict,
    select_valid_at,
)

__all__ = [
    "Fact",
    "FactConflict",
    "batch_detect_conflicts",
    "detect_conflict",
    "select_valid_at",
    "__version__",
]
