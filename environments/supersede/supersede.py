"""Environments Hub entry point for the supersede memory environment.

This thin module is what ``prime env push`` / ``prime-rl`` import. The actual
implementation lives in the ``supersede`` library (``supersede.env``); this
keeps the Hub package small and the logic unit-tested in one place.

    prime env install supersede
    prime eval run supersede -m openai/gpt-4.1-mini -a '{"data_path": "data/longmemeval_oracle.json"}'
"""

from __future__ import annotations

from supersede.env import load_environment

__all__ = ["load_environment"]
