"""Scoring for supersession answers: did the agent use current or stale info?

The verifier reduces a free-text answer to one of four verdicts against the
timeline's known current/stale values:

- ``correct``   : names the current value and no stale value
- ``stale``     : names a superseded value and not the current one
- ``ambiguous`` : names both current and stale (hedged / listed history)
- ``wrong``     : names neither

The headline metrics are **accuracy** (fraction ``correct``) and
**stale-rate** (fraction ``stale``). Stale-rate is the number the whole
project exists to drive down, so it is reported as a first-class signal, not
folded into accuracy.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

Verdict = Literal["correct", "stale", "ambiguous", "wrong"]


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _mentions(answer_norm: str, value: str) -> bool:
    """Word-boundary-aware containment of a (possibly multi-word) value."""
    v = normalize(value)
    if not v:
        return False
    return re.search(rf"(?<!\w){re.escape(v)}(?!\w)", answer_norm) is not None


def classify(answer: str, current: str, stale: list[str]) -> Verdict:
    """Classify a free-text answer against the current and stale values."""
    a = normalize(answer)
    cur = _mentions(a, current)
    stl = any(_mentions(a, s) for s in stale)
    if cur and not stl:
        return "correct"
    if stl and not cur:
        return "stale"
    if cur and stl:
        return "ambiguous"
    return "wrong"


@dataclass
class EvalSummary:
    """Aggregate verdicts over a set of rollouts."""

    n: int
    counts: Counter

    @property
    def accuracy(self) -> float:
        return self.counts["correct"] / self.n if self.n else 0.0

    @property
    def stale_rate(self) -> float:
        return self.counts["stale"] / self.n if self.n else 0.0

    @property
    def ambiguous_rate(self) -> float:
        return self.counts["ambiguous"] / self.n if self.n else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.counts["wrong"] / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "stale_rate": round(self.stale_rate, 4),
            "ambiguous_rate": round(self.ambiguous_rate, 4),
            "wrong_rate": round(self.wrong_rate, 4),
            "counts": dict(self.counts),
        }


def summarize(verdicts: list[Verdict]) -> EvalSummary:
    return EvalSummary(n=len(verdicts), counts=Counter(verdicts))


__all__ = ["EvalSummary", "Verdict", "classify", "normalize", "summarize"]
