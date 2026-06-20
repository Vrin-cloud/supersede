"""Rewards for the supersede environment.

The primary signal is ``answered_current``: did the agent's final answer convey
the current (gold) value? Two graders are provided:

- :func:`answer_matches` -- a programmatic, ungameable matcher (normalized
  variant + token-overlap). Fast, deterministic, no API. This is the default so
  the environment can be evaluated and trained without an extra judge model.
- An optional LLM judge (wired in :mod:`supersede.env` via verifiers'
  ``JudgeRubric``) for higher-fidelity grading on messy free-text answers.

A ``stale_use_penalty`` is provided for settings where the superseded values
are known (our synthetic timelines, or LongMemEval items annotated with the old
value): it penalizes answers that assert a value the conversation later
replaced. On bare LongMemEval (gold-only) it is a no-op.
"""

from __future__ import annotations

import re

_SPLIT = re.compile(r"\(or\b|\bor\b|/|;|,|\band\b", re.IGNORECASE)


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def gold_variants(gold: str) -> list[str]:
    """Split a gold answer into acceptable surface variants.

    LongMemEval golds often pack alternates like "25 minutes and 50 seconds
    (or 25:50)". We keep the full normalized gold plus each fragment, dropping
    fragments shorter than 2 chars.
    """
    raw = [gold] + _SPLIT.split(gold)
    out: list[str] = []
    for piece in raw:
        n = normalize(piece)
        if len(n) >= 2 and n not in out:
            out.append(n)
    return out


def _token_overlap(answer_n: str, gold_n: str) -> float:
    a, g = set(answer_n.split()), set(gold_n.split())
    if not g:
        return 0.0
    return len(a & g) / len(g)


def answer_matches(answer: str, gold: str, *, overlap_threshold: float = 0.6) -> bool:
    """True if ``answer`` conveys ``gold`` (variant substring or token overlap)."""
    a = normalize(answer)
    if not a:
        return False
    variants = gold_variants(gold)
    for v in variants:
        if v and v in a:
            return True
    full = normalize(gold)
    return _token_overlap(a, full) >= overlap_threshold


def stale_use_penalty(answer: str, stale_values: list[str]) -> float:
    """Return 1.0 if the answer asserts any known superseded value, else 0.0."""
    a = normalize(answer)
    for s in stale_values or []:
        sn = normalize(s)
        if sn and re.search(rf"(?<!\w){re.escape(sn)}(?!\w)", a):
            return 1.0
    return 0.0


# ---- verifiers-style reward functions (bind by parameter name) ----

def answered_current(state, answer) -> float:
    """+1 if the rollout's final answer conveys the gold value, else 0."""
    final = ""
    if isinstance(state, dict):
        ro = state.get("rollout")
        final = getattr(ro, "final_answer", None) or state.get("final_answer") or ""
    return 1.0 if answer_matches(final, answer or "") else 0.0


def stale_penalty(state, info) -> float:
    """-1 if the final answer uses a known superseded value (else 0).

    ``info['stale_values']`` must be present; otherwise this is a no-op so the
    reward is safe on gold-only datasets like bare LongMemEval.
    """
    if not isinstance(state, dict):
        return 0.0
    ro = state.get("rollout")
    final = getattr(ro, "final_answer", None) or state.get("final_answer") or ""
    stale = (info or {}).get("stale_values", []) if isinstance(info, dict) else []
    return -stale_use_penalty(final, stale)


__all__ = [
    "answer_matches",
    "answered_current",
    "gold_variants",
    "normalize",
    "stale_penalty",
    "stale_use_penalty",
]
