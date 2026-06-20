"""Offline tests for the reward / answer-matching logic."""

from __future__ import annotations

from supersede.reward import (
    answer_matches,
    answered_current,
    gold_variants,
    stale_penalty,
    stale_use_penalty,
)
from supersede.rollout import MemoryRollout


def test_gold_variants_splits_parenthetical_alternates():
    vs = gold_variants("25 minutes and 50 seconds (or 25:50)")
    assert any("25 50" in v or "2550" in v for v in vs)   # the 25:50 alternate
    assert "50 seconds" in vs                              # a split fragment


def test_answer_matches_exact_and_variant():
    assert answer_matches("It was 25:50.", "25 minutes and 50 seconds (or 25:50)")
    assert answer_matches("four", "four")
    assert answer_matches("You tried four of them.", "four")


def test_answer_matches_token_overlap():
    assert answer_matches("She moved to the suburbs recently", "the suburbs")


def test_answer_matches_rejects_wrong():
    assert not answer_matches("I don't know", "the suburbs")
    assert not answer_matches("downtown", "the suburbs")


def test_stale_use_penalty_detects_superseded_value():
    assert stale_use_penalty("You live in Boston", ["Boston", "Denver"]) == 1.0
    assert stale_use_penalty("You live in Austin", ["Boston", "Denver"]) == 0.0


def test_answered_current_reads_rollout_final_answer():
    ro = MemoryRollout(sessions=["S0"], question="Q?", budget=100)
    ro.step("notes")
    ro.step("the answer is four")
    state = {"rollout": ro}
    assert answered_current(state, "four") == 1.0
    assert answered_current(state, "seven") == 0.0


def test_stale_penalty_is_noop_without_stale_values():
    ro = MemoryRollout(sessions=["S0"], question="Q?", budget=100)
    ro.step("notes")
    ro.step("Boston")
    state = {"rollout": ro}
    assert stale_penalty(state, {}) == 0.0
    assert stale_penalty(state, {"stale_values": ["Boston"]}) == -1.0
