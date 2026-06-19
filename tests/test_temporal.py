"""Tests for the bi-temporal supersession core."""

from __future__ import annotations

from datetime import datetime

from supersede.models import Fact
from supersede.temporal import (
    active_facts,
    batch_detect_conflicts,
    detect_conflict,
    resolve_in_place,
    select_valid_at,
)


def _fact(subject: str, predicate: str, obj: str, **kwargs) -> Fact:
    return Fact(subject=subject, predicate=predicate, object=obj, **kwargs)


def test_update_conflict_supersedes_on_new_object():
    old = _fact("Alice", "lives in", "Boston")
    new = _fact("Alice", "lives in", "Denver")
    conflict = detect_conflict(new, [old])
    assert conflict is not None
    assert conflict.kind == "update"
    assert conflict.strategy == "supersede"
    assert conflict.existing_fact.id == old.id


def test_identical_object_is_duplicate_noop():
    old = _fact("Alice", "lives in", "Boston")
    new = _fact("Alice", "lives in", "Boston")
    conflict = detect_conflict(new, [old])
    assert conflict is not None
    assert conflict.kind == "duplicate"
    assert conflict.strategy == "ignore"


def test_no_conflict_across_different_predicate():
    old = _fact("Alice", "lives in", "Boston")
    new = _fact("Alice", "works at", "Acme")
    assert detect_conflict(new, [old]) is None


def test_normalization_is_case_insensitive():
    old = _fact("alice", "Lives In", "Boston")
    new = _fact("Alice", "lives in", "Denver")
    assert detect_conflict(new, [old]) is not None


def test_already_superseded_facts_are_skipped():
    old = _fact("Alice", "lives in", "Boston", superseded_by="fact_x")
    new = _fact("Alice", "lives in", "Denver")
    assert detect_conflict(new, [old]) is None


def test_resolve_in_place_closes_window_and_points_forward():
    old = _fact("Alice", "lives in", "Boston", valid_from=datetime(2020, 1, 1))
    new = _fact("Alice", "lives in", "Denver", valid_from=datetime(2023, 6, 1))
    facts = [old, new]
    conflict = detect_conflict(new, [old])
    assert conflict is not None
    resolve_in_place(conflict, facts)

    superseded = next(f for f in facts if f.id == old.id)
    assert superseded.superseded_by == new.id
    assert superseded.valid_to == datetime(2023, 6, 1)
    assert [f.id for f in active_facts(facts)] == [new.id]


def test_select_valid_at_respects_windows():
    old = _fact(
        "Alice",
        "lives in",
        "Boston",
        valid_from=datetime(2020, 1, 1),
        valid_to=datetime(2023, 6, 1),
    )
    new = _fact("Alice", "lives in", "Denver", valid_from=datetime(2023, 6, 1))
    facts = [old, new]
    at_2021 = select_valid_at(facts, datetime(2021, 1, 1))
    at_2024 = select_valid_at(facts, datetime(2024, 1, 1))
    assert {f.object for f in at_2021} == {"Boston"}
    assert {f.object for f in at_2024} == {"Denver"}


def test_batch_detects_chain_of_updates():
    facts = [
        _fact("Alice", "lives in", "Boston"),
        _fact("Alice", "lives in", "Denver"),
        _fact("Alice", "lives in", "Austin"),
    ]
    conflicts = batch_detect_conflicts(facts)
    # Two update steps: Boston->Denver and (Boston,Denver)->Austin pairings.
    assert all(c.kind == "update" for c in conflicts)
    assert len(conflicts) >= 2
