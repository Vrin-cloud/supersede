"""Bi-temporal conflict detection and supersession.

Ported from the Engram ``dialogue.temporal`` module (itself ported from the
production Vrin engine) with one structural change: the write-back path no
longer depends on a corpus backend. The supersession environment holds its
facts as an in-memory list inside the rollout ``state``, so resolution is a
pure, synchronous list mutation (:func:`resolve_in_place`).

The flow is:

- :func:`detect_conflict` returns a single :class:`FactConflict` when a new
  fact disagrees with an active existing fact on the same
  ``(subject, predicate)``. Returns ``None`` when there is no temporal
  overlap or no semantic disagreement.
- :func:`batch_detect_conflicts` groups facts by ``(subject, predicate)``
  and reports every cross-pair update conflict.
- :func:`resolve_in_place` applies a conflict by closing the superseded
  fact's validity window and pointing it at its successor.

Detection rules are intentionally conservative:

- ``duplicate``: existing active fact has the same object; the new fact is a
  no-op.
- ``update``: existing active fact has a different object but overlapping
  validity; the existing fact is superseded by the new one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from supersede.models import Fact

ConflictKind = Literal["duplicate", "update"]
"""Kind of conflict detected between two facts on the same subject + predicate."""

ResolutionStrategy = Literal["ignore", "supersede"]
"""How the conflict resolver should act on the conflict."""


@dataclass(frozen=True)
class FactConflict:
    """A detected conflict between a new fact and one existing active fact."""

    kind: ConflictKind
    new_fact: Fact
    existing_fact: Fact
    strategy: ResolutionStrategy
    reason: str


def _normalize(text: str) -> str:
    return text.strip().lower()


def _has_temporal_overlap(
    start_a: datetime | None,
    end_a: datetime | None,
    start_b: datetime | None,
    end_b: datetime | None,
) -> bool:
    """Return ``True`` if the two half-open intervals overlap.

    A ``None`` boundary means open-ended in that direction. When either
    ``valid_from`` is unknown the facts are treated as overlapping; the
    conservative call is to surface the conflict rather than hide it.
    """
    if start_a is None or start_b is None:
        return True

    effective_end_a = end_a if end_a is not None else datetime.max
    effective_end_b = end_b if end_b is not None else datetime.max

    return start_a <= effective_end_b and start_b <= effective_end_a


def detect_conflict(
    new_fact: Fact,
    existing_facts: Iterable[Fact],
) -> FactConflict | None:
    """Detect whether ``new_fact`` conflicts with any active existing fact.

    Active means not already superseded and sharing the same normalized
    subject and predicate. The first matching conflict is returned; callers
    needing the full list should use :func:`batch_detect_conflicts`.
    """
    new_subject = _normalize(new_fact.subject)
    new_predicate = _normalize(new_fact.predicate)
    new_object = _normalize(new_fact.object)

    for existing in existing_facts:
        if existing.superseded_by is not None:
            continue
        if _normalize(existing.subject) != new_subject:
            continue
        if _normalize(existing.predicate) != new_predicate:
            continue

        overlap = _has_temporal_overlap(
            new_fact.valid_from,
            new_fact.valid_to,
            existing.valid_from,
            existing.valid_to,
        )
        if not overlap:
            continue

        if _normalize(existing.object) == new_object:
            return FactConflict(
                kind="duplicate",
                new_fact=new_fact,
                existing_fact=existing,
                strategy="ignore",
                reason="Identical active fact already exists",
            )

        return FactConflict(
            kind="update",
            new_fact=new_fact,
            existing_fact=existing,
            strategy="supersede",
            reason=(
                f"New object {new_fact.object!r} supersedes existing object "
                f"{existing.object!r} for {new_fact.subject} / {new_fact.predicate}"
            ),
        )

    return None


def batch_detect_conflicts(facts: Sequence[Fact]) -> list[FactConflict]:
    """Detect every cross-pair update conflict in ``facts``.

    Facts are grouped by ``(subject, predicate)``. Within each group of size
    >= 2, each later fact is checked against all earlier ones.
    """
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for fact in facts:
        if fact.superseded_by is not None:
            continue
        key = (_normalize(fact.subject), _normalize(fact.predicate))
        groups[key].append(fact)

    conflicts: list[FactConflict] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for i in range(1, len(group)):
            conflict = detect_conflict(group[i], group[:i])
            if conflict is not None and conflict.kind == "update":
                conflicts.append(conflict)
    return conflicts


def select_valid_at(facts: Iterable[Fact], when: datetime) -> list[Fact]:
    """Return the subset of ``facts`` valid at ``when``.

    A fact is valid at ``when`` if ``valid_from <= when <= valid_to``, where
    each missing bound is treated as ``-infinity`` / ``+infinity``.
    """
    valid: list[Fact] = []
    for fact in facts:
        start = fact.valid_from if fact.valid_from is not None else datetime.min
        end = fact.valid_to if fact.valid_to is not None else datetime.max
        if start <= when <= end:
            valid.append(fact)
    return valid


def active_facts(facts: Iterable[Fact]) -> list[Fact]:
    """Return only the facts that nothing has superseded yet."""
    return [fact for fact in facts if fact.superseded_by is None]


def resolve_in_place(
    conflict: FactConflict,
    facts: list[Fact],
    *,
    valid_to: datetime | None = None,
) -> None:
    """Apply a conflict's resolution by mutating ``facts`` in place.

    ``duplicate`` conflicts are a no-op. ``update`` conflicts close the
    existing fact's validity window at ``valid_to`` (defaulting to the new
    fact's ``valid_from`` if set, otherwise now) and set its
    ``superseded_by`` pointer to the new fact's id. The superseded fact is
    kept in the list so the rollout retains the full version history needed
    to score stale-vs-current behaviour.
    """
    if conflict.strategy == "ignore":
        return

    closed_at = valid_to or conflict.new_fact.valid_from or datetime.now()
    for index, fact in enumerate(facts):
        if fact.id == conflict.existing_fact.id:
            facts[index] = fact.model_copy(
                update={
                    "valid_to": closed_at,
                    "superseded_by": conflict.new_fact.id,
                }
            )
            break


__all__ = [
    "ConflictKind",
    "FactConflict",
    "ResolutionStrategy",
    "active_facts",
    "batch_detect_conflicts",
    "detect_conflict",
    "resolve_in_place",
    "select_valid_at",
]
