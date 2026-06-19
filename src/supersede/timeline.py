"""Procedural generator for fact-mutation timelines.

A *timeline* is a multi-session interaction in which one tracked fact is
updated several times while unrelated distractor facts stay constant. The
agent is told the facts session by session and, at the end, asked for the
*current* value of the tracked fact. The generator records both the current
answer and every superseded ("stale") answer so a verifier can tell whether
the agent used current or stale information.

Generation is fully seeded and deterministic, so a timeline is reproducible
from its ``(seed, config)`` and never needs to be checked into the repo.
This is also what makes the benchmark contamination-resistant: the surface
strings are resampled per seed rather than memorised from a static set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from supersede.models import Fact

# Predicate -> object vocabulary. Objects within a pool are chosen to be
# mutually non-substring so stale/current matching is unambiguous.
_POOLS: dict[str, list[str]] = {
    "lives in": [
        "Boston", "Denver", "Seattle", "Miami", "Chicago", "Portland",
        "Atlanta", "Dallas", "Phoenix", "Detroit", "Nashville", "Tucson",
    ],
    "works at": [
        "Acme", "Globex", "Initech", "Umbrella", "Hooli", "Cyberdyne",
        "Soylent", "Wonka", "Massive Dynamic", "Vehement Capital",
    ],
    "drives a": [
        "Toyota", "Subaru", "Volvo", "Mazda", "Hyundai", "Honda",
        "Jeep", "Kia", "Nissan", "Lexus",
    ],
    "is allergic to": [
        "peanuts", "shellfish", "pollen", "penicillin", "latex",
        "dairy", "soy", "gluten",
    ],
    "manages the": [
        "Phoenix", "Atlas", "Orion", "Titan", "Nimbus", "Vertex",
        "Quantum", "Pinnacle", "Apex", "Helix",
    ],
}

_SUBJECTS: list[str] = [
    "Alice", "Bob", "Carla", "Devin", "Elena", "Frank", "Grace", "Hassan",
    "Iris", "Jamal", "Kira", "Liam", "Maya", "Noah", "Omar", "Priya",
    "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wes", "Xena",
]

_VERB_NOW: dict[str, str] = {
    "lives in": "now lives in",
    "works at": "now works at",
    "drives a": "now drives a",
    "is allergic to": "is now allergic to",
    "manages the": "now manages the",
}


@dataclass
class TimelineConfig:
    """Knobs that control timeline difficulty.

    Difficulty comes from *interference*: several facts evolve at once, so the
    conversation is full of "now ..." updates and the agent must bind each
    update to the right subject and recover only the queried fact's latest
    value. The queried fact's final update is also buried mid-history rather
    than being the last thing said.
    """

    n_tracked: int = 6          # subjects whose facts evolve concurrently
    chain_len: int = 3          # distinct values each tracked fact takes
    n_distractors: int = 6      # unrelated, never-updated facts


@dataclass
class Timeline:
    """One generated multi-session fact-tracking task."""

    subject: str
    predicate: str
    current_answer: str
    stale_answers: list[str]
    sessions: list[str]                 # one rendered message per session, in order
    query: str
    facts: list[Fact] = field(default_factory=list)  # full history, supersession resolved
    seed: int = 0
    metadata_gap: int = 0   # update events occurring after the queried final update

    def to_info(self) -> dict:
        """Serialise to a verifiers ``info`` dict (JSON-safe)."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "current_answer": self.current_answer,
            "stale_answers": list(self.stale_answers),
            "sessions": list(self.sessions),
            "query": self.query,
            "seed": self.seed,
            "gap": self.metadata_gap,
        }


def _statement(subject: str, predicate: str, obj: str, *, update: bool) -> str:
    verb = _VERB_NOW[predicate] if update else predicate
    return f"{subject} {verb} {obj}."


def _assign_predicate(rng: random.Random, subject: str,
                      used: set[tuple[str, str]]) -> str:
    """Pick a predicate for ``subject`` not already used for that subject."""
    preds = list(_POOLS)
    rng.shuffle(preds)
    for p in preds:
        if (subject, p) not in used:
            return p
    return preds[0]


def generate_timeline(seed: int, config: TimelineConfig | None = None) -> Timeline:
    """Generate one deterministic fact-mutation timeline from ``seed``."""
    cfg = config or TimelineConfig()
    rng = random.Random(seed)

    subjects = rng.sample(_SUBJECTS, k=min(cfg.n_tracked, len(_SUBJECTS)))
    base_time = datetime(2024, 1, 1)
    facts: list[Fact] = []

    # Build one evolving chain per tracked subject.
    chains: dict[str, tuple[str, list[str]]] = {}   # subject -> (predicate, values)
    used: set[tuple[str, str]] = set()
    initial_msgs: list[str] = []
    update_events: dict[str, list[tuple[int, str]]] = {}  # subject -> [(step, msg)]
    for subj in subjects:
        pred = _assign_predicate(rng, subj, used)
        used.add((subj, pred))
        values = rng.sample(_POOLS[pred], k=min(cfg.chain_len, len(_POOLS[pred])))
        chains[subj] = (pred, values)
        initial_msgs.append(_statement(subj, pred, values[0], update=False))
        update_events[subj] = [
            (i, _statement(subj, pred, values[i], update=True))
            for i in range(1, len(values))
        ]
        for i, obj in enumerate(values):
            facts.append(Fact(
                subject=subj, predicate=pred, object=obj,
                valid_from=base_time + timedelta(days=30 * i),
                superseded_by=None if i == len(values) - 1 else f"{subj}_{i + 1}",
                metadata={"role": "tracked", "subject": subj, "step": i},
            ))

    # Stable distractor facts (single statement each, never updated).
    spare = [s for s in _SUBJECTS if s not in subjects]
    rng.shuffle(spare)
    for j in range(cfg.n_distractors):
        d_subj = spare[j % len(spare)] if spare else rng.choice(_SUBJECTS)
        d_pred = rng.choice(list(_POOLS))
        d_obj = rng.choice(_POOLS[d_pred])
        facts.append(Fact(subject=d_subj, predicate=d_pred, object=d_obj,
                          metadata={"role": "distractor"}))
        initial_msgs.append(_statement(d_subj, d_pred, d_obj, update=False))

    # Session 0 seeds every initial value + distractors, shuffled.
    rng.shuffle(initial_msgs)
    sessions: list[str] = [" ".join(initial_msgs)]

    # Randomly interleave all update events, preserving per-subject order.
    pending = {s: list(evts) for s, evts in update_events.items() if evts}
    while pending:
        subj = rng.choice(list(pending))
        _, msg = pending[subj].pop(0)
        sessions.append(msg)
        if not pending[subj]:
            del pending[subj]

    # Query a subject whose fact actually changed (chain_len >= 2).
    queryable = [s for s in subjects if len(chains[s][1]) >= 2]
    queried = rng.choice(queryable) if queryable else subjects[0]
    q_pred, q_values = chains[queried]
    current_answer = q_values[-1]
    stale_answers = q_values[:-1]

    # How many update events occur after the queried fact's final update.
    final_msg = _statement(queried, q_pred, current_answer, update=True)
    gap = len(sessions) - 1 - sessions.index(final_msg)

    query = (
        f"As of now, what is the value for: '{queried} {q_pred} ___'? "
        f"Reply with only the single current value, nothing else."
    )

    return Timeline(
        subject=queried,
        predicate=q_pred,
        current_answer=current_answer,
        stale_answers=stale_answers,
        sessions=sessions,
        query=query,
        facts=facts,
        seed=seed,
        metadata_gap=gap,
    )


def generate_dataset(n: int, start_seed: int = 0,
                     config: TimelineConfig | None = None) -> list[Timeline]:
    """Generate ``n`` timelines with consecutive seeds."""
    return [generate_timeline(start_seed + i, config) for i in range(n)]


__all__ = ["Timeline", "TimelineConfig", "generate_dataset", "generate_timeline"]
