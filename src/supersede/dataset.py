"""Task loaders for the supersede environment.

Produces plain task dicts of the shape the environment consumes:

    {
        "question": str,           # the query asked after all sessions
        "answer": str,             # gold current value
        "sessions": list[str],     # rendered session transcripts, in time order
        "info": {...},             # qid, qtype, optional stale_values
    }

Primary source is LongMemEval's ``knowledge-update`` subset (real conversational
supersession). A synthetic source built from ``supersede.timeline`` is also
provided for unit tests and ablations (it ships known ``stale_values`` so the
stale penalty is exercised).
"""

from __future__ import annotations

import json
from pathlib import Path


def render_session(session: list[dict]) -> str:
    """Render a LongMemEval session (list of role/content turns) to text."""
    return "\n".join(f"{t['role']}: {t['content']}" for t in session)


def load_longmemeval(path: str | Path,
                     question_type: str = "knowledge-update") -> list[dict]:
    """Load LongMemEval items of ``question_type`` into task dicts."""
    data = json.loads(Path(path).read_text())
    tasks: list[dict] = []
    for ex in data:
        if question_type and ex.get("question_type") != question_type:
            continue
        tasks.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "sessions": [render_session(s) for s in ex["haystack_sessions"]],
            "info": {
                "qid": ex["question_id"],
                "qtype": ex["question_type"],
                "stale_values": [],  # not labeled in LongMemEval
            },
        })
    return tasks


def synthetic_tasks(n: int, start_seed: int = 0, **timeline_kwargs) -> list[dict]:
    """Build tasks from synthetic timelines (carry known stale_values)."""
    from supersede.timeline import TimelineConfig, generate_dataset

    cfg = TimelineConfig(**timeline_kwargs) if timeline_kwargs else None
    tasks: list[dict] = []
    for tl in generate_dataset(n, start_seed=start_seed, config=cfg):
        tasks.append({
            "question": tl.query,
            "answer": tl.current_answer,
            "sessions": list(tl.sessions),
            "info": {
                "qid": f"synth_{tl.seed}",
                "qtype": "synthetic",
                "stale_values": list(tl.stale_answers),
            },
        })
    return tasks


__all__ = ["load_longmemeval", "render_session", "synthetic_tasks"]
