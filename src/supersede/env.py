"""verifiers environment wrapper for the supersede memory task.

Builds a ``MultiTurnEnv`` in which the agent maintains a bounded notes memory
session-by-session and is then asked the question, scored on whether it answers
with the current (non-stale) value. The turn logic lives in the
framework-agnostic :class:`supersede.rollout.MemoryRollout` (unit-tested
offline); this module is the thin verifiers binding.

``verifiers`` is imported lazily inside :func:`load_environment` so the rest of
the package (rollout, reward, dataset, temporal) imports with no heavy deps.

Entry point for the Environments Hub / prime-rl:

    from supersede.env import load_environment
    env = load_environment(data_path="data/longmemeval_oracle.json",
                           max_examples=78, budget=300)
"""

from __future__ import annotations

import json

from supersede.dataset import load_longmemeval, synthetic_tasks
from supersede.reward import answer_matches
from supersede.rollout import MemoryRollout


def _build_tasks(data_path, question_type, max_examples, synthetic_n):
    if data_path:
        tasks = load_longmemeval(data_path, question_type=question_type)
    else:
        tasks = synthetic_tasks(synthetic_n)
    if max_examples:
        tasks = tasks[:max_examples]
    return tasks


def load_environment(
    data_path: str | None = None,
    question_type: str = "knowledge-update",
    max_examples: int | None = None,
    synthetic_n: int = 50,
    budget: int = 300,
    judge_model: str | None = None,
):
    """Construct the supersede ``MultiTurnEnv``.

    Args:
        data_path: path to a LongMemEval json file. If None, uses synthetic
            timelines (which carry known stale_values for the stale penalty).
        question_type: LongMemEval subset to load (default knowledge-update).
        max_examples: cap on number of tasks.
        budget: character cap on the agent's notes memory.
        judge_model: if set, grade answers with an LLM judge (JudgeRubric)
            instead of the programmatic matcher.
    """
    import verifiers as vf
    from datasets import Dataset

    tasks = _build_tasks(data_path, question_type, max_examples, synthetic_n)
    rows = []
    for t in tasks:
        info = dict(t["info"])
        info["sessions"] = t["sessions"]
        info["budget"] = budget
        rows.append({
            "question": t["question"],
            "answer": t["answer"],
            "info": json.dumps(info),
        })
    dataset = Dataset.from_list(rows)

    class SupersedeMemoryEnv(vf.MultiTurnEnv):
        """Bounded-memory supersession rollout: history is never re-fed."""

        async def setup_state(self, state: "vf.State") -> "vf.State":
            info = state["info"]
            if isinstance(info, str):
                info = json.loads(info)
            state["rollout"] = MemoryRollout(
                sessions=list(info["sessions"]),
                question=state.get("question") or info.get("question", ""),
                budget=int(info.get("budget", budget)),
            )
            return state

        async def get_prompt_messages(self, state: "vf.State"):
            ro: MemoryRollout = state["rollout"]
            return [
                {"role": "system", "content": ro.system_prompt()},
                {"role": "user", "content": ro.current_prompt()},
            ]

        async def env_response(self, messages, state: "vf.State"):
            ro: MemoryRollout = state["rollout"]
            last = ""
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    last = m.get("content") or ""
                    break
            nxt = ro.step(last)
            if nxt is None:
                state["final_answer"] = ro.final_answer
                return []
            return [{"role": "user", "content": nxt}]

        @vf.stop
        async def rollout_done(self, state: "vf.State") -> bool:
            ro: MemoryRollout = state.get("rollout")
            return bool(ro and ro.done)

    def answered_current(state, answer) -> float:
        ro = state.get("rollout") if isinstance(state, dict) else None
        final = getattr(ro, "final_answer", None) or (
            state.get("final_answer") if isinstance(state, dict) else "") or ""
        return 1.0 if answer_matches(final, answer or "") else 0.0

    if judge_model:
        rubric = vf.JudgeRubric(judge_model=judge_model)
    else:
        rubric = vf.Rubric(funcs=[answered_current], weights=[1.0])

    return SupersedeMemoryEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=200,
    )


__all__ = ["load_environment"]
