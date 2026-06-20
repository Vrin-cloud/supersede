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
import os
import sys

from supersede.dataset import load_longmemeval, synthetic_tasks


def _dbg(*a):
    if os.environ.get("SUPERSEDE_DEBUG"):
        print("[supersede]", *a, file=sys.stderr, flush=True)
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
    max_turns: int = 50,
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
    from verifiers.envs.multiturn_env import MultiTurnEnv

    tasks = _build_tasks(data_path, question_type, max_examples, synthetic_n)
    rows = []
    for t in tasks:
        info = dict(t["info"])
        info["sessions"] = t["sessions"]
        info["budget"] = budget
        info["question"] = t["question"]   # carry the query reliably via info
        rows.append({
            "question": str(t["question"]),
            "answer": str(t["answer"]),
            "info": json.dumps(info),
        })
    dataset = Dataset.from_list(rows)

    class SupersedeMemoryEnv(MultiTurnEnv):
        """Bounded-memory supersession rollout: history is never re-fed."""

        async def setup_state(self, state: "vf.State") -> None:
            info = state.get("info")
            if isinstance(info, str):
                info = json.loads(info)
            info = info or {}
            state["rollout"] = MemoryRollout(
                sessions=list(info["sessions"]),
                question=state.get("question") or info.get("question", ""),
                budget=int(info.get("budget", budget)),
            )

        def _bounded_view(self, ro: MemoryRollout):
            """The ONLY thing the model sees: system + notes + current item.

            Raw session history is never re-fed; the agent's memory is its
            (bounded) notes, carried in ``ro``.
            """
            return [
                vf.SystemMessage(content=ro.system_prompt()),
                vf.UserMessage(content=ro.current_prompt()),
            ]

        async def get_prompt_messages(self, state: "vf.State"):
            ro: MemoryRollout = state["rollout"]
            traj = state.get("trajectory") or []
            _dbg("gpm: traj_len", len(traj), "phase", ro.phase, "sidx", ro.sidx)
            # First turn: present session 0's memory prompt.
            if not traj:
                return self._bounded_view(ro)
            # Later turns: advance the rollout using the model's last reply,
            # then return only the bounded view (NOT the accumulated history).
            prev = traj[-1]
            prev_msgs = list(prev["prompt"]) + list(prev["completion"])
            await self.env_response(prev_msgs, state)
            if state.get("final_env_response") is not None:
                return prev_msgs  # rollout finished; loop skips model + stops
            view = self._bounded_view(ro)
            if os.environ.get("SUPERSEDE_DEBUG"):
                _dbg("  view ->", [(m.role, (m.content or "")[:140]) for m in view])
            return view

        async def env_response(self, messages, state: "vf.State"):
            ro: MemoryRollout = state["rollout"]
            last = ""
            for m in reversed(messages):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role == "assistant":
                    last = (m.get("content") if isinstance(m, dict)
                            else getattr(m, "content", "")) or ""
                    break
            _dbg("env_response: last[:40]", repr(last[:40]), "phase", ro.phase, "sidx", ro.sidx)
            nxt = ro.step(last)
            _dbg("  -> after step: phase", ro.phase, "sidx", ro.sidx, "done", ro.done)
            if nxt is None:
                state["final_answer"] = ro.final_answer
                state["final_env_response"] = []  # signal completion to the loop
                return []
            return [vf.UserMessage(content=nxt)]

        @vf.stop
        async def rollout_done(self, state: "vf.State") -> bool:
            ro = state.get("rollout")
            return bool(ro and ro.done)

    def _last_assistant(completion) -> str:
        """The final assistant message = the agent's answer to the question."""
        if isinstance(completion, str):
            return completion
        for m in reversed(completion or []):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "assistant":
                return (m.get("content") if isinstance(m, dict)
                        else getattr(m, "content", "")) or ""
        return ""

    async def answered_current(completion, answer, state, **kwargs) -> float:
        final = _last_assistant(completion)
        if not final and isinstance(state, dict):
            ro = state.get("rollout")
            final = getattr(ro, "final_answer", None) or state.get("final_answer") or ""
        ok = answer_matches(final, answer or "")
        _dbg("reward: gold", repr((answer or "")[:40]), "final", repr(final[:60]),
             "-> match", ok)
        return 1.0 if ok else 0.0

    if judge_model:
        rubric = vf.JudgeRubric(judge_model=judge_model)
    else:
        rubric = vf.Rubric(funcs=[answered_current], weights=[1.0])

    return SupersedeMemoryEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns,
    )


__all__ = ["load_environment"]
