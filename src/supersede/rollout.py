"""Framework-agnostic memory rollout state machine.

This is the heart of the supersede environment, written without any dependency
on ``verifiers`` so it can be unit-tested offline. Both the offline harness
(``scripts/eval_longmemeval.py``) and the verifiers ``MultiTurnEnv`` wrapper
(``supersede.env``) drive the same state machine.

The agent sees one session at a time and rewrites a bounded notes field; it
never re-sees raw sessions. After all sessions it is asked the question and must
answer from its notes alone. This is the regime in which frontier models
measurably fail at supersession (see docs/findings/v1-longmemeval.md).

Turn protocol (each "user" message is produced by :meth:`current_prompt`, each
"assistant" reply is fed back via :meth:`step`):

    initial: memory prompt for session 0
    step(notes_0) -> memory prompt for session 1
    ...
    step(notes_{k-1}) -> the question prompt
    step(answer)       -> None (rollout complete; answer stored)
"""

from __future__ import annotations

from dataclasses import dataclass, field

MEMORY_SYSTEM = (
    "You maintain a NOTES field that is your ONLY memory of a long, ongoing "
    "conversation with a user. You will never see earlier sessions again, only "
    "your notes. Each turn you receive your current notes and the transcript of "
    "one new session. Rewrite your COMPLETE notes to capture everything about "
    "the user that may matter later. When new information changes something you "
    "already noted, OVERWRITE it; never keep outdated facts. Your notes are "
    "hard-capped at {budget} characters. Output ONLY the new notes."
)
ANSWER_SYSTEM = (
    "These notes are your entire memory of the conversation. Answer the user's "
    "question using only the notes. Be concise."
)


def memory_prompt(notes: str, session: str, budget: int) -> str:
    return (f"CURRENT NOTES:\n{notes or '(empty)'}\n\nNEW SESSION:\n{session}\n\n"
            f"Rewrite your complete notes (max {budget} chars).")


def question_prompt(notes: str, question: str) -> str:
    return f"NOTES:\n{notes}\n\nQuestion: {question}"


@dataclass
class MemoryRollout:
    """Drives one bounded-memory rollout over a list of rendered sessions."""

    sessions: list[str]
    question: str
    budget: int = 300
    notes: str = ""
    sidx: int = 0
    phase: str = "memory"          # memory -> answer -> done
    final_answer: str | None = None
    _system: str = field(default="", repr=False)

    def system_prompt(self) -> str:
        """System prompt for the current phase."""
        if self.phase == "answer":
            return ANSWER_SYSTEM
        return MEMORY_SYSTEM.format(budget=self.budget)

    def current_prompt(self) -> str:
        """The user message to send this turn."""
        if self.phase == "memory":
            return memory_prompt(self.notes, self.sessions[self.sidx], self.budget)
        if self.phase == "answer":
            return question_prompt(self.notes, self.question)
        raise RuntimeError("rollout is already complete")

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def step(self, assistant_text: str) -> str | None:
        """Feed the model's reply; return the next user prompt, or None if done."""
        text = (assistant_text or "").strip()
        if self.phase == "memory":
            self.notes = text[: self.budget]
            self.sidx += 1
            if self.sidx < len(self.sessions):
                return self.current_prompt()
            self.phase = "answer"
            return self.current_prompt()
        if self.phase == "answer":
            self.final_answer = text
            self.phase = "done"
            return None
        return None


__all__ = [
    "ANSWER_SYSTEM",
    "MEMORY_SYSTEM",
    "MemoryRollout",
    "memory_prompt",
    "question_prompt",
]
