"""Supersede: a bounded-memory supersession environment for the Hub.

Self-contained (no external project dependency): the rollout state machine,
reward, and data loading are all inlined here so ``prime env install supersede``
works out of the box. The research library version lives at
https://github.com/Vrin-cloud/supersede.

Task: a multi-session interaction in which a fact is updated, followed by a
query for the fact's *current* value. Two modes:

* ``full_context`` (upper bound): all sessions are placed in the context.
* bounded memory (default): the agent rewrites a capped notes field one session
  at a time and never re-sees raw sessions, then answers from memory alone.

Reward ``answered_current``: 1.0 iff the final answer conveys the current value
(programmatic, ungameable matcher; no judge model needed).

    prime env install supersede
    prime eval run supersede -m openai/gpt-4.1-mini -a '{"max_examples": 78}'
    prime eval run supersede -m openai/gpt-4.1-mini -a '{"full_context": true}'
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------- #
# Data: LongMemEval knowledge-update (auto-downloaded, MIT license).
# --------------------------------------------------------------------------- #
_LME_URL = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
            "resolve/main/longmemeval_oracle.json")
_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "supersede"


def _ensure_longmemeval() -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    dest = _CACHE / "longmemeval_oracle.json"
    if not dest.exists():
        urllib.request.urlretrieve(_LME_URL, dest)
    return dest


def _render_session(session: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in session)


def _load_tasks(question_type: str, max_examples: int | None) -> list[dict]:
    data = json.loads(_ensure_longmemeval().read_text())
    tasks = []
    for ex in data:
        if question_type and ex.get("question_type") != question_type:
            continue
        tasks.append({
            "question": str(ex["question"]),
            "answer": str(ex["answer"]),
            "sessions": [_render_session(s) for s in ex["haystack_sessions"]],
            "qid": ex["question_id"],
        })
    return tasks[:max_examples] if max_examples else tasks


# --------------------------------------------------------------------------- #
# Training data: procedural moderate-length supersession episodes.
# The eval split (~48 sessions) is too long for affordable RL rollouts, so we
# train on conversational 6-8 session episodes carrying the same skill (track
# the latest value, ignore superseded ones), then validate on the real split.
# --------------------------------------------------------------------------- #
_ATTRS = {
    "city": {"intro": ["I just moved to {v}.", "I'm living in {v} these days.",
                       "I recently settled in {v}."],
             "update": ["Quick update — I relocated to {v} last month.",
                        "Change of plans, I live in {v} now.",
                        "I moved again; I'm in {v} now."],
             "query": "Which city do I currently live in?",
             "values": ["Boston", "Denver", "Seattle", "Austin", "Miami",
                        "Portland", "Chicago", "Dallas", "Atlanta", "Phoenix"]},
    "employer": {"intro": ["I just started a job at {v}.", "I work at {v} now.",
                          "I joined {v} recently."],
                 "update": ["I switched jobs — I'm at {v} now.",
                            "Update: I left for {v}.", "I now work at {v}."],
                 "query": "Where do I currently work?",
                 "values": ["Acme", "Globex", "Initech", "Hooli", "Umbrella",
                            "Soylent", "Wonka", "Cyberdyne", "Vehement", "Massive Dynamic"]},
    "car": {"intro": ["I just bought a {v}.", "I drive a {v} now.", "I got myself a {v}."],
            "update": ["I traded it in for a {v}.", "I switched to a {v} now.",
                       "New ride update: it's a {v} now."],
            "query": "What car do I currently drive?",
            "values": ["Toyota", "Subaru", "Volvo", "Mazda", "Honda", "Jeep",
                       "Kia", "Lexus", "Hyundai", "Nissan"]},
    "pet": {"intro": ["I adopted a {v}.", "I have a {v} now.", "I got a {v} recently."],
            "update": ["I rehomed the old one; my pet is a {v} now.",
                       "My current pet is a {v}.", "Update: my pet is now a {v}."],
            "query": "What pet do I currently have?",
            "values": ["beagle", "tabby cat", "parrot", "corgi", "rabbit",
                       "hamster", "husky", "goldfish", "gecko", "poodle"]},
}
_DISTRACTORS = [
    ("I'm thinking about picking up a new hobby.", "That sounds fun — what are you considering?"),
    ("The weather has been wild this week.", "Hopefully it settles down soon."),
    ("I tried a new recipe last night.", "Nice! How did it turn out?"),
    ("Work has been busy lately.", "Sounds intense — take breaks."),
    ("I started reading a new book.", "What's it about?"),
    ("I went for a long walk this morning.", "Great way to start the day."),
    ("I'm planning a trip later this year.", "Exciting — anywhere in particular?"),
    ("I reorganized my whole desk today.", "A tidy desk feels great."),
]
_ACK = ["Got it, noted.", "Thanks for the update!", "Understood.", "Good to know.", "Noted!"]


def _gen_train_episode(seed: int, min_sessions: int, max_sessions: int) -> dict:
    rng = random.Random(seed)
    a = _ATTRS[rng.choice(list(_ATTRS))]
    n = rng.randint(min_sessions, max_sessions)
    chain_len = rng.randint(2, 3)
    values = rng.sample(a["values"], chain_len)
    update_slots = sorted(rng.sample(range(1, n - 1), chain_len - 1))
    schedule = {0: values[0]}
    for slot, val in zip(update_slots, values[1:]):
        schedule[slot] = val
    sessions = []
    for i in range(n):
        if i in schedule:
            tmpl = a["intro"] if i == 0 else a["update"]
            turns = [{"role": "user", "content": rng.choice(tmpl).format(v=schedule[i])},
                     {"role": "assistant", "content": rng.choice(_ACK)}]
        else:
            u, asst = rng.choice(_DISTRACTORS)
            turns = [{"role": "user", "content": u}, {"role": "assistant", "content": asst}]
        sessions.append(_render_session(turns))
    return {"question": a["query"], "answer": values[-1], "sessions": sessions,
            "qid": f"train_{seed}", "stale_values": values[:-1]}


def _load_train_tasks(n: int, min_sessions: int, max_sessions: int,
                      start_seed: int = 100000) -> list[dict]:
    return [_gen_train_episode(start_seed + i, min_sessions, max_sessions) for i in range(n)]


# --------------------------------------------------------------------------- #
# Reward: does the answer convey the current (gold) value?
# --------------------------------------------------------------------------- #
_SPLIT = re.compile(r"\(or\b|\bor\b|/|;|,|\band\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def _gold_variants(gold: str) -> list[str]:
    out = []
    for piece in [gold] + _SPLIT.split(gold):
        n = _normalize(piece)
        if len(n) >= 2 and n not in out:
            out.append(n)
    return out


def answer_matches(answer: str, gold: str, overlap_threshold: float = 0.6) -> bool:
    a = _normalize(answer)
    if not a:
        return False
    for v in _gold_variants(gold):
        if v in a:
            return True
    g = set(_normalize(gold).split())
    if not g:
        return False
    return len(set(a.split()) & g) / len(g) >= overlap_threshold


# --------------------------------------------------------------------------- #
# Bounded-memory rollout state machine (framework-agnostic core).
# --------------------------------------------------------------------------- #
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
FULL_SYSTEM = (
    "The following is the full history of your past conversations with the "
    "user. Answer the user's question based on it. Be concise."
)


class _Rollout:
    def __init__(self, sessions, question, budget):
        self.sessions, self.question, self.budget = sessions, question, budget
        self.notes, self.sidx, self.phase, self.final_answer = "", 0, "memory", None

    @property
    def done(self):
        return self.phase == "done"

    def system_prompt(self):
        return ANSWER_SYSTEM if self.phase == "answer" else MEMORY_SYSTEM.format(budget=self.budget)

    def current_prompt(self):
        if self.phase == "memory":
            return (f"CURRENT NOTES:\n{self.notes or '(empty)'}\n\nNEW SESSION:\n"
                    f"{self.sessions[self.sidx]}\n\nRewrite your complete notes "
                    f"(max {self.budget} chars).")
        return f"NOTES:\n{self.notes}\n\nQuestion: {self.question}"

    def step(self, text):
        text = (text or "").strip()
        if self.phase == "memory":
            self.notes = text[: self.budget]
            self.sidx += 1
            if self.sidx >= len(self.sessions):
                self.phase = "answer"
            return self.current_prompt()
        self.final_answer = text
        self.phase = "done"
        return None


# --------------------------------------------------------------------------- #
# verifiers environment.
# --------------------------------------------------------------------------- #
def load_environment(
    mode: str = "eval",
    question_type: str = "knowledge-update",
    max_examples: int | None = None,
    budget: int = 300,
    max_turns: int = 60,
    full_context: bool = False,
    n_train: int = 2000,
    min_sessions: int = 6,
    max_sessions: int = 8,
):
    """Build the Supersede environment.

    Args:
        mode: ``"eval"`` (LongMemEval knowledge-update, the real benchmark) or
            ``"train"`` (unlimited procedural moderate-length episodes for RL).
        question_type: LongMemEval subset (eval mode).
        max_examples: cap on eval tasks.
        budget: character cap on the agent's notes memory (bounded mode).
        full_context: upper-bound condition (all sessions in context, single
            turn) instead of bounded memory. Eval only.
        n_train / min_sessions / max_sessions: training-episode controls.
    """
    import verifiers as vf
    from datasets import Dataset
    from verifiers.envs.multiturn_env import MultiTurnEnv
    from verifiers.envs.singleturn_env import SingleTurnEnv

    if mode == "train":
        tasks = _load_train_tasks(n_train, min_sessions, max_sessions)
        full_context = False  # training is always the bounded-memory agent
    else:
        tasks = _load_tasks(question_type, max_examples)

    def _last_assistant(completion):
        if isinstance(completion, str):
            return completion
        for m in reversed(completion or []):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role == "assistant":
                return (m.get("content") if isinstance(m, dict)
                        else getattr(m, "content", "")) or ""
        return ""

    async def answered_current(completion, answer, **kwargs) -> float:
        return 1.0 if answer_matches(_last_assistant(completion), answer or "") else 0.0

    rubric = vf.Rubric(funcs=[answered_current], weights=[1.0])

    if full_context:
        rows = []
        for t in tasks:
            blocks = "\n\n".join(f"[Session {i + 1}]\n{s}"
                                 for i, s in enumerate(t["sessions"]))
            rows.append({
                "prompt": [
                    {"role": "system", "content": FULL_SYSTEM},
                    {"role": "user",
                     "content": f"{blocks}\n\nQuestion: {t['question']}"},
                ],
                "answer": t["answer"],
            })
        return SingleTurnEnv(dataset=Dataset.from_list(rows), rubric=rubric)

    rows = [{
        "question": t["question"],
        "answer": t["answer"],
        "info": json.dumps({"sessions": t["sessions"], "question": t["question"],
                            "budget": budget, "qid": t["qid"]}),
    } for t in tasks]
    dataset = Dataset.from_list(rows)

    class SupersedeMemoryEnv(MultiTurnEnv):
        async def setup_state(self, state):
            # verifiers' rollout reassigns: `state = await self.setup_state(state)`,
            # so this MUST return state (returning None crashes downstream stops).
            info = state.get("info")
            info = json.loads(info) if isinstance(info, str) else (info or {})
            state["rollout"] = _Rollout(list(info["sessions"]),
                                        info.get("question", ""),
                                        int(info.get("budget", budget)))
            return state

        def _view(self, ro):
            return [{"role": "system", "content": ro.system_prompt()},
                    {"role": "user", "content": ro.current_prompt()}]

        async def get_prompt_messages(self, state):
            ro = state["rollout"]
            traj = state.get("trajectory") or []
            if not traj:
                return self._view(ro)
            prev = traj[-1]
            await self.env_response(list(prev["prompt"]) + list(prev["completion"]), state)
            if state.get("final_env_response") is not None:
                return list(prev["prompt"]) + list(prev["completion"])
            return self._view(ro)

        async def env_response(self, messages, state):
            ro = state["rollout"]
            last = ""
            for m in reversed(messages):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if role == "assistant":
                    last = (m.get("content") if isinstance(m, dict)
                            else getattr(m, "content", "")) or ""
                    break
            if ro.step(last) is None:
                state["final_env_response"] = []
                return []
            return [{"role": "user", "content": ro.current_prompt()}]

        @vf.stop
        async def rollout_done(self, state) -> bool:
            ro = state.get("rollout")
            return bool(ro and ro.done)

    return SupersedeMemoryEnv(dataset=dataset, rubric=rubric, max_turns=max_turns)


__all__ = ["load_environment", "answer_matches"]
