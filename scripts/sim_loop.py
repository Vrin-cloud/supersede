"""Offline simulation of the verifiers rollout loop (no API).

Drives the real SupersedeMemoryEnv methods through the same loop verifiers uses
(setup_state -> while not is_completed: get_prompt_messages -> [model] ->
add trajectory step), with a stub model, to verify the turn sequencing,
termination, and which prompt the model sees each turn.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from supersede.dataset import load_longmemeval  # noqa: E402
from supersede.env import load_environment  # noqa: E402


def _text(msgs):
    parts = []
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        parts.append((role, content))
    return parts


async def run():
    env = load_environment(data_path="data/longmemeval_oracle.json",
                           max_examples=1, budget=300, max_turns=20)
    task = load_longmemeval("data/longmemeval_oracle.json")[0]
    info = {"qid": "t0", "qtype": "knowledge-update", "stale_values": [],
            "sessions": task["sessions"], "budget": 300}
    state = {"trajectory": [], "info": json.dumps(info),
             "question": task["question"], "final_env_response": None}
    await env.setup_state(state)
    print(f"sessions={len(task['sessions'])}  question={task['question'][:60]!r}")
    print(f"gold={task['answer']!r}\n")

    turn = 0
    while turn < 25:
        ro = state["rollout"]
        if state.get("final_env_response") is not None:
            print(">> STOP: final_env_response set"); break
        if ro.done:
            print(">> STOP: rollout_done"); break
        msgs = await env.get_prompt_messages(state)
        if state.get("final_env_response") is not None:
            print(">> STOP: final_env_response after get_prompt_messages"); break
        roles = _text(msgs)
        user_content = next((c for r, c in roles if r == "user"), "")
        kind = "QUESTION" if user_content.startswith("NOTES:") and "Question:" in user_content else "memory"
        print(f"turn {turn}: phase={ro.phase} sidx={ro.sidx} kind={kind} "
              f"user_prompt[:80]={user_content[:80]!r}")
        # stub model: emit an answer when shown the question, else fake notes
        stub = "the answer is 25:50" if kind == "QUESTION" else f"notes@turn{turn}"
        state["trajectory"].append({
            "prompt": msgs,
            "completion": [{"role": "assistant", "content": stub}],
        })
        turn += 1

    print(f"\ntotal model turns={turn}  final_answer={state.get('final_answer')!r}")


if __name__ == "__main__":
    asyncio.run(run())
