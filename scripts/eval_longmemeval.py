"""Ground the supersession question on LongMemEval's knowledge-update subset.

LongMemEval (Wu et al., ICLR 2025) tags 78 of its 500 questions as
``knowledge-update``: a fact is stated in one session and changed in a later
one, and the correct answer is the current value. This is real conversational
data with genuine supersession, unlike our clean synthetic timelines.

Two conditions on the same questions:

- FULLCONTEXT: every evidence session is placed in the model's context, then
  the question is asked. Upper bound (memory is not the bottleneck).
- NOTES: a bounded-memory agent sees one session at a time and rewrites a
  capped notes field; it never re-sees raw sessions. At query time it answers
  only from its notes. This is the realistic memory-agent setting.

A gap (FULLCONTEXT high, NOTES low) localizes the failure to memory
maintenance under updates. Answers are graded by an LLM judge, following the
LongMemEval protocol (judge checks whether the current/gold answer is present).

Usage:
    python scripts/eval_longmemeval.py --n 20 --model gpt-4.1-mini \
        --budget 300 --data data/longmemeval_oracle.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

NOTES_SYSTEM = (
    "You maintain a NOTES field that is your ONLY memory of a long, ongoing "
    "conversation with a user. You will never see earlier sessions again, only "
    "your notes. Each turn you receive your current notes and the transcript of "
    "one new session. Rewrite your COMPLETE notes to capture everything about "
    "the user that may matter later. When new information changes something you "
    "already noted, OVERWRITE it; never keep outdated facts. Your notes are "
    "hard-capped at {budget} characters. Output ONLY the new notes."
)
ANSWER_FROM_NOTES = (
    "These notes are your entire memory of the conversation. Answer the user's "
    "question using only the notes. Be concise."
)
ANSWER_FULL = (
    "The following is the full history of your past conversations with the "
    "user. Answer the user's question based on it. Be concise."
)
JUDGE = (
    "You are grading whether a model answer is correct.\n"
    "Question: {q}\nGold answer: {gold}\nModel answer: {ans}\n\n"
    "Is the model answer correct, i.e. does it convey the gold answer (the "
    "current/updated value)? Minor wording differences are fine. Reply with "
    "exactly 'yes' or 'no'."
)


def session_text(session: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in session)


def run_notes(client, model, sessions, question, budget):
    notes = ""
    sysmsg = NOTES_SYSTEM.format(budget=budget)
    for s in sessions:
        user = (f"CURRENT NOTES:\n{notes or '(empty)'}\n\nNEW SESSION:\n"
                f"{session_text(s)}\n\nRewrite your complete notes (max {budget} chars).")
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=500,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": user}])
        notes = (r.choices[0].message.content or "").strip()[:budget]
    a = client.chat.completions.create(
        model=model, temperature=0, max_tokens=80,
        messages=[{"role": "system", "content": ANSWER_FROM_NOTES},
                  {"role": "user", "content": f"NOTES:\n{notes}\n\nQuestion: {question}"}])
    return (a.choices[0].message.content or "").strip()


def run_fullcontext(client, model, sessions, question):
    blocks = "\n\n".join(f"[Session {i + 1}]\n{session_text(s)}"
                         for i, s in enumerate(sessions))
    a = client.chat.completions.create(
        model=model, temperature=0, max_tokens=80,
        messages=[{"role": "system", "content": ANSWER_FULL},
                  {"role": "user", "content": f"{blocks}\n\nQuestion: {question}"}])
    return (a.choices[0].message.content or "").strip()


def judge(client, model, q, gold, ans):
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=2,
        messages=[{"role": "user", "content": JUDGE.format(q=q, gold=gold, ans=ans)}])
    return (r.choices[0].message.content or "").strip().lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--data", default="data/longmemeval_oracle.json")
    ap.add_argument("--conditions", nargs="+", default=["fullcontext", "notes"])
    ap.add_argument("--out", default="results/longmemeval_ku.jsonl")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    data = json.load(open(args.data))
    ku = [x for x in data if x["question_type"] == "knowledge-update"][: args.n]
    print(f"knowledge-update questions: {len(ku)}  model={args.model} "
          f"budget={args.budget}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    correct = {c: 0 for c in args.conditions}
    with out.open("w") as fh:
        for ex in ku:
            sessions = ex["haystack_sessions"]
            q, gold = ex["question"], ex["answer"]
            rec = {"qid": ex["question_id"], "q": q, "gold": gold,
                   "n_sessions": len(sessions)}
            for cond in args.conditions:
                if cond == "fullcontext":
                    ans = run_fullcontext(client, args.model, sessions, q)
                else:
                    ans = run_notes(client, args.model, sessions, q, args.budget)
                ok = judge(client, args.judge_model, q, gold, ans)
                correct[cond] += ok
                rec[cond] = {"answer": ans, "correct": ok}
            fh.write(json.dumps(rec) + "\n")
            tag = "  ".join(f"{c}={'Y' if rec[c]['correct'] else 'n'}"
                            for c in args.conditions)
            print(f"[{tag}] {q[:70]}")

    print("\n=== ACCURACY (knowledge-update) ===")
    for c in args.conditions:
        print(f"  {c:12} {correct[c] / len(ku):.0%}  ({correct[c]}/{len(ku)})")


if __name__ == "__main__":
    main()
