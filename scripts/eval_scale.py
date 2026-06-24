"""Scale experiment: does growing history degrade supersession at fixed /
proportional memory? (oracle vs _s on LongMemEval knowledge-update.)

Bounded-notes agent, one session at a time, never re-feeding raw sessions.
Graded by the programmatic matcher (consistent across conditions, no judge
cost). Budget is either fixed (--budget) or scaled per-question to a constant
chars-per-session ratio (--per-session), which holds the compression ratio
roughly constant as history grows.

    python scripts/eval_scale.py --variant oracle --budget 300 --n 25
    python scripts/eval_scale.py --variant s --budget 300 --n 25
    python scripts/eval_scale.py --variant s --per-session 150 --n 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:  # Hub single-module env exposes answer_matches at top level
    from supersede import answer_matches  # noqa: E402
except ImportError:  # research library layout (src/supersede package)
    from supersede.reward import answer_matches  # noqa: E402

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


def session_text(session):
    return "\n".join(f"{t['role']}: {t['content']}" for t in session)


def run_one(client, model, sessions, question, budget):
    notes = ""
    sysmsg = MEMORY_SYSTEM.format(budget=budget)
    out_tok = max(64, budget // 3)
    for s in sessions:
        user = (f"CURRENT NOTES:\n{notes or '(empty)'}\n\nNEW SESSION:\n"
                f"{session_text(s)}\n\nRewrite your complete notes (max {budget} chars).")
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=out_tok,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": user}])
        notes = (r.choices[0].message.content or "").strip()[:budget]
    a = client.chat.completions.create(
        model=model, temperature=0, max_tokens=64,
        messages=[{"role": "system", "content": ANSWER_SYSTEM},
                  {"role": "user", "content": f"NOTES:\n{notes}\n\nQuestion: {question}"}])
    return (a.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["oracle", "s"], default="oracle")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--per-session", type=int, default=0,
                    help="if set, budget = per_session * n_sessions (constant ratio)")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--base-url", default="",
                    help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1 for a local vLLM server")
    ap.add_argument("--api-key", default="",
                    help="api key (any string for a local vLLM server)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from openai import OpenAI

    fn = ("data/longmemeval_oracle.json" if args.variant == "oracle"
          else "data/longmemeval_s.json")
    data = [x for x in json.load(open(fn)) if x["question_type"] == "knowledge-update"]
    data = data[: args.n]
    if args.base_url:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key or "local")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set")
        client = OpenAI()
    out = Path(args.out or f"results/scale_{args.variant}_"
               f"{'ps' + str(args.per_session) if args.per_session else 'b' + str(args.budget)}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    t0 = time.time()
    with out.open("w") as fh:
        for i, ex in enumerate(data):
            sessions = ex["haystack_sessions"]
            budget = args.per_session * len(sessions) if args.per_session else args.budget
            ans = run_one(client, args.model, sessions, ex["question"], budget)
            ok = answer_matches(ans, str(ex["answer"]))
            correct += ok
            fh.write(json.dumps({"qid": ex["question_id"], "sessions": len(sessions),
                                 "budget": budget, "gold": ex["answer"],
                                 "answer": ans, "correct": bool(ok)}) + "\n")
            fh.flush()
            print(f"[{i+1}/{len(data)}] sess={len(sessions)} budget={budget} "
                  f"running_acc={correct}/{i+1} {'OK ' if ok else 'x  '} {ans[:45]!r}",
                  flush=True)

    dt = time.time() - t0
    print(f"\n=== {args.variant} budget="
          f"{'per-session ' + str(args.per_session) if args.per_session else args.budget} ===")
    print(f"accuracy = {correct}/{len(data)} = {correct/len(data):.1%}   ({dt:.0f}s)")


if __name__ == "__main__":
    main()
