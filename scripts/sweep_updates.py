"""Update-count curve: accuracy vs how many times facts are superseded.

Holds memory load fixed (same number of tracked facts, same character budget,
same distractors) and varies only the number of updates each tracked fact
receives (U = chain_len - 1). U=0 is the static control. A monotone fall in
accuracy as U grows isolates supersession (not capacity) as the cause.

Reuses the bounded-notes agentic rollout from eval_agentic.

Usage:
    python scripts/sweep_updates.py --n 20 --model gpt-4.1-mini \
        --tracked 10 --distractors 3 --budget 200 --updates 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from supersede.scoring import classify, summarize  # noqa: E402
from supersede.timeline import TimelineConfig, generate_dataset  # noqa: E402

WRITE_SYSTEM = (
    "You maintain a NOTES field that is your ONLY memory. You will never see "
    "earlier messages again, only your notes. Each turn you receive your "
    "current notes and one new update. Rewrite your COMPLETE notes to reflect "
    "the latest state of the world. When an update changes a fact, OVERWRITE "
    "the old value; do not keep outdated information. Your notes are hard-"
    "capped at {budget} characters, so be concise. Output ONLY the new notes."
)
QUERY_SYSTEM = (
    "Answer using ONLY the notes provided. They are your entire memory. "
    "Reply with only the single current value, nothing else."
)


def run_timeline(client, model, tl, budget):
    notes = ""
    sysmsg = WRITE_SYSTEM.format(budget=budget)
    for session in tl.sessions:
        user = (f"CURRENT NOTES:\n{notes or '(empty)'}\n\nNEW UPDATE:\n{session}"
                f"\n\nRewrite your complete notes (max {budget} chars).")
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=400,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": user}])
        notes = (r.choices[0].message.content or "").strip()[:budget]
    q = client.chat.completions.create(
        model=model, temperature=0, max_tokens=32,
        messages=[{"role": "system", "content": QUERY_SYSTEM},
                  {"role": "user", "content": f"NOTES:\n{notes}\n\n{tl.query}"}])
    answer = (q.choices[0].message.content or "").strip()
    return classify(answer, tl.current_answer, tl.stale_answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--tracked", type=int, default=10)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--updates", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default="results/sweep_updates.json")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    rows = []
    for u in args.updates:
        cfg = TimelineConfig(n_tracked=args.tracked, chain_len=u + 1,
                             n_distractors=args.distractors)
        tls = generate_dataset(args.n, start_seed=1000, config=cfg)
        verdicts = [run_timeline(client, args.model, tl, args.budget) for tl in tls]
        s = summarize(verdicts)
        row = {"updates": u, **s.as_dict()}
        rows.append(row)
        print(f"U={u}  acc={s.accuracy:.1%}  stale={s.stale_rate:.1%}  "
              f"wrong={s.wrong_rate:.1%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "tracked": args.tracked,
         "distractors": args.distractors, "budget": args.budget, "n": args.n,
         "rows": rows}, indent=2))

    print("\n| updates | accuracy | stale | wrong |")
    print("| --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['updates']} | {r['accuracy']:.0%} | {r['stale_rate']:.0%} "
              f"| {r['wrong_rate']:.0%} |")


if __name__ == "__main__":
    main()
