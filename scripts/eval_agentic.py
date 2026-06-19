"""Agentic memory eval: bounded notes, no re-fed history.

Unlike ``eval_local.py`` (full transcript in context), here the agent never
sees past messages. Its ONLY carried state is a bounded ``notes`` string it
rewrites each session. The environment hard-caps the notes to a character
budget (lossy memory), so under many concurrently-evolving facts the agent
must compress and, crucially, *overwrite* superseded values. Stale values
that it fails to remove linger and surface at query time.

This is the setting where the failure actually lives. One model call per
session (the memory write) plus one for the final query.

Usage:
    python scripts/eval_agentic.py --n 15 --model gpt-4.1-mini \
        --tracked 8 --chain-len 3 --budget 240
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--tracked", type=int, default=8)
    ap.add_argument("--chain-len", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=4)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/eval_agentic.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    from openai import OpenAI

    client = OpenAI()
    cfg = TimelineConfig(n_tracked=args.tracked, chain_len=args.chain_len,
                         n_distractors=args.distractors)
    timelines = generate_dataset(args.n, start_seed=args.seed, config=cfg)
    write_system = WRITE_SYSTEM.format(budget=args.budget)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts = []
    with out_path.open("w") as fh:
        for idx, tl in enumerate(timelines):
            notes = ""
            for session in tl.sessions:
                user = (
                    f"CURRENT NOTES:\n{notes or '(empty)'}\n\n"
                    f"NEW UPDATE:\n{session}\n\n"
                    f"Rewrite your complete notes (max {args.budget} chars)."
                )
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": write_system},
                              {"role": "user", "content": user}],
                    temperature=0,
                    max_tokens=400,
                )
                notes = (resp.choices[0].message.content or "").strip()
                notes = notes[: args.budget]  # hard cap: lossy memory

            q = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": QUERY_SYSTEM},
                          {"role": "user",
                           "content": f"NOTES:\n{notes}\n\n{tl.query}"}],
                temperature=0,
                max_tokens=32,
            )
            answer = (q.choices[0].message.content or "").strip()
            verdict = classify(answer, tl.current_answer, tl.stale_answers)
            verdicts.append(verdict)
            fh.write(json.dumps({
                "seed": tl.seed, "subject": tl.subject, "predicate": tl.predicate,
                "current": tl.current_answer, "stale": tl.stale_answers,
                "final_notes": notes, "answer": answer, "verdict": verdict,
                "sessions": len(tl.sessions),
            }) + "\n")
            print(f"[{idx + 1:>3}/{args.n}] {verdict:<9} ans={answer!r} "
                  f"cur={tl.current_answer!r} stale={tl.stale_answers}")

    summary = summarize(verdicts)
    print("\n=== SUMMARY (agentic, bounded notes) ===")
    print(f"model={args.model} tracked={args.tracked} chain={args.chain_len} "
          f"budget={args.budget}")
    print(json.dumps(summary.as_dict(), indent=2))
    print(f"\naccuracy={summary.accuracy:.1%}  stale_rate={summary.stale_rate:.1%}  "
          f"wrong_rate={summary.wrong_rate:.1%}")


if __name__ == "__main__":
    main()
