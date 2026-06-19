"""Lightweight local eval: play timelines against an OpenAI model.

This is the cheap path to the headline failure number. It does NOT depend on
the verifiers / prime-rl stack; it talks to the OpenAI Chat Completions API
directly. Each timeline is rendered as a multi-session conversation (one user
turn per session, neutral assistant acknowledgements between them) followed by
the final query, and scored with :mod:`supersede.scoring`.

Usage:
    python scripts/eval_local.py --n 30 --model gpt-4.1-mini \
        --chain-len 3 --distractors 4 --seed 0
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

SYSTEM_PROMPT = (
    "You are an assistant that tracks facts as they change over a long "
    "conversation. Facts stated later override earlier ones. When asked for a "
    "current value, answer using ONLY the most recent valid information and "
    "ignore any value that has since been superseded."
)


def build_messages(timeline) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for session in timeline.sessions:
        messages.append({"role": "user", "content": session})
        messages.append({"role": "assistant", "content": "Understood."})
    messages.append({"role": "user", "content": timeline.query})
    return messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--chain-len", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/eval_local.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    from openai import OpenAI

    client = OpenAI()
    cfg = TimelineConfig(chain_len=args.chain_len, n_distractors=args.distractors)
    timelines = generate_dataset(args.n, start_seed=args.seed, config=cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts = []
    with out_path.open("w") as fh:
        for i, tl in enumerate(timelines):
            resp = client.chat.completions.create(
                model=args.model,
                messages=build_messages(tl),
                temperature=0,
                max_tokens=32,
            )
            answer = (resp.choices[0].message.content or "").strip()
            verdict = classify(answer, tl.current_answer, tl.stale_answers)
            verdicts.append(verdict)
            fh.write(json.dumps({
                "seed": tl.seed,
                "subject": tl.subject,
                "predicate": tl.predicate,
                "current": tl.current_answer,
                "stale": tl.stale_answers,
                "answer": answer,
                "verdict": verdict,
            }) + "\n")
            print(f"[{i + 1:>3}/{args.n}] {verdict:<9} "
                  f"ans={answer!r} cur={tl.current_answer!r} stale={tl.stale_answers}")

    summary = summarize(verdicts)
    print("\n=== SUMMARY ===")
    print(f"model={args.model}  chain_len={args.chain_len}  distractors={args.distractors}")
    print(json.dumps(summary.as_dict(), indent=2))
    print(f"\naccuracy={summary.accuracy:.1%}  stale_rate={summary.stale_rate:.1%}")


if __name__ == "__main__":
    main()
