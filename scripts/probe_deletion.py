"""Targeted probe: deletion/retraction vs replacement.

Hypothesis (grounded in FAMA): models overwrite a fact fine when given a new
value, but when a fact is *retracted with no replacement* ("X no longer ...")
they keep emitting the last-known (now stale) value instead of "unknown".

Two matched conditions over the same interfering multi-fact timeline and the
same bounded-notes agentic agent:

- REPLACE (control): the queried fact ends on a normal update; correct answer
  is the final value.
- DELETE: one extra session retracts the queried fact's current value with no
  replacement; correct answer is "unknown", and emitting any prior value is a
  stale error.

Usage:
    python scripts/probe_deletion.py --n 20 --model gpt-4.1-mini \
        --tracked 8 --chain-len 3 --budget 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from supersede.scoring import normalize  # noqa: E402
from supersede.timeline import TimelineConfig, generate_dataset  # noqa: E402

WRITE_SYSTEM = (
    "You maintain a NOTES field that is your ONLY memory. You will never see "
    "earlier messages again, only your notes. Each turn you receive your "
    "current notes and one new update. Rewrite your COMPLETE notes to reflect "
    "the latest state of the world. When an update changes OR retracts a fact, "
    "update your notes so they never assert something that is no longer true. "
    "Your notes are hard-capped at {budget} characters. Output ONLY the notes."
)
QUERY_SYSTEM = (
    "Answer using ONLY the notes provided; they are your entire memory. Reply "
    "with only the single current value. If the value is no longer known or was "
    "retracted, reply exactly 'unknown'."
)

UNKNOWN_MARKERS = [
    "unknown", "not known", "no longer", "not stated", "not specified",
    "n/a", "na", "none", "unspecified", "unclear", "not available", "retracted",
]


def classify_delete(answer: str, current_unknown: bool, current: str,
                    stale: list[str]) -> str:
    a = normalize(answer)
    says_unknown = any(m in a for m in UNKNOWN_MARKERS)
    names_stale = any(normalize(s) and normalize(s) in a for s in stale)
    if current_unknown:
        if says_unknown and not names_stale:
            return "correct"
        if names_stale:
            return "stale"
        return "wrong"
    # REPLACE control
    if normalize(current) in a and not any(
            normalize(s) in a for s in stale if normalize(s) != normalize(current)):
        return "correct"
    if names_stale:
        return "stale"
    return "wrong"


def run(client, model, sessions, query, budget):
    notes = ""
    sysmsg = WRITE_SYSTEM.format(budget=budget)
    for s in sessions:
        user = (f"CURRENT NOTES:\n{notes or '(empty)'}\n\nNEW UPDATE:\n{s}\n\n"
                f"Rewrite your complete notes (max {budget} chars).")
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=400,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": user}])
        notes = (r.choices[0].message.content or "").strip()[:budget]
    q = client.chat.completions.create(
        model=model, temperature=0, max_tokens=16,
        messages=[{"role": "system", "content": QUERY_SYSTEM},
                  {"role": "user", "content": f"NOTES:\n{notes}\n\n{query}"}])
    return (q.choices[0].message.content or "").strip(), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--tracked", type=int, default=8)
    ap.add_argument("--chain-len", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--out", default="results/probe_deletion.jsonl")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    cfg = TimelineConfig(n_tracked=args.tracked, chain_len=args.chain_len,
                         n_distractors=args.distractors)
    tls = generate_dataset(args.n, start_seed=5000, config=cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"REPLACE": {}, "DELETE": {}}
    with out.open("w") as fh:
        for tl in tls:
            subj, pred = tl.subject, tl.predicate
            base_query = (f"As of now, what is the value for: '{subj} {pred} "
                          f"___'? Reply with only the current value.")

            # REPLACE control: timeline as-is, answer = final value.
            ans, notes = run(client, args.model, tl.sessions, base_query, args.budget)
            v = classify_delete(ans, False, tl.current_answer, tl.stale_answers)
            counts["REPLACE"][v] = counts["REPLACE"].get(v, 0) + 1
            fh.write(json.dumps({"cond": "REPLACE", "seed": tl.seed, "subj": subj,
                                 "pred": pred, "current": tl.current_answer,
                                 "answer": ans, "verdict": v}) + "\n")

            # DELETE: append a retraction of the current value; answer = unknown.
            retraction = f"{subj} no longer {pred} {tl.current_answer}."
            del_sessions = list(tl.sessions) + [retraction]
            all_stale = tl.stale_answers + [tl.current_answer]
            ans2, notes2 = run(client, args.model, del_sessions, base_query, args.budget)
            v2 = classify_delete(ans2, True, "", all_stale)
            counts["DELETE"][v2] = counts["DELETE"].get(v2, 0) + 1
            fh.write(json.dumps({"cond": "DELETE", "seed": tl.seed, "subj": subj,
                                 "pred": pred, "retracted": tl.current_answer,
                                 "answer": ans2, "verdict": v2}) + "\n")

    def pct(d, k):
        return d.get(k, 0) / args.n
    for cond in ("REPLACE", "DELETE"):
        c = counts[cond]
        print(f"{cond:8} acc={pct(c,'correct'):.0%}  stale={pct(c,'stale'):.0%}  "
              f"wrong={pct(c,'wrong'):.0%}   {c}")


if __name__ == "__main__":
    main()
