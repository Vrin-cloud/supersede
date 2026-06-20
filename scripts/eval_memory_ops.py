"""Retrieval-memory agent: the realistic regime where supersession bites.

Instead of a full-rewrite notes field, the agent manages a discrete memory
store via operations, exactly like a production memory system (Mem0-style
ADD/UPDATE/DELETE). For each incoming fact the agent sees ONLY the new fact
plus the memories that lexically match its subject (a top-k retrieval), and
must emit operations to keep the store consistent. It never sees full history
and cannot rewrite the whole store at once.

We then inspect the FINAL store directly for the queried (subject, predicate):

- correct : store asserts the current value and no superseded value
- stale   : store still asserts a superseded value (contamination)
- missing : store asserts no value for the queried fact

This measures memory hygiene under updates directly, with no query-time
ambiguity. The hypothesis is that the agent will fail to delete superseded
entries, leaving stale contamination that a static (no-update) store never has.

Usage:
    python scripts/eval_memory_ops.py --n 20 --model gpt-4.1-mini \
        --tracked 8 --chain-len 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from supersede.scoring import normalize  # noqa: E402
from supersede.timeline import TimelineConfig, generate_dataset  # noqa: E402

OPS_SYSTEM = (
    "You maintain a long-term memory store of discrete facts. For each new "
    "piece of information you are shown the new fact and the existing memories "
    "that may relate to it. Your job is to keep the store CONSISTENT: it must "
    "never hold two conflicting values for the same thing, and it must never "
    "keep a value that has been superseded. Respond with ONLY a JSON object: "
    '{\"ops\": [...]} where each op is either '
    '{\"op\": \"add\", \"text\": \"<fact>\"} or '
    '{\"op\": \"delete\", \"id\": \"<memory id>\"}. '
    "When a new fact changes an existing one, delete the old memory AND add the "
    "new one. Output nothing but the JSON."
)


def subject_of(fact_text: str) -> str:
    return normalize(fact_text).split(" ")[0] if fact_text else ""


def retrieve(store: dict[str, str], subject: str) -> dict[str, str]:
    """Top-k by subject match (small stores: return all matching the subject)."""
    return {mid: txt for mid, txt in store.items() if subject_of(txt) == subject}


def parse_ops(content: str) -> list[dict]:
    content = content.strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    ops = obj.get("ops", [])
    return ops if isinstance(ops, list) else []


def apply_ops(store: dict[str, str], ops: list[dict], counter: list[int]) -> None:
    for op in ops:
        if not isinstance(op, dict):
            continue
        if op.get("op") == "delete":
            store.pop(str(op.get("id", "")), None)
        elif op.get("op") == "add" and op.get("text"):
            mid = f"m{counter[0]}"
            counter[0] += 1
            store[mid] = str(op["text"])


def classify_store(store: dict[str, str], subject: str, predicate: str,
                   current: str, stale: list[str]) -> str:
    subj_n, pred_n = normalize(subject), normalize(predicate)
    rel = [normalize(t) for t in store.values()
           if subj_n in normalize(t) and pred_n in normalize(t)]
    has_current = any(normalize(current) in t for t in rel)
    has_stale = any(any(normalize(s) in t for t in rel) for s in stale)
    if has_current and not has_stale:
        return "correct"
    if has_stale:
        return "stale"
    return "missing"


def run_one(client, model, sessions_facts, subject, predicate, current, stale):
    """sessions_facts: list of individual fact statements (the stream)."""
    store: dict[str, str] = {}
    counter = [0]
    for fact in sessions_facts:
        subj = subject_of(fact)
        related = retrieve(store, subj)
        related_str = ("\n".join(f"  {mid}: {txt}" for mid, txt in related.items())
                       or "  (none)")
        user = (f"NEW FACT:\n  {fact}\n\nEXISTING RELATED MEMORIES:\n{related_str}\n\n"
                f"Emit ops to keep the store consistent.")
        r = client.chat.completions.create(
            model=model, temperature=0, max_tokens=200,
            messages=[{"role": "system", "content": OPS_SYSTEM},
                      {"role": "user", "content": user}])
        apply_ops(store, parse_ops(r.choices[0].message.content or ""), counter)
    verdict = classify_store(store, subject, predicate, current, stale)
    return verdict, store


def flatten_facts(tl) -> list[str]:
    """Expand the timeline into a flat stream of individual fact statements."""
    facts: list[str] = []
    for session in tl.sessions:
        # session 0 packs several statements separated by ". "; split them.
        for piece in re.split(r"(?<=\.)\s+", session.strip()):
            piece = piece.strip()
            if piece:
                facts.append(piece)
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--tracked", type=int, default=8)
    ap.add_argument("--chain-len", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--out", default="results/eval_memory_ops.jsonl")
    args = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()

    # UPDATED condition: facts evolve (chain_len) ; STATIC control: chain_len 1.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"STATIC": {}, "UPDATED": {}}
    with out.open("w") as fh:
        for cond, clen in (("STATIC", 1), ("UPDATED", args.chain_len)):
            cfg = TimelineConfig(n_tracked=args.tracked, chain_len=clen,
                                 n_distractors=args.distractors)
            tls = generate_dataset(args.n, start_seed=7000, config=cfg)
            for tl in tls:
                facts = flatten_facts(tl)
                v, store = run_one(client, args.model, facts, tl.subject,
                                   tl.predicate, tl.current_answer, tl.stale_answers)
                results[cond][v] = results[cond].get(v, 0) + 1
                fh.write(json.dumps({
                    "cond": cond, "seed": tl.seed, "subj": tl.subject,
                    "pred": tl.predicate, "current": tl.current_answer,
                    "stale": tl.stale_answers, "verdict": v,
                    "store": list(store.values()),
                }) + "\n")

    def pct(d, k):
        return d.get(k, 0) / args.n
    print(f"model={args.model} tracked={args.tracked} chain={args.chain_len}\n")
    for cond in ("STATIC", "UPDATED"):
        c = results[cond]
        print(f"{cond:8} clean={pct(c,'correct'):.0%}  stale={pct(c,'stale'):.0%}  "
              f"missing={pct(c,'missing'):.0%}   {c}")


if __name__ == "__main__":
    main()
