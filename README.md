# Supersede

**An RL environment that trains LLM agents to use the *current* fact, not the *stale* one.**

Across a long, multi-session interaction, facts change: a user moves, a price
updates, a policy is replaced. Current memory systems and long-context models
are good at *recalling* what they were told, and bad at *dropping* what is no
longer true. Benchmarks have started to measure this (FAMA, MemoryArena,
MemoryAgentBench), but they only ever *score a frozen model*.

Supersede turns that measurement into a **training reward**. It is a
multi-session environment where facts are superseded over time, and the agent
is rewarded for acting on the currently-valid version and penalized for
relying on a superseded one, with turn-level credit.

> Status: early scaffold. The bi-temporal supersession core is ported and
> tested; the environment, timeline generator, and reward are in progress.

## Why it matters

- **For labs:** a ready-made RL environment + verifier for a known, unsolved
  weakness, on the `verifiers` / `prime-rl` rails the field already trains on.
- **For research:** the first work (to our knowledge) to make
  supersession-correctness a *learning signal* rather than an eval metric.

## What's novel

| Prior work | What it does | What it does not |
| --- | --- | --- |
| FAMA / Memora | Metric for using current vs. stale memory | Eval only, frozen models |
| MemAgent | RL for memory agents | Rewards final answer only, not fact-currency |
| LongRLVR | Verifiable reward on evidence *relevance* | No notion of temporal validity |
| MemoryAgentBench | Has a conflict-resolution eval task | Eval only, not a training environment |

Supersede sits in the intersection none of them occupy: a **trainable
environment whose verifiable reward is temporal fact-currency**.

## Install (dev)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[env,dev]"
pytest
```

## Layout

```
src/supersede/
  models.py      # minimal bi-temporal Fact (subject, predicate, object, validity, supersession)
  temporal.py    # conflict detection + supersession (ported from Engram/Vrin)
  timeline.py    # synthetic fact-mutation timeline generator (ablation source)
  rollout.py     # framework-agnostic bounded-memory rollout state machine
  reward.py      # answer matching + answered_current / stale_penalty rewards
  dataset.py     # LongMemEval + synthetic task loaders
  env.py         # verifiers MultiTurnEnv wrapper (load_environment)
environments/supersede/   # Environments Hub package (prime env push)
scripts/         # eval harnesses (LongMemEval, agentic, sweeps)
docs/findings/   # the empirical results, with caveats
tests/           # offline tests (temporal, rollout, reward) -- 21 passing
```

## The result so far

On LongMemEval `knowledge-update`, bounded memory vs full context, n=78:

| Model | Full-context | Bounded memory |
| --- | --- | --- |
| gpt-4.1-mini | 82% | 63% |
| gpt-4.1 | 91% | 64% |
| gpt-5.4 | 92% | 77% |

Even the frontier model loses 15 points (paired McNemar p=0.0033); the gap is
memory maintenance, not comprehension. This environment exists to close it.

The temporal core is independent of the environment and can be imported on its
own:

```python
from supersede import Fact, detect_conflict

old = Fact(subject="Alice", predicate="lives in", object="Boston")
new = Fact(subject="Alice", predicate="lives in", object="Denver")
conflict = detect_conflict(new, [old])   # -> update / supersede
```

## License

Apache-2.0. Built by [Vrin](https://vrin.cloud).
