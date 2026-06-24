<div align="center">

# Supersede

**An RL environment that trains LLM agents to use the *current* fact, not the *stale* one.**

[![CI](https://github.com/Vrin-cloud/supersede/actions/workflows/ci.yml/badge.svg)](https://github.com/Vrin-cloud/supersede/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Built on verifiers](https://img.shields.io/badge/built%20on-verifiers-8A2BE2)](https://github.com/willccbb/verifiers)

[Quickstart](#quickstart) · [Results](#results) · [How it works](#how-the-environment-works) · [Contributing](#contributing) · [Cite](#citation) · [Contact](#contact)

</div>

---

Across a long, multi-session interaction, facts change: a user moves, a price
updates, a policy is replaced. Current memory systems and long-context models
are good at *recalling* what they were told, and bad at *dropping* what is no
longer true — so an assistant keeps booking flights to your old city. We call
the correct handling of such updates **supersession**.

Benchmarks have started to *measure* this (FAMA, MemoryArena, MemoryAgentBench),
but they only ever **score a frozen model**. Supersede turns that measurement
into a **training reward**: a multi-session environment where facts are
superseded over time and the agent is rewarded for acting on the
currently-valid version and penalized for relying on a superseded one — with
turn-level credit, on the [`verifiers`](https://github.com/willccbb/verifiers) /
[`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) rails the field
already trains on.

To our knowledge it is the **first trainable environment whose verifiable reward
is temporal fact-currency** — and we use it to train the gap down, not just
measure it.

## Why it matters

- **For labs & memory products:** a ready-made RL environment + verifier for a
  known, unsolved production failure (assistants that cite your *old* job,
  address, or preference).
- **For research:** the first work, to our knowledge, to make
  supersession-correctness a *learning signal* rather than an eval-only metric.

## What's novel

| Prior work | What it does | What it does not |
| --- | --- | --- |
| FAMA / Memora | Metric for using current vs. stale memory | Eval only, frozen models |
| MemAgent | RL for memory agents | Rewards final answer only, not fact-currency |
| LongRLVR | Verifiable reward on evidence *relevance* | No notion of temporal validity |
| MemoryAgentBench | Has a conflict-resolution eval task | Eval only, not a training environment |

Supersede sits in the intersection none of them occupy: a **trainable
environment whose verifiable reward is temporal fact-currency.**

## Quickstart

```bash
# 1. install (offline core + dev tools; no API key needed)
uv venv && source .venv/bin/activate
uv pip install -e ".[env,dev]"

# 2. verify it works
pytest                       # -> 21 passed

# 3. see the temporal core decide a supersession, no model required
python - <<'PY'
from supersede import Fact, detect_conflict
old = Fact(subject="Alice", predicate="lives in", object="Boston")
new = Fact(subject="Alice", predicate="lives in", object="Denver")
print(detect_conflict(new, [old]).strategy)   # -> supersede
PY
```

Run the full environment against a model (needs an OpenAI key):

```bash
prime env install supersede
prime eval run supersede -m openai/gpt-4.1-mini -a '{"max_examples": 78}'
```

The environment auto-downloads the LongMemEval `knowledge-update` data (MIT
license) on first run.

## Results

**The problem — bounded memory breaks supersession, even at the frontier.**
On LongMemEval `knowledge-update` (n=78), swapping full context for a bounded,
self-maintained memory drops accuracy sharply, and the gap *survives* on the
strongest model:

| Model | Full context | Bounded memory |
| --- | --- | --- |
| gpt-4.1-mini | 82% | 63% |
| gpt-4.1 | 91% | 64% |
| gpt-5.4 | 92% | **77%** |

Even gpt-5.4 loses 15 points (paired McNemar *p* = 0.0033). The bottleneck is
memory *maintenance*, not comprehension — and it doesn't close with a bigger
model, or with a bigger memory ([see the paper](#citation) for the scale study).

**The fix — training closes part of the gap.** We can't fine-tune the
proprietary models above, so we train a small open model (Qwen2.5-3B) on this
environment with GRPO and evaluate on the *same, held-out* real questions. Its
accuracy nearly doubles, monotonically as it learns:

| Checkpoint | Held-out oracle accuracy |
| --- | --- |
| base (untrained) | 9.0% |
| GRPO step 150 | 12.8% |
| **GRPO step 175** | **16.7%** |

Trained on *synthetic* episodes, improving on *real* held-out conversations —
i.e. a learned skill, not memorization. It is a proof of mechanism on a small
model (still far from the full-context ceiling, and the curve was still rising
when training ran out of hard examples), not a finished policy.

## How the environment works

The agent sees one session at a time and maintains a capped notes memory; it
**never re-sees raw sessions**, then answers a query using the current value of
a fact that changed along the way.

**Reward**
- `answered_current` (**+1**): the final answer conveys the current/gold value
  (programmatic, ungameable matcher — no judge model needed).
- `stale_penalty` (**−1**): the answer asserts a known superseded value — active
  only when the task ships `stale_values` (synthetic timelines; LongMemEval is
  gold-only).

**`load_environment` arguments**

| arg | default | meaning |
| --- | --- | --- |
| `question_type` | `knowledge-update` | LongMemEval subset |
| `max_examples` | `None` | cap on tasks |
| `budget` | `300` | character cap on the agent's notes memory |
| `full_context` | `False` | upper-bound mode: all sessions in context, single turn |
| `mode` | `eval` | `eval` (real data) or `train` (procedural curriculum) |

## Repo layout

```
src/supersede/
  models.py      # bi-temporal Fact (subject, predicate, object, validity, supersession)
  temporal.py    # conflict detection + supersession logic
  timeline.py    # synthetic fact-mutation timeline generator
  rollout.py     # framework-agnostic bounded-memory rollout state machine
  reward.py      # answer matching + answered_current / stale_penalty rewards
  dataset.py     # LongMemEval + synthetic task loaders
  env.py         # verifiers MultiTurnEnv wrapper (load_environment)
environments/supersede/   # Environments Hub package (prime env push)
scripts/         # eval harnesses (LongMemEval, validation)
docs/findings/   # empirical results, with caveats
tests/           # offline tests (temporal, rollout, reward) — 21 passing
```

## Contributing

Contributions are welcome — bug reports, harder training episodes, new model
results, and reward refinements especially. See
[CONTRIBUTING.md](CONTRIBUTING.md); `good first issue`s are labeled in the
[issue tracker](https://github.com/Vrin-cloud/supersede/issues). Please run
`ruff check` and `pytest` before opening a PR.

## Citation

```bibtex
@misc{patel2026supersede,
  title  = {Supersede: Diagnosing and Training Memory-Consistent Agents under Fact Supersession},
  author = {Patel, Vedant},
  year   = {2026},
  note   = {Vrin. https://github.com/Vrin-cloud/supersede}
}
```

## Contact

- **Questions / ideas:** [GitHub Discussions](https://github.com/Vrin-cloud/supersede/discussions)
- **Bugs:** [open an issue](https://github.com/Vrin-cloud/supersede/issues)
- **Security:** see [SECURITY.md](SECURITY.md)
- **Email:** supersede@vrin.cloud · **Author:** Vedant Patel ([vedant@vrin.cloud](mailto:vedant@vrin.cloud), [vrin.cloud](https://vrin.cloud))
- **Social:** [X / Twitter](https://x.com/Vedant7129) · [LinkedIn](https://www.linkedin.com/in/vedant1033/)

## License & acknowledgements

Apache-2.0. Built by [Vrin](https://vrin.cloud) on
[`verifiers`](https://github.com/willccbb/verifiers), the
[Prime Intellect Environments Hub](https://www.primeintellect.ai/), and
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) — thanks to their
maintainers for the open tooling and data.
