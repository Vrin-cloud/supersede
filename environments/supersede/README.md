# supersede

*An RL environment that trains LLM agents to use the **current** fact, not the
**stale** one, across long multi-session interactions.*
*by **Vedant Patel** · [vedant@vrin.cloud](mailto:vedant@vrin.cloud) · [GitHub](https://github.com/Vrin-cloud/supersede)*

### Overview
- **Environment ID**: `supersede`
- **Short description**: A bounded-memory, multi-session environment where facts
  are superseded over time. The agent sees one session at a time and maintains a
  capped notes memory — it **never re-sees raw sessions** — then answers a query
  using the *current* value of a fact that changed along the way. The reward is
  **temporal fact-currency**, which (to our knowledge) no other trainable
  environment targets.
- **Tags**: `multi-turn`, `memory`, `long-horizon`, `qa`, `train`, `eval`

### Datasets
- **Primary dataset**: LongMemEval `knowledge-update` subset (real conversational
  supersession) — [repo](https://github.com/xiaowu0162/LongMemEval), MIT license,
  auto-downloaded on first run.
- **Splits**: `oracle` (~2 evidence sessions/question, **n=78**) and `_s`
  (~48 sessions, ~122k tokens/question). With `mode="train"` the environment
  instead generates unlimited procedural supersession episodes for RL.

### Task
- **Type**: multi-turn, bounded-memory rollout. Each turn the agent rewrites its
  capped notes from the current session; raw history is never re-fed. After the
  final session it answers from memory alone. `full_context=True` exposes a
  single-turn upper bound (all sessions in context).
- **Parser**: last assistant message, scored by a programmatic answer matcher
  (normalized variant match + token-overlap fallback) — no judge model required.
- **Rubric overview**: `reward = answered_current` (weight 1.0): 1.0 iff the
  final answer conveys the current/gold value. Procedural `train` tasks also ship
  `stale_values` (the superseded answers), enabling a stale-answer penalty.

### Quickstart
```bash
# local dev
uv run vf-eval supersede

# via the Hub
prime env install supersede
prime eval run supersede -m openai/gpt-4.1-mini -a '{"max_examples": 78}'
# full-context upper bound (for the gap)
prime eval run supersede -m openai/gpt-4.1-mini -a '{"full_context": true}'
```

The LongMemEval `knowledge-update` data downloads automatically on first run.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | str | `"eval"` | `eval` (real LongMemEval) or `train` (procedural RL episodes) |
| `question_type` | str | `"knowledge-update"` | LongMemEval subset (eval mode) |
| `max_examples` | int \| None | `None` | cap on eval tasks |
| `budget` | int | `300` | character cap on the agent's notes memory |
| `full_context` | bool | `False` | upper-bound mode: all sessions in context, single turn |
| `min_sessions` / `max_sessions` | int | `6` / `8` | session-count range (train mode) |

### Metrics
| Metric | Meaning |
| --- | --- |
| `reward` | aggregate reward = `answered_current` (weight 1.0) |
| `answered_current` | 1.0 iff the final answer conveys the current/gold value |

### Results
**The gap — bounded memory breaks supersession, even at the frontier**
(oracle, n=78):

| Model | Full context | Bounded memory |
| --- | --- | --- |
| gpt-4.1-mini | 82% | 63% |
| gpt-4.1 | 91% | 64% |
| gpt-5.4 | 92% | **77%** |

Even gpt-5.4 loses 15 points (paired McNemar *p* = 0.0033); the bottleneck is
memory *maintenance*, not comprehension, and it closes with neither a bigger
model nor a bigger memory.
Repro: `prime eval run supersede -m openai/gpt-4.1-mini -a '{"max_examples": 78}'`

**Training closes part of it** — Qwen2.5-3B trained on this environment with GRPO
(`mode="train"`), evaluated on the *same, held-out* real questions
(`mode="eval"`, oracle n=78):

| Checkpoint | Held-out accuracy |
| --- | --- |
| base (untrained) | 9.0% |
| GRPO step 175 | **16.7%** |

Trained on synthetic episodes, improving on real held-out conversations — a
learned skill, not memorization. Full diagnosis, scale study, and paper in the
[GitHub repo](https://github.com/Vrin-cloud/supersede).

### Acknowledgements & Citation
Built on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al.) and
[`verifiers`](https://github.com/willccbb/verifiers). Apache-2.0.

```bibtex
@misc{patel2026supersede,
  title  = {Supersede: Diagnosing and Training the Memory-Update Gap in LLM Agents},
  author = {Patel, Vedant},
  year   = {2026},
  note   = {Vrin. https://github.com/Vrin-cloud/supersede}
}
```

## Evaluation Reports

Reproduced with this environment (default bounded-memory mode):

| Model | Mode | n × r | Mean reward (`answered_current`) |
| --- | --- | --- | --- |
| gpt-4.1-mini | bounded memory | 20 × 3 | **0.633** |

Repro: `uv run vf-eval supersede -m gpt-4.1-mini -n 20 -r 3` — consistent with the
63% bounded-memory figure in [Results](#results). Future saved eval runs render here.
