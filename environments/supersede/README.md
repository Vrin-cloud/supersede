# supersede

**Train and evaluate agents to use the *current* fact, not the *stale* one.**

A bounded-memory environment over multi-session interactions: the agent sees one
session at a time and maintains a capped notes memory (it never re-sees raw
sessions), then must answer a question using the current value of a fact that
was updated along the way.

## The failure it targets

On LongMemEval's `knowledge-update` questions, giving an agent bounded memory
instead of full context drops supersession accuracy sharply — and the gap
survives on the frontier model:

| Model | Full-context | Bounded memory |
| --- | --- | --- |
| gpt-4.1-mini | 82% | 63% |
| gpt-4.1 | 91% | 64% |
| gpt-5.4 | 92% | **77%** |

Even gpt-5.4 loses 15 points (paired McNemar p=0.0033) and fails ~23% of
supersession questions under bounded memory, while full-context saturates near
92%. The bottleneck is memory maintenance, not comprehension. (Details:
`docs/findings/` in the repo.)

## Usage

```bash
prime env install supersede
prime eval run supersede -m openai/gpt-4.1-mini \
  -a '{"data_path": "data/longmemeval_oracle.json", "max_examples": 78}'
```

Arguments to `load_environment`:

| arg | default | meaning |
| --- | --- | --- |
| `data_path` | `None` | LongMemEval json; if omitted, uses synthetic timelines |
| `question_type` | `knowledge-update` | LongMemEval subset |
| `max_examples` | `None` | cap on tasks |
| `budget` | `300` | character cap on the agent's notes memory |
| `judge_model` | `None` | grade with an LLM judge instead of the programmatic matcher |

## Reward

- `answered_current` (+1): the final answer conveys the current/gold value
  (programmatic, ungameable matcher; no API needed).
- `stale_penalty` (-1): the answer asserts a known superseded value — active
  only when the task ships `stale_values` (synthetic timelines; LongMemEval is
  gold-only).

## Status

The memory rollout state machine and reward are unit-tested offline against
real LongMemEval data. The verifiers binding (`supersede.env`) follows the
documented `MultiTurnEnv` API; end-to-end validation under `prime eval run`
(and the Hub push) is the remaining step.
