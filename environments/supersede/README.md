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
# bounded memory (the failure regime)
prime eval run supersede -m openai/gpt-4.1-mini -a '{"max_examples": 78}'
# full-context upper bound (for the gap)
prime eval run supersede -m openai/gpt-4.1-mini -a '{"full_context": true}'
```

The environment auto-downloads the LongMemEval knowledge-update data
(MIT license) on first run. Arguments to `load_environment`:

| arg | default | meaning |
| --- | --- | --- |
| `question_type` | `knowledge-update` | LongMemEval subset |
| `max_examples` | `None` | cap on tasks |
| `budget` | `300` | character cap on the agent's notes memory (bounded mode) |
| `full_context` | `False` | upper-bound mode: all sessions in context, single turn |

## Reward

- `answered_current` (+1): the final answer conveys the current/gold value
  (programmatic, ungameable matcher; no API needed).
- `stale_penalty` (-1): the answer asserts a known superseded value — active
  only when the task ships `stale_values` (synthetic timelines; LongMemEval is
  gold-only).

## Status

Validated end-to-end under `verifiers` 0.1.14 against OpenAI: all 78
knowledge-update rollouts terminate cleanly and the environment reports
**57.7%** accuracy for gpt-4.1-mini (programmatic matcher), consistent with the
offline harness's 63% (LLM judge). The remaining step is the Hub push
(`prime env push`, which authenticates under your Prime Intellect account).
