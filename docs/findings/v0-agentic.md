# v0 finding: supersession breaks bounded-memory agents

**Setup.** Same fact-mutation timelines as the in-context study, but the agent
never sees past messages. Its only carried state is a notes string it rewrites
each session, hard-capped to a character budget (lossy memory). One model call
per session (the memory write) plus one for the final query. Reader:
`gpt-4.1-mini`, temperature 0. See `scripts/eval_agentic.py`.

## The isolation experiment

Memory load is held fixed (10 tracked facts + 3 distractors, budget 200
chars, identical query). The *only* variable is whether the tracked facts
were updated over the conversation.

| Condition | n | Accuracy | Stale-rate | Wrong-rate |
| --- | --- | --- | --- | --- |
| **Static** (facts never change) | 20 | 90% | 0% | 10% |
| **Updated** (same facts, each superseded twice) | 20 | 75% | 5% | 20% |

At equal memory load, updating facts costs ~15 points of accuracy and
introduces genuine stale errors (the agent reports a value that was later
superseded). This isolates **supersession**, not raw capacity, as a distinct
failure cause: the static condition shows the model *can* hold these facts in
200 chars; it fails specifically when they change.

## Context

| Setting | Accuracy |
| --- | --- |
| In-context (full history visible) | 100% |
| Agentic, bounded memory, static facts | 90% |
| Agentic, bounded memory, updated facts | 75% |

The effect lives entirely in the bounded-memory agentic regime, which is the
setting RL-trained memory agents actually operate in.

> Caveats: `gpt-4.1-mini` only (a frontier-model confirmation is pending);
> n=20 (CIs are wide, ~±10pp); the update-count curve (U = 0..3 at fixed load)
> is the definitive figure and is being measured.
