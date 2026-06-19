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

At first read this looked like supersession costing ~15 points. **The
update-count curve did not confirm it.**

## The update-count curve (the honest check)

Same fixed load (10 tracked + 3 distractors, budget 200), n=20 per point,
fresh timelines (seed 1000+), varying only U = number of updates per fact:

| Updates (U) | Accuracy | Stale-rate | Wrong-rate |
| --- | --- | --- | --- |
| 0 (static) | 85% | 0% | 15% |
| 1 | 80% | 0% | 15% |
| 2 | 85% | 0% | 5% |
| 3 | 80% | 0% | 5% |

**Flat.** Accuracy is ~80-85% regardless of how many times facts change, and
stale-rate is 0 throughout. The earlier 90-vs-75 gap was small-sample noise
(n=20, CI ~±10pp), not a real effect.

## Honest conclusion (negative result)

At this difficulty, **supersession is not a distinct failure cause** for
`gpt-4.1-mini`. The residual ~15-20% error is capacity/compression (the agent
occasionally drops a fact when squeezing into 200 chars), and it does not grow
with updates. Explicit `X now Y` updates over a compressible notes field are
too easy: the model overwrites correctly almost every time, so stale retention
essentially never happens.

## Why, and what it implies

The supersession failure the literature reports (FAMA, MemoryArena) is real,
but our v0 task makes supersession trivial in three ways: updates are explicit
and restate the predicate (a literal overwrite cue), the memory is free-form
(easy to edit), and the query is a direct lookup (not an action). To elicit a
genuine, supersession-specific failure the task must make *recognizing* the
supersession the hard part:

- **Implicit / indirect updates** that do not restate the predicate
  ("Maya relocated from Detroit to Miami"; "Maya took the role Bob just left").
- **Retraction / negation** ("disregard the earlier note about Maya's city").
- **Misattribution pressure**: distractor updates on the same predicate.
- **Action queries** rather than direct value lookups.

This is also the more novel and differentiated version, so the negative result
points straight at the contribution rather than away from it.

> Status: v0 simple task is a negative result for the supersession hypothesis.
> Next iteration implements indirect-update semantics and re-tests.
