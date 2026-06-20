# v1 finding: the failure is real on LongMemEval (knowledge-update)

After three synthetic mechanisms all scored ~100% (see `v0-agentic.md`), we
moved to real data. **LongMemEval** (Wu et al., ICLR 2025) tags 78 questions as
`knowledge-update`: a fact stated in one session is changed in a later one, and
the correct answer is the current value. Real conversational text, genuine
supersession. MIT-licensed; oracle variant (evidence sessions only).

Reader: `gpt-4.1-mini`, temperature 0. Two conditions: **fullcontext** (all
evidence sessions in context) and **notes** (bounded 300-char memory rewritten
session-by-session, raw sessions never re-shown). LLM-judge grading.

## Result (n=78, full knowledge-update subset)

| Condition | Accuracy |
| --- | --- |
| Full-context (upper bound) | **82%** (64/78) |
| Bounded-memory agent (300-char notes) | **63%** (49/78) |

Paired analysis (same questions, both conditions):

- Full-context correct but memory wrong: **19**
- Memory correct but full-context wrong: **4**
- Both wrong (task hard even in-context): **10 (13%)**
- McNemar chi-square = 8.52, **p = 0.0035** -> the memory degradation is
  statistically significant.

Two things the synthetic study never showed:

1. **The task is hard even in-context** (82%, not 100%): real updates are
   implicit and conversational, so even with everything visible the model
   sometimes returns the stale value.
2. **Bounded memory roughly doubles the error** (18% -> 37%): compressing
   sessions session-by-session loses or corrupts the updated fact. Example:
   "How many Korean restaurants have I tried?" gold *four*; the memory agent
   answers *"you haven't mentioned any"* -- the fact was dropped during
   compression.

### Caveats

- LLM-judge grading introduces some noise: a few of the 19 memory failures are
  judge strictness ("over 25" graded wrong vs gold "25"). p=0.0035 survives a
  handful of flips, but a cleaner judge / manual audit of the 19 would tighten
  the number.
- This is the **oracle** variant (evidence sessions only). The realistic `_s`
  variant (~50 distractor sessions) should lower both numbers and widen the
  gap.
- `gpt-4.1-mini` only; a frontier-model row is needed to show the gap is not a
  small-model artifact.

> Bottom line: the supersession-memory failure is real and significant on a
> trusted benchmark. This is the empirical foundation for the environment
> (build the task around knowledge-update) and the reward (penalize stale
> answers during memory maintenance).
