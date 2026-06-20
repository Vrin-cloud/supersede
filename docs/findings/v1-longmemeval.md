# v1 finding: the failure is real on LongMemEval (knowledge-update)

After three synthetic mechanisms all scored ~100% (see `v0-agentic.md`), we
moved to real data. **LongMemEval** (Wu et al., ICLR 2025) tags 78 questions as
`knowledge-update`: a fact stated in one session is changed in a later one, and
the correct answer is the current value. Real conversational text, genuine
supersession. MIT-licensed; oracle variant (evidence sessions only).

Reader: `gpt-4.1-mini`, temperature 0. Two conditions: **fullcontext** (all
evidence sessions in context) and **notes** (bounded 300-char memory rewritten
session-by-session, raw sessions never re-shown). LLM-judge grading.

## Preliminary (n=8)

| Condition | Accuracy |
| --- | --- |
| Full-context (upper bound) | 75% (6/8) |
| Bounded-memory agent | 62% (5/8) |

Two things the synthetic study never showed:

1. **The task is hard even in-context** (75%, not 100%): real updates are
   implicit and conversational, so even with everything visible the model
   sometimes returns the stale value.
2. **Bounded memory is worse** (62%): compressing sessions session-by-session
   loses or corrupts the updated fact.

n=8 has a wide CI; the full 78-question run is in progress. But this is already
qualitatively different from the synthetic 100% and confirms the gap exists on
trusted data.

> Next: full n=78 on oracle, then the harder `_s` variant (~50 distractor
> sessions) where memory pressure is real and the fullcontext/notes gap should
> widen. Then a frontier-model row to show the gap is not just a small-model
> artifact.
