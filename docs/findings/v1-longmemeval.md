# Finding: the failure is real on LongMemEval (knowledge-update)

Synthetic templated supersession is saturated (frontier models score ~100% with
full history in context), so the failure must be measured on real data.
**LongMemEval** (Wu et al., ICLR 2025) tags 78 questions as
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

## Holds on the frontier (gpt-5.4)

Same 78 questions, three models, two conditions:

| Model | Full-context | Bounded memory | Gap | Paired McNemar |
| --- | --- | --- | --- | --- |
| gpt-4.1-mini | 82% | 63% | 19 | p = 0.0035 |
| gpt-4.1 | 91% | 64% | 27 | — |
| **gpt-5.4 (frontier)** | **92%** | **77%** | **15** | **p = 0.0033** |

The bounded-memory gap is statistically significant on both the small model
and the frontier model. Scaling the model helps memory *somewhat* (63% -> 77%
across the family) but does **not** close the gap: gpt-5.4 still loses 15
points and fails ~23% of supersession questions under bounded memory, dropping
or garbling the updated fact (e.g. "Where did Rachel move to?" -> "no
information about Rachel"). Full-context accuracy, by contrast, saturates near
92%, so the bottleneck is memory maintenance, not reading comprehension.

The honest framing is therefore "scaling helps but leaves a significant,
frontier-level supersession failure," not "scaling does nothing."

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
