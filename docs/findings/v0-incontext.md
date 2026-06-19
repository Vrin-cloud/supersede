# v0 finding: in-context supersession is solved

**Setup.** Multi-session fact-mutation timelines (several facts evolving
concurrently, queried fact's final update buried mid-history, stable
distractors), rendered as a single conversation with the full history in
context. Reader: `gpt-4.1-mini`, temperature 0. Scoring: current vs. stale
vs. wrong (see `supersede.scoring`).

**Result.**

| config | n | accuracy | stale-rate |
| --- | --- | --- | --- |
| 6 tracked, chain 3, 6 distractors (13 sessions) | 30 | 100% | 0% |
| 12 tracked, chain 4, 10 distractors (38 sessions) | 20 | 100% | 0% |

**Why.** When the entire history is in the context window, recovering the
current value is a literal last-mention scan ("find the last `X now <value>`").
Frontier models do this perfectly regardless of interference or horizon. This
is a retrieval task, not a memory task. It matches the MemoryArena observation
that recall benchmarks overstate capability: the gap only appears when memory
must be *maintained* and *drive action*, not when it can be re-read.

**Implication.** The dramatic, defensible failure lives in the **agentic
memory-tool** setting: the agent maintains a bounded external memory it must
update, the full transcript is *not* re-fed each turn, and stale entries
linger and corrupt later actions. That is the version labs care about
(StatefulToolEnv, "memory must drive action") and the one worth building.
The in-context generator and scorer here remain useful as the easy control
condition and as the timeline source for the agentic version.
