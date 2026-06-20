"""Validate the verifiers env end-to-end against OpenAI (small, paid)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import verifiers as vf  # noqa: E402
from supersede.env import load_environment  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1-mini"
    env = load_environment(data_path="data/longmemeval_oracle.json",
                           max_examples=n, budget=300, max_turns=12)
    cfg = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_key_var="OPENAI_API_KEY",
        api_base_url="https://api.openai.com/v1",
    )
    out = env.evaluate_sync(cfg, model=model,
                            sampling_args={"temperature": 0, "max_tokens": 120},
                            num_examples=n, rollouts_per_example=1,
                            save_results=False)

    def last_assistant(completion):
        for m in reversed(completion or []):
            r = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if r == "assistant":
                return (m.get("content") if isinstance(m, dict)
                        else getattr(m, "content", "")) or ""
        return ""

    outputs = out.get("outputs") or []
    rewards = [ro.get("reward") or 0.0 for ro in outputs]
    stops = {}
    for ro in outputs:
        sc = ro.get("stop_condition")
        stops[sc] = stops.get(sc, 0) + 1
    print(f"=== {len(outputs)} rollout(s) ===")
    if rewards:
        print(f"MEAN REWARD (env accuracy) = {sum(rewards) / len(rewards):.2%}")
        print(f"stop_conditions: {stops}")
    if len(outputs) > 6:
        return
    for i, ro in enumerate(outputs):
        comp = ro.get("completion")
        print(f"\n[rollout {i}]")
        print(f"  reward         = {ro.get('reward')}")
        print(f"  is_completed   = {ro.get('is_completed')}")
        print(f"  is_truncated   = {ro.get('is_truncated')}")
        print(f"  stop_condition = {ro.get('stop_condition')}")
        print(f"  error          = {ro.get('error')}")
        print(f"  trajectory_len = {len(ro.get('trajectory') or [])}")
        print(f"  completion_len = {len(comp or [])}")
        print(f"  gold answer    = {ro.get('answer')!r}")
        print(f"  model answer   = {last_assistant(comp)[:160]!r}")


if __name__ == "__main__":
    main()
