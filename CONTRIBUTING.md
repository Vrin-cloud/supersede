# Contributing to Supersede

Thanks for your interest — contributions are genuinely welcome.

## Especially valuable right now

- **Harder training episodes.** The current procedural curriculum saturates
  (the model solves it and training self-terminates). Longer, more implicit,
  more distractor-heavy episodes are the clearest path to a bigger lift.
- **New model results.** Run the environment on other models / sizes and report
  the numbers.
- **Reward refinements** and edge cases the matcher mishandles.
- **Bug reports** with a minimal repro.

## Dev setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[env,dev]"
pytest          # 21 tests, offline, no API key
ruff check src tests
```

## Pull requests

1. Fork and branch from `main`.
2. Keep changes focused; add a test for new behavior.
3. Run `ruff check src tests` and `pytest` — both must pass (CI enforces this).
4. Open the PR with a clear description of *what* and *why*.

New contributors: look for the
[`good first issue`](https://github.com/Vrin-cloud/supersede/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
label.

## Questions

Open a [Discussion](https://github.com/Vrin-cloud/supersede/discussions) or email
`vedant@vrin.cloud`. By contributing you agree your work is licensed under
Apache-2.0.
