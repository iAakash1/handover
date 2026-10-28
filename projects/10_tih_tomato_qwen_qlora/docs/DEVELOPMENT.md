# Development Guide

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # package + ruff/mypy/pytest/pre-commit
pre-commit install
```

Training extras (Apple Silicon only): `pip install -e ".[train]"`.
Grammar validation stack (V11): `pip install -e ".[nlp]"`.

## Everyday commands

```bash
make help          # list tasks
make fmt           # ruff format + autofix
make lint          # ruff check
make type          # mypy --strict
make test          # pytest (all)
make test-unit     # unit only (the Milestone-1 green suite)
make cov           # coverage report
make check         # fmt + lint + type + test  (run before every PR)
```

## Coding standards

- **Python 3.10+**, fully typed; `mypy --strict` must pass. Public functions and
  classes have Google-style docstrings.
- **Line length 100**, ruff-formatted. Import order handled by ruff (isort rules).
- **No domain facts in code.** Disease facts live only in the DKB; vocabulary in
  `assets/`/derived artifacts; syntax in templates. Never hard-code a symptom,
  color, or forbidden term in a `.py` file.
- **Determinism.** No wall-clock or unseeded randomness in the caption path; draw
  from the seed functions in `core.seeding`.
- **Frozen dataclasses / pydantic** for value objects and config; prefer pure
  functions and small single-responsibility classes.

## Testing conventions

- `tests/` mirrors the package. Markers: `unit`, `integration`, `benchmark`,
  `requires_dkb`, `requires_mlx`.
- Milestone-1 tests assert **structure and contracts** (imports, public API,
  enums/dataclasses, CLI surface, registries, stubs raise `NotImplementedError`).
- Behavioural tests for a milestone are added **skipped** ahead of time so the
  acceptance criteria are captured and the suite stays green; the milestone's PR
  removes the skip and implements the behaviour.

## Adding a component (per milestone)

1. Implement the interface already declared in Milestone 1 (do not change its
   signature without an ADR).
2. Keep it deterministic and image-blind if it is on the caption path.
3. Add unit tests + unskip the relevant integration/benchmark placeholders.
4. Update `CHANGELOG.md` and, if a design choice was made, add an ADR
   (`docs/adr/`).

## Invariants (blocking review criteria)

Any change must preserve the seven design invariants
(`caption_framework/README.md`). A PR that violates one is rejected regardless of
code quality. When in doubt, the specification in `caption_framework/` and
`knowledge_base/` wins.
