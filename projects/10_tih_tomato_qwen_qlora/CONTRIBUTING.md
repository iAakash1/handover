# Contributing to PlantDx

Thank you for your interest in PlantDx. This document describes how to set up a
development environment and the standards contributions must meet.

## Golden rule: the specification is the source of truth

PlantDx follows a **research design that is final**. Implementation must match it
exactly — do not redesign, simplify, or change the methodology.

- **Stage 1 — Disease Knowledge Base:** [`knowledge_base/`](knowledge_base/) (final).
- **Stage 2 — Caption Framework specification:** [`caption_framework/`](caption_framework/) (final).

Every change must preserve the **seven design invariants** listed in
[`caption_framework/README.md`](caption_framework/README.md):
label-only grounding, DKB as single source of truth, closed vocabulary,
observability, pest/pathogen register integrity, severity honesty, and
reproducibility. A PR that violates an invariant will be rejected regardless of
code quality.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a PR

Run the full local CI:

```bash
make check      # ruff format + ruff lint + mypy (strict) + pytest
```

Requirements for every PR:

- **Typed.** `mypy --strict` passes; no untyped public functions.
- **Documented.** Google-style docstrings on all public classes/functions.
- **Tested.** New behavior has unit tests; pipeline changes have integration tests.
- **Deterministic.** No wall-clock or unseeded randomness in the caption path.
- **No domain facts in code.** Disease facts live only in the DKB; vocabulary in
  `assets/`/derived artifacts; syntax in templates. Never hard-code a symptom,
  color, or forbidden term in a `.py` file.

## Milestones

Work is organized into milestones (see [`docs/ROADMAP.md`](docs/ROADMAP.md)).
Please scope PRs to a single milestone/component. Milestone 1 is scaffolding
only; do not implement generator/validator/dataset/training logic until the
corresponding milestone is open.

## Commit and branch conventions

- Branch from `main`: `feat/<milestone>-<component>`, `fix/<area>`, `docs/<area>`.
- Conventional Commits style messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Keep commits focused and reviewable.

## Code of Conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
