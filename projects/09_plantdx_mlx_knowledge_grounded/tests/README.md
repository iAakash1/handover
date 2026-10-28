# tests/

Test suite for PlantDx. Layout mirrors the package.

| Directory | Scope | Marker |
|-----------|-------|--------|
| `unit/` | Fast, isolated tests of a single module (imports, public API, enums, dataclasses, CLI surface, registries, stub contracts). | `unit` |
| `integration/` | Cross-module pipeline wiring tests. Behavioural cases are skipped until the relevant milestone implements them. | `integration` |
| `benchmark/` | Dataset-quality / diversity-gate benchmarks over a generated corpus (slow; run on demand). Skipped until a corpus exists. | `benchmark` |
| `fixtures/` | Small static fixtures (tiny DKB slice, sample config) used across tests. | — |

## Running

```bash
pytest                # everything
pytest -m unit        # unit only (Milestone-1 green suite)
pytest -m integration # integration
pytest -m benchmark   # benchmarks (require a generated library)
make cov              # with coverage
```

## Milestone-1 policy

During scaffolding the package exposes typed interfaces whose bodies raise
`NotImplementedError`. Unit tests therefore assert **structure and contracts**
(symbols exist, enums/dataclasses behave, the CLI parses, registries are
complete, stubs raise `NotImplementedError` with a milestone tag). Behavioural
tests are written as **skipped** placeholders that unlock in their milestone, so
the suite stays green and the intent is captured.
