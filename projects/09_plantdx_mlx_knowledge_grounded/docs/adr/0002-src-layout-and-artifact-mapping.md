# ADR 0002 — `src/` layout with an artifact mapping layer

- **Status:** Accepted
- **Date:** 2026-07-11
- **Spec:** `caption_framework/06_folder_structure_spec.md`

## Context

Spec doc 06 lists the pipeline's artifact directories (`ontology/`,
`vocabulary/`, `captions/`, `datasets/`, …) directly under `experiments/`. A
mature Python repository also wants an importable package, isolated code, and a
clean separation between source-controlled code and regenerable outputs.

## Decision

Use a **`src/plantdx/` package layout** for code, keep **authored inputs** in
`assets/` and `configs/`, and place **generated outputs** under `artifacts/`
whose subdirectories carry the **exact doc-06 names and semantics**.
`configs/paths.yaml` is the single mapping layer; setting `artifact_root: "."`
reproduces the literal doc-06 root-level layout.

## Consequences

- No methodology is redesigned: the doc-06 tree is the *logical* artifact layout
  and remains the contract; only the physical root is parameterized.
- `src/` layout prevents accidental imports of the un-installed tree and forces
  testing the installed package; `pythonpath = ["src"]` keeps the suite runnable
  pre-install.
- Generated artifacts are gitignored and grouped by `library_version`; authored
  inputs are version-controlled.
- Slight indirection (one mapping file) in exchange for a clean, conventional
  repository. Accepted.
