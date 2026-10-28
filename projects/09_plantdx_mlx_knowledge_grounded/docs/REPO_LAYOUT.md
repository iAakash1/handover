# Repository Layout & Spec Mapping

This repository organizes the code as a mature Python project (`src/` layout)
while preserving the **final** artifact layout defined in
[`../caption_framework/06_folder_structure_spec.md`](../caption_framework/06_folder_structure_spec.md)
(doc 06). Nothing in the methodology was redesigned; the doc-06 tree is the
*logical* artifact layout, and `configs/paths.yaml` is the single mapping layer
onto the physical repository.

## Top level

```
experiments/                     # repository root (== the spec's `experiments/`)
├── src/plantdx/                 # Python package (code)
├── configs/                     # authored configuration (Step 4)
├── assets/                      # AUTHORED inputs (templates, static vocab, label_map, overrides)
├── artifacts/                   # GENERATED outputs (gitignored) — the doc-06 artifact tree
├── tests/                       # unit / integration / benchmark
├── docs/                        # developer documentation
├── scripts/                     # thin CLI wrappers
├── knowledge_base/              # Stage 1 — the DKB (FINAL)
├── caption_framework/           # Stage 2 — design specification (FINAL)
├── tomato/raw/PlantVillage/     # dataset (existing; images gitignored)
└── mango/raw/MangoLeafBD/       # dataset (existing; images gitignored)
```

## doc-06 artifact directory → repository path

The spec lists artifact directories directly under `experiments/`. For repository
hygiene they are nested under `artifacts/` by default, with **identical
subdirectory names and semantics**. To reproduce the literal doc-06 root-level
layout, set `artifact_root: "."` in `configs/paths.yaml`.

| doc-06 directory | repo path (default) | inputs/outputs |
|------------------|---------------------|----------------|
| `ontology/` | `artifacts/ontology/` (+ authored `assets/ontology_overrides/`) | generated |
| `vocabulary/` | `artifacts/vocabulary/` (+ authored sources in `assets/vocabulary/`) | mixed |
| `templates/` | authored in `assets/templates/`; derived index in `artifacts/templates/` | mixed |
| `generation/` | authored `configs/generation.yaml`; `artifacts/generation/provenance/` | mixed |
| `validators/` | authored `configs/validation.yaml`; `artifacts/validators/reports/` | mixed |
| `captions/` | `artifacts/captions/` | generated |
| `datasets/` | `artifacts/datasets/` | generated |
| `metadata/` | authored `assets/metadata/label_map.json`; `artifacts/metadata/` | mixed |
| `qa/` | `artifacts/qa/` | generated |

Key distinction the mapping preserves:
- **Authored inputs** (version-controlled): templates, instruction bank, static
  vocabulary sources, `label_map.json`, ontology overrides, and the `configs/`.
- **Generated outputs** (gitignored, regenerable): derived ontology/vocabulary,
  the caption library, converted datasets, provenance, reports, stats, QA results.

## Why `src/` layout

Prevents accidental imports of the un-installed working tree, forces testing the
installed package, and cleanly separates code from generated data. Tests add
`src` to `pythonpath` (see `pyproject.toml`) so the suite runs without an install
in CI too.

## Versioning

Generated artifacts are grouped by `library_version` (`L1`, `L2`, …), bumped on
any change to the DKB, ontology, templates, vocabulary, or config. A frozen,
QA-accepted `library_version` is immutable. See doc 06 §5.
