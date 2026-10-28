# artifacts/

**Generated pipeline outputs** — gitignored and fully regenerable from the DKB +
`assets/` + `configs/`. Only `.gitkeep`/README placeholders are tracked so the
directory structure is preserved.

Subdirectory names and semantics match the specification's artifact tree
(`caption_framework/06_folder_structure_spec.md`); see
[`../docs/REPO_LAYOUT.md`](../docs/REPO_LAYOUT.md) for the mapping. Set
`artifact_root: "."` in `configs/paths.yaml` to emit these at the repo root
instead (the literal doc-06 layout).

| Directory | Produced by | Milestone |
|-----------|-------------|-----------|
| `ontology/` | Ontology Builder (A) | M2 |
| `vocabulary/` | Vocabulary (B) + Symptom Lexicon (C) builders | M2 |
| `templates/` | Derived template index | M3 |
| `generation/provenance/` | Emitter (per-caption provenance) | M3–M4 |
| `validators/reports/` | Validator battery run reports | M3 |
| `captions/` | The accepted caption library (`caption_library.jsonl`) | M3–M4 |
| `datasets/` | Splits + per-model converted training files | M4 |
| `metadata/` | Version manifest, dataset card, diversity stats, eval | M4–M6 |
| `qa/` | Audit samples, review results, acceptance sign-off | M4 |

Every artifact family is grouped by `library_version` and immutable once
QA-accepted.
