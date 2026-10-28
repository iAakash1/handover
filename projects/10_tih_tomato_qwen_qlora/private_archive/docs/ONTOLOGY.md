# Domain Ontology Compiler

A CPU-only, **deterministic** compiler: `Ontology = f(DKB, Policies)`. It reads
`knowledge_base/dkb.json` plus the fixed policies in
`src/plantdx/ontology/domain/policies.py` and emits a typed knowledge graph. It
uses no images, no datasets, no LLM/VLM, and no randomness — repeated builds are
byte-identical. Design rationale lives in [`../ontology_design/`](../ontology_design/).

> This is the **domain ontology** (the knowledge-graph substrate). It is separate
> from the caption-concept model in `plantdx.ontology` (a downstream view / later
> milestone) — the two coexist in the `plantdx.ontology` package.

## Run it

From the repository root (`experiments/`):

```bash
plantdx ontology                       # compile + validate + write artifacts
plantdx ontology --validate-only       # compile + validate; write nothing
plantdx ontology --stats-only          # compile + validate; print statistics JSON
plantdx ontology --output some/dir     # override output directory
python -m plantdx ontology             # equivalent, without the console script
```

## Artifacts

Written to `artifacts/ontology/` (gitignored):

| File | Contents |
|------|----------|
| `ontology.json` | The complete ontology: T-Box (concept + relation types) + A-Box (nodes + edges) + provenance. |
| `concept_graph.json` | A graph-centric view: light nodes + directed, attributed edges. |
| `concept_index.json` | Lookup indices: by type, by crop, condition → symptoms. |
| `ontology_statistics.json` | Counts, coverage, inheritance depth, validation status, checksum. |
| `ontology_checksum.txt` | The content-only SHA-256 (`sha256:…`); identifies the build. |
| `ontology_build.log` | Deterministic, timestamp-free build log. |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success (valid ontology built / validated). |
| `1` | Validation failure (a V-ONT rule was violated; fail closed — no artifacts written). |
| `2` | Configuration or DKB load/validation error. |

## Guarantees

- **Deterministic.** No timestamps in artifacts, no UUIDs, everything sorted; the
  checksum depends only on ontology content (not location/machine/OS/time).
- **Evidence-linked.** Every domain assertion references an `Evidence` node
  resolving to the DKB `reference_registry`.
- **Observability & severity honesty are structural** (symptoms carry an
  `observable` flag; `typical_at_severity` edges are always `image_licensed=false`).
- **Fail closed.** Any rule violation aborts the build with a specific `V-ONT-*` error.
