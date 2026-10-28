# 11. CLI Contract — `plantdx ontology`

The ontology commands integrate into the **existing** `plantdx` CLI (one entry point, consistent with `audit` and `normalize`). Nothing here is implemented in this milestone; this is the contract the builder milestone must satisfy.

## 11.1 Commands

```
plantdx ontology build     [--config PATH] [--out DIR] [--schema PATH] [--check] [--fail-on-warning]
plantdx ontology validate  [--config PATH] [--graph PATH] [--schema PATH] [--fail-on-warning]
plantdx ontology stats     [--config PATH] [--graph PATH] [--format md|json]
```

- `build` — run `f(DKB, policies, schema)`, self-validate (V-ONT-1…12), emit artifacts.
- `validate` — re-run the full battery (V-ONT-1…15) on an existing graph, including a determinism rebuild (V-ONT-12).
- `stats` — print/emit the statistics + coverage metrics for an existing graph.

## 11.2 Arguments

| Arg | Applies to | Default | Meaning |
|-----|-----------|---------|---------|
| `--config` | all | `configs/config.yaml` | Reuses the existing config system (paths, ontology dir, policy locations). |
| `--out` | build | `paths.artifacts.ontology_dir` (`artifacts/ontology/`) | Output directory for artifacts. |
| `--schema` | build/validate | `<ontology_dir>/ontology_schema.json` (or the packaged T-Box) | The T-Box to build against / validate with. |
| `--graph` | validate/stats | `<ontology_dir>/ontology_graph.json` | The A-Box to check. |
| `--check` | build | off | Build in memory and validate but **do not write** (dry run / CI gate). |
| `--fail-on-warning` | build/validate | off | Treat V-ONT warnings as failures (strict CI). |
| `--format` | stats | `md` | `md` human summary or `json`. |

Config, not flags, is the source of truth for paths and policy locations (no magic constants) — consistent with prior milestones.

## 11.3 Artifacts (written by `build`)

Into `artifacts/ontology/`:

| File | Description |
|------|-------------|
| `ontology_schema.json` | The T-Box (may be copied/frozen from the packaged schema). |
| `ontology_graph.json` | The A-Box: nodes + edges + provenance (incl. `content_hash`). |
| `ontology_stats.json` | Counts, coverage metrics ([06_statistics.md](06_statistics.md)). |
| `validation_report.json` | Per-validator results (V-ONT-1…15). |
| `build_report.md` | Human-readable: DKB field-coverage table, decisions log, warnings, disputed-taxonomy list, unclassifiable-phrase list (if any). |

`build` is **atomic**: artifacts are written only after self-validation passes; a failing build writes `validation_report.json` (for diagnosis) but **no** graph.

## 11.4 Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success (valid ontology built / validated; no blocking issues). |
| `1` | Validation failure — one or more V-ONT **errors** (or warnings under `--fail-on-warning`). |
| `2` | Configuration/input error — missing DKB, unreadable schema, bad config. |
| `3` | Determinism failure — rebuild `content_hash` ≠ recorded (V-ONT-12). |
| `4` | Coverage failure — a DKB field neither consumed nor allow-listed (V-ONT-11). |

Distinct codes let CI distinguish "the knowledge is inconsistent" (1) from "the pipeline is misconfigured" (2) from "reproducibility broke" (3) — each has a different owner and fix.

## 11.5 Logging

- Structured logging via the existing `plantdx.utils.logging` (console + optional file), namespace `plantdx.ontology`.
- **INFO**: per-phase progress (`[phase 2] condition tomato_early_blight → 12 symptoms, 6 observable`), final counts, `content_hash` prefix.
- **WARNING**: differential asymmetry, orphan value nodes, low-evidence hallmarks.
- **ERROR**: the first failing validator with the offending node/edge id and the rule id (e.g., `V-ONT-4 F7: asserted non-observable symptom symptom:…`).
- A machine log line summarizing the run (counts + hash + exit code) for CI capture.

## 11.6 Interaction with other commands (pipeline ordering)

```
plantdx audit        # milestone 2   (datasets inventory)
plantdx normalize    # milestone 2.1 (canonical datasets)
plantdx ontology build   # this layer (DKB → ontology)   ← prerequisite for all knowledge-consuming stages
plantdx ontology validate
# later:
plantdx vocabulary build / generate / ...   # consume the ontology, not the DKB
```

`ontology build` depends only on `knowledge_base/dkb.json` + policies (not on `datasets/` or `audit`/`normalize` outputs). It is therefore runnable at any time after the DKB exists, and it is a hard prerequisite for every downstream knowledge-consuming stage.
