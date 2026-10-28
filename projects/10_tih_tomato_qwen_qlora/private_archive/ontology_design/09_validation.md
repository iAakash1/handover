# 9. Validation Strategy

Validation is what makes the ontology a *scientific instrument* rather than a data dump: an ontology is publishable only if it passes the full validator battery. Validation is deterministic and requires no learned components.

## 9.1 What makes an ontology **valid**

An ontology is valid iff **all** of the following hold:

1. **Schema-conformant.** Every node's `type` is a declared, non-abstract concept type; every node satisfies its (inherited) property schema; every edge's `type` is declared and its endpoints satisfy `domain`/`range`.
2. **Cardinality-satisfying.** Every cardinality constraint in [05_rules.md](05_rules.md) §5.3 holds.
3. **Rule-clean.** No forbidden relationship (F1–F10) occurs; all consistency rules (C1–C6) hold. (C7 differential symmetry is a warning.)
4. **Referentially closed.** Every edge `source`/`target` and every `evidence` id refers to an existing node; every `Evidence` node resolves to a DKB `reference_registry` key.
5. **Observability-consistent.** For every symptom, stored `observable` equals the value recomputed from its `appears_on` targets (§4.3).
6. **Severity-honest.** Every `typical_at_severity` edge has `image_licensed=false`; every `has_extent` edge has `image_licensed=true`.
7. **Acyclic where required.** `is_a` and `part_of` are acyclic; `is_a` is a single-parent tree.
8. **DKB-covering.** Every DKB per-disease field is either consumed (mapped to nodes/edges) or listed in the schema's `non_structural_fields` allow-list. Nothing is silently dropped.
9. **Deterministic.** A rebuild from the same inputs produces an identical `content_hash`.
10. **Provenance-complete.** `dkb_sha256`, `policy_hash`, `schema_hash`, `builder_version`, `content_hash` are all present and internally consistent (e.g., `schema_hash` matches the emitted schema).

## 9.2 What makes it **invalid** (failure taxonomy)

| Class | Examples | Severity |
|-------|----------|----------|
| Type violation | node typed as an abstract class; edge endpoint violates domain/range | **error** |
| Cardinality violation | a `Condition` with 0 symptoms; a `Symptom` with 2 sign types | **error** |
| Forbidden relation | `PestDamage caused_by Pathogen` (F1); asserted non-observable symptom (F7) | **error** |
| Consistency violation | category/agent mismatch (C1); mutual-exclusion breach (C3) | **error** |
| Dangling reference | edge to a missing node; evidence id absent from registry | **error** |
| Observability drift | stored `observable` ≠ recomputed (C4) | **error** |
| Severity licensing breach | `typical_at_severity` with `image_licensed=true` (F8/C5) | **error** |
| Coverage gap | a DKB field neither consumed nor allow-listed | **error** |
| Non-determinism | rebuild hash mismatch | **error** |
| Differential asymmetry | `A→B` without `B→A` (C7) | **warning** |
| Orphan value node | a `Color` node referenced by nothing | **warning** (pruned or flagged) |
| Low evidence quality | a hallmark symptom with only textbook (no peer-reviewed/extension) evidence | **info** (reported metric) |

**Policy:** any **error** ⇒ the build fails (non-zero exit; [11_cli_contract.md](11_cli_contract.md)). **Warnings** are reported and may be gated in CI. **Info** feeds coverage metrics.

## 9.3 The automated validator battery

Each validator is a small, single-responsibility, deterministic check. Named `V-ONT-*`.

| ID | Validator | Checks | Blocking |
|----|-----------|--------|----------|
| V-ONT-1 | Schema conformance | node/edge types declared; property schema satisfied | ✓ |
| V-ONT-2 | Domain/range | every edge respects `domain`/`range` (with inheritance) | ✓ |
| V-ONT-3 | Cardinality | all §5.3 constraints | ✓ |
| V-ONT-4 | Forbidden relations | F1–F10 | ✓ |
| V-ONT-5 | Consistency | C1–C6 | ✓ |
| V-ONT-6 | Referential integrity | no dangling node/edge/evidence ids | ✓ |
| V-ONT-7 | Evidence closure | every `Evidence` resolves to `reference_registry` | ✓ |
| V-ONT-8 | Observability recompute | stored == derived `observable` | ✓ |
| V-ONT-9 | Severity licensing | flags on `typical_at_severity`/`has_extent` | ✓ |
| V-ONT-10 | Acyclicity & single-parent | `is_a` tree, `part_of` DAG | ✓ |
| V-ONT-11 | DKB coverage | every field consumed or allow-listed | ✓ |
| V-ONT-12 | Determinism/hash | rebuild → identical `content_hash` | ✓ |
| V-ONT-13 | Differential symmetry | C7 | warning |
| V-ONT-14 | Orphan detection | unreferenced value nodes | warning |
| V-ONT-15 | Evidence-quality report | per-symptom source tier distribution | info |

Execution order is fixed (V-ONT-1 … V-ONT-15) for deterministic, comprehensible reports. Validators run on `(schema, graph)` in memory and emit a structured `validation_report.json` plus a human-readable summary.

## 9.4 Two validation surfaces

1. **Build-time self-validation.** The builder runs V-ONT-1…12 before emitting; a failing build produces **no** graph artifact (fail closed). This guarantees only valid ontologies ever exist on disk.
2. **Standalone re-validation.** `plantdx ontology validate` re-runs the battery on an existing graph (e.g., in CI, or to check a graph a collaborator sent). Because the ontology is content-hashed, re-validation also re-verifies determinism (V-ONT-12) by rebuilding and comparing.

## 9.5 Relationship to downstream validation

The Caption Framework's 12-stage caption validator (frozen) enforces caption-level grounding. This ontology validator enforces *knowledge-level* soundness one layer earlier. They are complementary:
- Ontology validation guarantees the *substrate* is sound (evidence-linked, observability-correct, consistent).
- Caption validation guarantees each *rendered sentence* stays within what the substrate licenses.

A caption can only be as trustworthy as the ontology it draws from; validating the ontology first is what lets the caption validator's job be reduced to "did the realization stay inside the licensed set?" — a much simpler, checkable property.
