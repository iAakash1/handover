# 12. Testing Strategy

Tests are designed now (with the spec) so the builder milestone implements against them. Every test is deterministic and uses **synthetic mini-DKBs** or the real DKB; none uses an LLM/VLM or images. Markers reuse the repo convention: `unit`, `integration`, `benchmark`/`regression`.

## 12.1 Unit tests (per builder function / rule)

| Area | Test |
|------|------|
| Id scheme | `symptom_id`, `value_id`, etc. are pure and stable; same inputs → same id; different inputs → different id. |
| Sign-type classification | each policy keyword maps to the expected `SignType`; ambiguous phrase without a keyword → surfaced (not guessed). |
| Vocabulary canonicalization | "dark-brown" and "dark brown" → one `color:dark_brown` node (R3). |
| Observability rule | symptom on `margin` → observable; symptom on `fruit` → non-observable; mixed → non-observable (R4). |
| Confidence precedence | a feature in both `diagnostic_visual_features` and `secondary_symptoms` merges to one node at `asserted`, evidence unioned (R1). |
| Condition subtype selection | `(is_pathogen=false, arthropod_pest)` → `PestDamage`; `(saprophytic_fungus)` → `SurfaceColonization`; healthy → `HealthyState`. |
| Severity split | extent terms → `has_extent [image_licensed=true]`; stages → `typical_at_severity [image_licensed=false]` (F8). |
| Disputed taxonomy | 2 candidate agents → 2 `caused_by` edges, both `disputed=true`, distinct evidence (R2). |
| Evidence linking | every quality/symptom edge carries a resolvable `evidence`; missing evidence → error (R5). |
| Forbidden relations | each of F1–F10 is rejected by the corresponding validator on a crafted bad graph. |

## 12.2 Integration tests (end-to-end build + validate)

| Test | Assertion |
|------|-----------|
| Tiny synthetic DKB (2 conditions: one Disease, one PestDamage) | build succeeds; graph passes V-ONT-1…12; expected node/edge counts. |
| Real DKB (18 conditions) | build succeeds; **100% DKB field coverage** (V-ONT-11); 0 errors. |
| Grounded-set query | for a Disease, the "asserted + observable" symptom set matches the DKB's diagnostic/primary features; the "non-observable" set matches `forbidden_symptoms_not_leaf_observable`. |
| Forbidden-set query | every DKB `forbidden_symptoms_not_leaf_observable` entry appears as an `observable=false` symptom reachable from its condition. |
| Provenance round-trip | every edge's evidence resolves to a `reference_registry` key present in the DKB. |
| Severity honesty | no reachable `typical_at_severity` edge is `image_licensed=true` (F8/C5) across the whole graph. |
| Fail-closed | a deliberately inconsistent synthetic DKB (pest `caused_by` a fungus) produces **no** graph and exit code 1. |

## 12.3 Regression / golden tests

| Test | Assertion |
|------|-----------|
| **Golden hash** | building the real 18-condition DKB yields a recorded `content_hash`; drift fails (any DKB/policy/builder change must be an intentional, reviewed version bump). |
| **Determinism** | build twice → identical `content_hash` (V-ONT-12); shuffle DKB input order → identical output (order-independence). |
| **Crop-additivity** | build `{tomato}` then `{tomato, mango}`; the **tomato subgraph is byte-identical** (adding mango does not perturb tomato) — proves crop independence ([07_crop_independence.md](07_crop_independence.md) §7.6). |
| **Schema-stability** | `ontology_schema.json` is byte-identical before/after adding a crop (only the graph changes). |
| **New-crop smoke** | a synthetic 3rd-crop DKB fragment builds + validates with **zero** schema edits. |

## 12.4 Property-based tests (invariants over all nodes/edges)

Run over both synthetic and real graphs:

- ∀ edge: endpoints satisfy `domain`/`range` (with inheritance).
- ∀ Symptom: exactly one `has_sign_type`; ≥1 `appears_on`; stored `observable` == recomputed.
- ∀ Condition: ≥1 `has_symptom`; `HealthyState` has exactly the `healthy_surface` symptom and no `caused_by`.
- ∀ evidence-carrying edge: `|evidence| ≥ 1` and all ids resolve.
- ∀ `typical_at_severity`: `image_licensed=false`.
- The `is_a` graph is a single-parent tree; `part_of` is acyclic.

Property tests catch classes of bugs that example-based tests miss and are the strongest guard as the DKB grows.

## 12.5 Future regression suite (as scale grows)

- **Snapshot tests** per crop: a stored, reviewed subgraph fingerprint per crop; adding diseases to *other* crops must not change it.
- **Migration tests**: when `schema_version` bumps MAJOR, assert that rebuilding an old DKB under the new schema preserves every prior *fact* (no knowledge lost) and still validates ([08_versioning.md](08_versioning.md) §8.5).
- **Coverage-trend test**: DKB field coverage stays at 100%; any new DKB field must be consumed or explicitly allow-listed (fails otherwise) — prevents silent drift when the DKB schema evolves.
- **Performance guard**: build time stays O(D) (a large synthetic DKB builds within a linear-scaling budget) — cheap insurance for the "hundreds of diseases" goal.

## 12.6 Test data policy

- Synthetic mini-DKBs live beside the tests and use the **real 46-field schema** (so tests exercise the true contract).
- The real DKB is used read-only for integration/regression.
- No test writes into `knowledge_base/`, `datasets/`, or `raw/`.
- Golden hashes are checked in and updated only via a reviewed change with a recorded rationale (a golden-hash change is a semantic event, like a version bump).
