# 5. Ontology Rules

The rules are declarative and enforced by the validator ([09_validation.md](09_validation.md)). They are *monotonic* (no rule requires probabilistic or learned inference) and *deterministic*.

## 5.1 Allowed relationships (domain/range matrix)

An edge `(s) –r→ (t)` is **type-allowed** iff `type(s)` satisfies `domain(r)` and `type(t)` satisfies `range(r)`, respecting `is_a` inheritance. The authoritative matrix is the relation catalog in [04_concept_graph.md](04_concept_graph.md) §4.1. Summary of the permitted spine:

```
Condition   –affects→ Crop
Condition   –caused_by→ CausalAgent
Condition   –has_symptom→ Symptom
Condition   –has_extent→ Extent
Condition   –typical_at_severity→ Severity
Condition   –differentiated_from→ Condition
Condition   –favored_by→ EnvironmentalCondition
CausalAgent –agent_in_category→ AgentCategory
Pathogen    –member_of_family→ PathogenFamily
Symptom     –has_sign_type→ SignType
Symptom     –has_{color|shape|size|texture|distribution|morphology}→ Quality
Symptom     –appears_on→ (LeafRegion ∪ PlantPart)
Symptom     –has_observability→ Observability
(LeafRegion∪PlantPart) –part_of→ PlantPart
SignType    –mutually_exclusive_with→ SignType
```

## 5.2 Forbidden relationships (hard errors)

These are *type-level* prohibitions; any instance is a build failure.

| # | Forbidden pattern | Rationale |
|---|-------------------|-----------|
| F1 | `PestDamage –caused_by→ Pathogen` | Pest damage is not an infection (register integrity). Must point to a `Pest`. |
| F2 | `SurfaceColonization –has_symptom→ Symptom` with a `Necrosis`-implying quality on host tissue | Sooty mould does not necrose host tissue; the coating is superficial. |
| F3 | `HealthyState –has_symptom→ Symptom` where `sign_type ≠ healthy_surface` | Healthy has no disease signs. |
| F4 | `HealthyState –caused_by→ *` | Healthy has no causal agent. |
| F5 | `Symptom –appears_on→ CausalAgent` / `→ Crop` / `→ Condition` | `appears_on` ranges over anatomy only. |
| F6 | `Disease –caused_by→ Pest` / `→ Saprophyte` (when `agent_category` says pathogen) | Category/agent-type must agree. |
| F7 | Any `has_symptom` edge with `confidence=asserted` whose target has `observable=false` | You cannot *assert* (label-license) a non-visible symptom. Non-observable symptoms may only be `typical`/`hedged` and exist to be *excluded* downstream. |
| F8 | `typical_at_severity` edge with `image_licensed=true` | Severity honesty: per-image stage is never licensed. |
| F9 | Two `has_color` (etc.) edges from one Symptom to `mutually_exclusive_with` value nodes | Internal contradiction (e.g., a symptom both white and black coating). |
| F10 | `caused_by` with empty `evidence` | Every causal claim must be sourced. |

## 5.3 Cardinality constraints

| Relation / node | Constraint |
|-----------------|------------|
| `Condition –has_symptom→` | `≥ 1` for every non-abstract `Condition`. |
| `HealthyState` | Exactly `1` `has_symptom` whose sign type is `healthy_surface`; `0` other symptoms; `0` `caused_by`. |
| `Symptom –has_sign_type→` | Exactly `1`. |
| `Symptom –appears_on→` | `≥ 1`. |
| `Symptom` (in-degree on `has_symptom`) | Exactly `1` (each symptom belongs to one condition). |
| `Symptom –has_size→` | `0..1` (a symptom has at most one size band). |
| `Disease –caused_by→` | `≥ 1` (may be `> 1` only if `disputed=true` on each). |
| `CausalAgent –agent_in_category→` | Exactly `1`. |
| `Pathogen –member_of_family→` | `0..1`. |
| every evidence-carrying edge | `evidence` list length `≥ 1`. |

## 5.4 Consistency rules (semantic, cross-edge)

- **C1 — Category/agent coherence.** `Condition.agent_category` (from DKB) must equal the `AgentCategory` reached via `caused_by → agent_in_category`. (For disputed agents, all candidates must share the category, else it is flagged.)
- **C2 — Sign/condition coherence.** The `SignType` of a condition's *primary* symptom must be compatible with its `Condition` subtype: `PestDamage` primaries ∈ {stippling, cut, gall, deformation}; `SurfaceColonization` primary = coating; `HealthyState` primary = healthy_surface; `Disease` primaries ∈ {lesion, mottle, coating(mildew), deformation}. Compatibility table owned by a policy file.
- **C3 — Mutual exclusion.** No condition may present two symptoms whose sign types (or color/texture values) are declared `mutually_exclusive_with` **at asserted confidence** (e.g., not both white-powdery and black-sooty as hallmarks). Exclusions across *different* conditions are fine.
- **C4 — Observability materialization.** Stored `observable` == recomputed `observable` (§4.3) for every symptom.
- **C5 — Severity licensing.** All `typical_at_severity` edges have `image_licensed=false`; all `has_extent` edges have `image_licensed=true`. No exceptions.
- **C6 — Evidence closure.** Every `Evidence` id referenced by any edge exists as an `Evidence` node and resolves to a `reference_registry` key present in the DKB.
- **C7 — Differential symmetry (soft).** If `A differentiated_from B`, expect `B differentiated_from A`. Asymmetry is a *warning* (DKB `confused_with` lists may be one-directional), not an error; the builder may symmetrize under policy.

## 5.5 Inheritance rules

- **I1 — Property inheritance.** A node satisfies the union of properties along its `is_a` chain (see [02_schema.md](02_schema.md) §2.9).
- **I2 — Relation applicability by subsumption.** A relation with domain `Condition` applies to all four `Condition` subtypes; a rule stated on `Condition` binds all subtypes unless a subtype states a stricter rule (subtype rules override, never loosen — see I3).
- **I3 — Monotonic specialization.** A subtype may *tighten* a constraint (e.g., `HealthyState` tightens `has_symptom` to exactly 1) but may **never loosen** an ancestor's constraint. The validator checks that subtype constraints are ⊆ parent constraints in permissiveness.
- **I4 — No multiple inheritance.** Enforced structurally ([02_schema.md](02_schema.md) §2.9); a second "is-a-like" need is modeled as an explicit relation.

## 5.6 Conflict resolution (deterministic)

When the DKB contains ambiguity, the builder resolves it by fixed, documented precedence — **never** by choosing arbitrarily and never by an LLM.

- **R1 — Symptom priority.** When the same visual feature appears in multiple DKB fields, precedence for `confidence` assignment is: `diagnostic_visual_features` (→ asserted) > `key_differentiating_features` (→ asserted) > `primary_symptoms` (→ asserted/typical per policy) > `secondary_symptoms` (→ hedged). Duplicates are merged into one `Symptom` node keyed by canonical label; the *highest* confidence wins; evidence lists are unioned.
- **R2 — Disputed taxonomy.** If the DKB records multiple candidate agents (via `taxonomy_note` / `documented_taxonomic_disagreements`), emit **one `caused_by` edge per candidate**, each `disputed=true`, each with its own evidence. The ontology does not pick a winner; downstream stays at the disease level. This directly preserves the DKB's stated scientific honesty.
- **R3 — Vocabulary canonicalization conflict.** If two DKB surface strings normalize to the same canonical value (e.g., "dark-brown" and "dark brown"), they map to one value node; the mapping table (a policy) is authoritative and its application is order-independent.
- **R4 — Observability conflict.** If a symptom's phrasing references both a leaf region and a non-leaf part, `observable=false` wins (conservative: if any part is non-observable, the symptom is treated as non-leaf-observable and excluded from asserted use). Rationale: false-positive grounding is worse than losing a symptom.
- **R5 — Missing evidence.** A domain assertion with no resolvable reference is a **hard error** (not silently dropped, not defaulted). The DKB must be fixed. This keeps "evidence at the edge" true by construction.
- **R6 — Determinism tiebreak.** Any residual ordering choice (e.g., sibling symptom order) is resolved by lexicographic sort of canonical ids, so the output is stable.

## 5.7 What is intentionally NOT a rule

- No probabilistic thresholds, no confidence *arithmetic* (confidence is a 3-value ordinal label, not a number to be combined). Rationale: numeric confidence invites false precision and nondeterministic aggregation.
- No cross-disease inference ("if A and B share a symptom then …"). The ontology asserts only what the DKB states. Emergent reasoning is out of scope and would break determinism/traceability.
- No auto-repair of the DKB. Rule violations rooted in the DKB are surfaced as errors for a human to fix the source of truth, not patched in the ontology.
