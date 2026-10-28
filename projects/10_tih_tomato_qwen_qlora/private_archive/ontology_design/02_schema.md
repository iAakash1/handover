# 2. Ontology JSON Schema (`ontology.json`)

This document specifies the on-disk schema of the ontology. It is a **design specification**; the JSON below is illustrative (fenced examples), not a delivered artifact. The builder (later milestone) emits documents conforming to this schema.

## 2.1 Physical layout: two files, one logical model

The ontology is emitted as **two files** plus a stats/report file:

| File | Contents | Changes when |
|------|----------|--------------|
| `ontology_schema.json` | The **T-Box**: `concept_types`, `relation_types`, `constraints`, closed-vocabulary declarations. Crop-independent. | Only on a deliberate `schema_version` bump. |
| `ontology_graph.json` | The **A-Box**: `nodes`, `edges`, provenance. DKB-derived. | Whenever DKB / policies / schema change (`ontology_version` bump). |
| `ontology_stats.json` | Derived counts and coverage metrics (see [06_statistics.md](06_statistics.md)). | With every build. |

**Rationale.** Separating T-Box from A-Box lets consumers pin the *schema* they were written against (a stable contract) independently of the *graph* (which grows with data). It also makes diffs meaningful: a graph-only diff proves the schema was untouched. The two files share `schema_version`; the graph additionally carries `ontology_version` and `dkb_sha256`.

> Where they live: under the existing `paths.artifacts.ontology_dir` (i.e., `artifacts/ontology/`), consistent with the frozen folder-structure spec. The build never writes into `knowledge_base/` or `datasets/`.

## 2.2 Top-level envelope

`ontology_schema.json`:

```json
{
  "kind": "plantdx.ontology.schema",
  "schema_version": "1.0.0",
  "generated_by": "plantdx.ontology.builder",
  "concept_types": [ /* §2.3 */ ],
  "relation_types": [ /* §2.5 */ ],
  "closed_vocabularies": [ /* §2.7 */ ],
  "constraints": { /* §2.8, and see 05_rules.md */ }
}
```

`ontology_graph.json`:

```json
{
  "kind": "plantdx.ontology.graph",
  "schema_version": "1.0.0",
  "ontology_version": "O1",
  "provenance": {
    "dkb_sha256": "…",
    "policy_hash": "…",
    "schema_hash": "…",
    "builder_version": "…",
    "content_hash": "…",
    "created_utc": "2026-…"
  },
  "nodes": [ /* §2.4 */ ],
  "edges": [ /* §2.6 */ ]
}
```

`content_hash` is the deterministic hash of `(nodes ⧺ edges)` under canonical ordering (see [10_build_algorithm.md](10_build_algorithm.md) §6). It is the ontology's identity and the basis of regression tests.

## 2.3 Concept type (T-Box)

A **concept type** is a node type. It declares its parent (`is_a`), whether it is abstract, its data properties, and — for closed classes — that its individuals are a fixed enumeration.

```json
{
  "id": "Symptom",
  "label": "Symptom",
  "is_a": "Observation",
  "abstract": false,
  "closed": false,
  "properties": [
    { "name": "canonical_label", "type": "string",  "required": true,  "cardinality": "1" },
    { "name": "source_field",    "type": "string",  "required": true,  "cardinality": "1" },
    { "name": "source_text",     "type": "string",  "required": false, "cardinality": "0..1" },
    { "name": "observable",      "type": "boolean", "required": true,  "cardinality": "1" }
  ]
}
```

Fields:

| Field | Meaning |
|-------|---------|
| `id` | Unique type identifier (PascalCase). Referenced by nodes' `type` and by relations' `domain`/`range`. |
| `is_a` | Parent type id (single inheritance); `null` only for the root `Entity`. |
| `abstract` | If `true`, no node may have this exact type (only its descendants). E.g., `Observation`, `CausalAgent`. |
| `closed` | If `true`, individuals are a fixed enumeration from a `closed_vocabulary` (§2.7); the builder never creates new ones from the DKB. |
| `properties` | Data (literal) properties. Object properties are modeled as **relations**, never as node properties. |

**Property typing.** `type ∈ {string, integer, number, boolean, enum:<VocabId>, ref:<ConceptType>}`. `cardinality ∈ {"1","0..1","0..n","1..n"}`. A `ref:` property is discouraged (relations are preferred) and reserved for tightly-owned 1:1 attributes.

## 2.4 Node (A-Box individual)

```json
{
  "id": "symptom:tomato_early_blight:primary:0",
  "type": "Symptom",
  "properties": {
    "canonical_label": "brown concentric-ring lesion",
    "source_field": "diagnostic_visual_features",
    "source_text": "target/bull's-eye concentric rings with yellow halo on older leaves",
    "observable": true
  }
}
```

Rules:
- `id` is **deterministic and stable** (see [10_build_algorithm.md](10_build_algorithm.md) §3): `"<typeslug>:<owner>:<discriminator>"`. Stable ids are essential for regression diffs and for downstream references.
- `type` must be a non-abstract concept type; for a `closed` type, `id` must be one of its declared individuals.
- `properties` must satisfy the type's property schema (required present, cardinality respected, enum values in-vocabulary).
- Nodes carry **no** edges inline; all relations live in `edges` (§2.6). This keeps nodes immutable value objects and edges independently indexable.

## 2.5 Relation type (T-Box)

```json
{
  "id": "has_symptom",
  "label": "has symptom",
  "domain": ["Condition"],
  "range":  ["Symptom"],
  "cardinality_out": "1..n",
  "cardinality_in":  "1",
  "symmetric": false,
  "transitive": false,
  "inverse": "symptom_of",
  "carries_confidence": true,
  "carries_evidence": true,
  "carries_flags": ["primary"]
}
```

Fields:

| Field | Meaning |
|-------|---------|
| `domain` / `range` | Allowed source / target concept types (a set; a subtype of a listed type also satisfies it via inheritance). |
| `cardinality_out` | How many targets a source node may have on this relation (e.g., a `Condition` has `1..n` symptoms). |
| `cardinality_in` | How many sources may point at a given target (e.g., a `Symptom` belongs to exactly `1` `Condition`; a `Color` value node has `0..n`). |
| `symmetric`, `transitive` | Logical properties used by the consistency checker (e.g., `mutually_exclusive_with` is symmetric; `part_of` is transitive). |
| `inverse` | Inverse relation id (for query convenience; the inverse is *derived*, not stored twice). |
| `carries_confidence` | If `true`, each edge of this type MUST have a `confidence` attribute. |
| `carries_evidence` | If `true`, each edge MUST reference `≥1` `Evidence` node. |
| `carries_flags` | Named boolean qualifiers permitted on the edge (e.g., `primary`, `image_licensed`). |

The full relation catalog is in [04_concept_graph.md](04_concept_graph.md); the domain/range matrix and forbidden pairs are in [05_rules.md](05_rules.md).

## 2.6 Edge (A-Box relation instance)

```json
{
  "id": "e:tomato_early_blight:has_symptom:primary:0",
  "type": "has_symptom",
  "source": "condition:tomato_early_blight",
  "target": "symptom:tomato_early_blight:primary:0",
  "attributes": {
    "confidence": "asserted",
    "flags": { "primary": true },
    "evidence": ["evidence:UCIPM-Tomato", "evidence:APS-CompTomato"]
  }
}
```

Rules:
- `type` must be a declared relation type; `source`/`target` node types must satisfy its `domain`/`range` (respecting inheritance).
- If the relation `carries_confidence`, `attributes.confidence ∈ {asserted, typical, hedged}` is **required**.
- If the relation `carries_evidence`, `attributes.evidence` must be a non-empty list of existing `Evidence` node ids.
- `flags` may only contain keys listed in the relation type's `carries_flags`.
- Edge `id` is deterministic (`"e:<owner>:<relation>:<discriminator>"`).

### Design decision: edge attributes vs reified statements

Confidence and evidence qualify a *relationship*, not a node. Two representations were considered:

- **(A) Reified `Statement` nodes** (RDF-style): each qualified assertion becomes a node `Statement(subject, predicate, object)` with `has_confidence`/`supported_by` edges. Maximally standard; enables statements *about* statements. **Cost:** roughly triples node/edge count, complicates every query, and is overkill for our monotonic needs.
- **(B) Edge attributes** (chosen): confidence and evidence live directly on the edge. Compact, fast to query, deterministic. **Cost:** cannot make statements about statements (we never need to), and requires a property-graph serialization (which we already chose).

We choose **(B)**. If full RDF interoperability is later required, edges map cleanly to **RDF-star** qualified triples (§1.6), so nothing is lost.

## 2.7 Closed vocabularies

A closed vocabulary declares the fixed individuals of a `closed` concept type, owned by a global policy file (not the DKB).

```json
{
  "id": "SeverityVocab",
  "for_type": "Severity",
  "policy_source": "assets/ontology/severity.json",
  "individuals": [
    { "id": "severity:mild",     "properties": { "canonical_label": "mild",     "stage_rank": 1 } },
    { "id": "severity:moderate", "properties": { "canonical_label": "moderate", "stage_rank": 2 } },
    { "id": "severity:severe",   "properties": { "canonical_label": "severe",   "stage_rank": 3 } }
  ]
}
```

Closed vocabularies in v1: `Severity`, `Extent`, `Confidence`, `Observability`, `EvidenceTier`, `SignType`, `LeafRegion`, `PlantPart`, `AgentCategory`. (Their members are enumerated in [03_concept_hierarchy.md](03_concept_hierarchy.md).) `Color`, `Shape`, `Size`, `Texture`, `Distribution` are **open** value classes populated (and canonicalized) from the DKB controlled vocabularies — see [07_crop_independence.md](07_crop_independence.md) for why they are open but bounded.

## 2.8 Constraints block

Global constraints referenced by the validator (full text in [05_rules.md](05_rules.md)):

```json
{
  "cardinality":        [ /* per relation, mirrors relation_types */ ],
  "domain_range":       [ /* per relation */ ],
  "mutual_exclusion":   [ ["sign:coating.white_powdery", "sign:coating.black_sooty"], … ],
  "observability_rule": { "leaf_regions_observable": true, "non_leaf_parts_observable": false },
  "required_presence":  [ { "type": "Condition", "must_have_out": ["has_symptom"], "min": 1 }, … ],
  "acyclic_relations":  ["is_a", "part_of"]
}
```

## 2.9 Inheritance semantics

- **Single inheritance** among concept types via `is_a`, forming a tree rooted at `Entity` (validator enforces acyclicity and single parent).
- **Property inheritance:** a node of type `T` must satisfy the union of properties declared on `T` and all its ancestors. Example: `Disease is_a Condition`; a `Disease` node satisfies `Condition`'s properties plus any `Disease`-specific ones.
- **Domain/range inheritance:** a relation whose `domain` lists `Condition` accepts any subtype (`Disease`, `PestDamage`, `SurfaceColonization`, `HealthyState`). This is what makes the relation catalog crop- and condition-kind-independent.
- **No multiple inheritance, no mixins.** Deliberately excluded to keep validation decidable and diffs simple. If a concept genuinely needs two parents, model the second relationship as an explicit relation (e.g., not `X is_a A,B` but `X is_a A` + `X plays_role B`).

## 2.10 Extensibility

The schema is extensible along four axes **without breaking existing consumers** (all are additive → minor `schema_version`; see [08_versioning.md](08_versioning.md)):

1. **New concept type** as a leaf under an existing parent (e.g., a future `Oomycete`-specific sign). Existing nodes/edges unaffected.
2. **New relation type** (e.g., `co_occurs_with`). Consumers that don't query it ignore it.
3. **New closed-vocabulary individual** (e.g., a new `LeafRegion`). Additive if it only *enables* new edges.
4. **New data property** on a concept type, if optional (`required:false`). Required-property additions or any removal/rename are **breaking** (major).

Extension is **policy-driven, not code-driven**: adding a concept type or relation is an edit to `ontology_schema.json` (T-Box) and its owning policy file, not to the builder's control flow. The builder reads the T-Box; it does not hard-code the type list. This is the property that lets the ontology reach "hundreds of diseases, dozens of crops" without a rewrite.
