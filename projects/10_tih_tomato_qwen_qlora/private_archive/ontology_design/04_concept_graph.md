# 4. Concept Graph (Semantic Relations)

This document defines the **relation types** (the edges of the graph) and shows a **fully worked instance subgraph**. Together with [03_concept_hierarchy.md](03_concept_hierarchy.md) (the nodes) this is the complete semantic graph.

## 4.1 Relation catalog (T-Box)

Each relation lists domain → range, out/in cardinality, and whether edges of that type carry confidence (`C`), evidence (`E`), and named flags (`F`). Subtypes of a listed domain/range are accepted via inheritance.

| Relation | Domain → Range | out : in | C | E | Flags | Notes |
|----------|----------------|----------|---|---|-------|-------|
| `is_a` | Type → Type (T-Box) | 1 : 0..n | – | – | – | Taxonomy; acyclic tree. |
| `instance_of` | Node → Type | 1 : 0..n | – | – | – | Implicit via `node.type`; not stored as an edge. |
| `affects` | Condition → Crop | 1..n : 0..n | asserted | ✓ | – | The crop(s) a condition occurs on. |
| `caused_by` | Condition → CausalAgent | 0..n : 0..n | ✓ | ✓ | `disputed` | `0..n`: HealthyState has none; disputed taxonomy → parallel edges. |
| `agent_in_category` | CausalAgent → AgentCategory | 1 : 0..n | asserted | – | – | Mirrors DKB `agent_category`. |
| `member_of_family` | Pathogen → PathogenFamily | 0..1 : 0..n | asserted | ✓ | – | DKB `pathogen_family`. |
| `has_symptom` | Condition → Symptom | 1..n : 1 | ✓ | ✓ | `primary` | The core clinical edge; `in=1` (a symptom belongs to one condition). |
| `has_sign_type` | Symptom → SignType | 1 : 0..n | asserted | – | – | Exactly one sign type per symptom. |
| `has_color` | Symptom → Color | 0..n : 0..n | ✓ | ✓ | – | Shared `Color` value nodes. |
| `has_shape` | Symptom → Shape | 0..n : 0..n | ✓ | ✓ | – | |
| `has_size` | Symptom → Size | 0..1 : 0..n | ✓ | ✓ | – | |
| `has_texture` | Symptom → Texture | 0..n : 0..n | ✓ | ✓ | – | |
| `has_distribution` | Symptom → Distribution | 0..n : 0..n | ✓ | ✓ | – | Spatial pattern/density. |
| `has_morphology` | Symptom → MorphologyModifier | 0..n : 0..n | ✓ | ✓ | – | e.g. concentric, raised, shot-hole. |
| `appears_on` | Symptom → (LeafRegion ∪ PlantPart) | 1..n : 0..n | ✓ | ✓ | – | Determines observability (§4.3). |
| `has_observability` | Symptom → Observability | 1 : 0..n | asserted | – | – | Derived, then stored for fast query. |
| `has_extent` | Condition → Extent | 0..n : 0..n | typical | ✓ | `image_licensed` | Visible density; `image_licensed=true`. |
| `typical_at_severity` | Condition → Severity | 0..n : 0..n | typical | ✓ | `image_licensed` | **Always** `image_licensed=false` (severity honesty). |
| `differentiated_from` | Condition → Condition | 0..n : 0..n | ✓ | ✓ | `via_symptom` | Confusable pairs; qualifier points to distinguishing symptom(s). |
| `favored_by` | Condition → EnvironmentalCondition | 0..n : 0..n | typical | ✓ | – | Optional; from DKB `environmental_conditions`. |
| `part_of` | (LeafRegion ∪ PlantPart) → PlantPart | 0..1 : 0..n | asserted | – | – | Anatomy mereology; transitive; acyclic. |
| `supported_by` | (edge) → Evidence | (edge attr) | – | – | – | Realized as the `evidence` edge-attribute, not a node-edge. |
| `mutually_exclusive_with` | SignType → SignType | 0..n : 0..n | asserted | – | – | Symmetric; drives consistency (white_powdery ⟂ black_sooty). |

**Inverses** (derived, not stored): `symptom_of` (has_symptom⁻¹), `causes` (caused_by⁻¹), `host_of` (affects⁻¹), `region_of` (part_of⁻¹). Consumers may materialize inverses in memory; the on-disk graph stores each relation once (determinism + no update anomalies).

## 4.2 Why these relations (and not fewer/more)

- **One clinical spine** (`Condition –has_symptom→ Symptom –has_*→ Quality / –appears_on→ Region`) carries the bulk of the knowledge. Everything a caption needs is a 2–3 hop query along this spine.
- **Quality relations are separated by axis** (`has_color`, `has_shape`, …) rather than a single generic `has_quality`. Trade-off: more relation types, but each is independently constrained, queryable ("all brown symptoms"), and validated. A single generic relation would lose the axis and force string parsing downstream — exactly what the ontology exists to avoid.
- **`appears_on` targets a union** of `LeafRegion` and `PlantPart`. This is the deliberate mechanism for observability: a leaf-region target ⇒ observable; a non-leaf-part target ⇒ not. One relation, two meanings resolved structurally.
- **`differentiated_from` is first-class**, not derived from `confused_with` prose, because the *distinguishing feature* (the `via_symptom` flag/qualifier) is what the caption's differential clause needs and what the diagnostic evaluation split is built on.
- **`typical_at_severity` exists but is permanently non-image-licensed.** We represent the DKB severity staging (so the knowledge is not lost) while structurally forbidding any per-image stage claim. Removing the relation would lose knowledge; leaving it unflagged would risk dishonest captions. The flag is the honest middle path.

## 4.3 The observability derivation (structural, deterministic)

```
observable(symptom) :=
    true   if every appears_on target of the symptom is a LeafRegion
    false  if any appears_on target is a non-leaf PlantPart
```

The builder computes this and materializes it as `has_observability` and the `observable` node property. Consumers never recompute it. This single rule is the formal statement of the project's central scientific constraint (single-leaf grounding), and it is *checkable*: [09_validation.md](09_validation.md) asserts that every symptom's stored `observable` equals the recomputed value.

## 4.4 Worked instance subgraph — Tomato Early Blight

Node ids abbreviated. `[C]`=confidence, `[E]`=evidence, `[F]`=flags on the edge.

```
                         (Crop) crop:tomato
                               ▲
                               │ affects  [C:asserted]
                               │
   (AgentCategory) cat:fungus ◄──agent_in_category── (Fungus) agent:alternaria_solani
        ▲                                                   ▲
        │                                                   │ caused_by [C:asserted, E:APS-CompTomato]
        │ (member_of_family)                                │
   (PathogenFamily) fam:pleosporaceae ◄────────────────  (Disease) condition:tomato_early_blight
                                                              │
        ┌─────────────────────────────────────────────┬──────┴───────────────┬───────────────────────┐
        │ has_symptom [C:asserted, F:primary,          │ has_symptom          │ typical_at_severity   │
        │              E:UCIPM-Tomato, APS-CompTomato]  │ [C:hedged]           │ [C:typical,           │
        ▼                                               ▼                      │  F:image_licensed=FALSE]
  (Symptom) symptom:…:primary:0                 (Symptom) …:secondary:0        ▼
  "brown concentric-ring lesion"                "leaflet chlorosis"    (Severity) severity:{mild,moderate,severe}
  observable=true                               observable=true
        │                                               │
        ├─ has_sign_type ─────► (SignType) sign:lesion  ├─ has_sign_type ─► sign:lesion
        ├─ has_color ─────────► (Color) color:brown ◄───┼─ has_color ─────► color:yellow
        ├─ has_color ─────────► (Color) color:dark_brown│
        ├─ has_shape ─────────► (Shape) shape:circular  │
        ├─ has_morphology ────► (Morph) morph:concentric│
        ├─ has_morphology ────► (Morph) morph:target_like
        ├─ has_size ──────────► (Size)  size:moderate
        ├─ has_distribution ──► (Distribution) dist:lower_leaves_first
        └─ appears_on ────────► (LeafRegion) region:lamina   ⇒ observable=true

  condition:tomato_early_blight
        ├─ has_extent ────────► (Extent) extent:coalescing   [C:typical, F:image_licensed=TRUE, E:UCIPM-Tomato]
        └─ differentiated_from► (Disease) condition:tomato_target_spot
                                   [C:asserted, F:via_symptom=symptom:…:primary:0,
                                    E:UFIFAS-PP351]        (rings coarser & lower-leaf-first vs target spot)
```

Note the **shared value nodes**: `color:brown`, `color:yellow`, `sign:lesion`, `region:lamina` are single nodes reused by many symptoms across many diseases and crops. A query "every observable symptom that is brown and lesion-typed" is a two-hop traversal, no text parsing.

### Contrast subgraph — Mango Sooty Mould (a `SurfaceColonization`)

```
(SurfaceColonization) condition:mango_sooty_mould
   ├─ caused_by ─► (Saprophyte) agent:capnodium_complex   [C:asserted, E:Chomnunti2011]
   ├─ has_symptom [F:primary] ─► (Symptom) "black superficial coating"
   │        ├─ has_sign_type ─► (SignType) sign:coating
   │        ├─ has_color ─────► color:black
   │        ├─ has_texture ───► texture:velvety
   │        └─ appears_on ────► region:adaxial_surface     ⇒ observable=true
   └─ (no) typical_at_severity, (no) necrosis symptom      — tissue is healthy beneath
```

`sign:coating` here is `mutually_exclusive_with` the powdery-mildew white coating sign at the *value* level (black vs white), which the consistency checker uses to forbid a caption mixing them. The graph makes "sooty mould ≠ powdery mildew" a structural fact.

## 4.5 Query patterns the graph must support (acceptance for the design)

The relation set is sufficient iff each of these is a bounded-hop query returning deterministic results:

1. **Grounded symptom set** for caption generation: `condition –has_symptom→ Symptom[observable=true]`, partitioned by edge `confidence` (asserted → required, typical/hedged → optional).
2. **Forbidden set**: `condition –has_symptom→ Symptom[observable=false]` ∪ rivals' hallmark symptoms via `differentiated_from`.
3. **Controlled vocabulary** for a disease: value nodes reachable from its observable symptoms via `has_color|shape|size|texture|distribution|morphology`.
4. **Provenance of any claim**: any edge → its `evidence` attribute → `Evidence` node → `reference_registry`.
5. **Severity licensing**: `typical_at_severity` edges are all `image_licensed=false`; `has_extent` edges are `image_licensed=true`.
6. **Differential probe set**: all `differentiated_from` pairs with their `via_symptom` qualifier (feeds the evaluation's confusable split).
7. **Source-quality metric**: fraction of `has_symptom[primary]` edges whose `evidence` includes a `PeerReviewed`/`ExtensionService` node.

All seven are satisfied by the catalog above; this is the design's completeness argument.
