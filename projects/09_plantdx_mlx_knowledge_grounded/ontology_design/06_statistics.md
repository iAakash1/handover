# 6. Ontology Statistics

All counts are **estimates from the current DKB** (18 conditions: 10 tomato + 8 mango) and a **growth model** for future scale. Exact counts are produced by the builder into `ontology_stats.json`; the numbers here are the design's sizing expectations and the basis for the "does it scale?" argument.

## 6.1 T-Box size (fixed, crop-independent)

| Category | Count (v1) |
|----------|-----------|
| Concept types (total) | ~52 |
| — abstract | ~9 (Entity, BiologicalTaxon, CausalAgent, Pathogen, Pest, Condition, Observation, Quality, Anatomy, Epistemic, Evidence) |
| — closed classes | 9 (+ their ~45 fixed individuals) |
| — open classes | ~14 |
| Relation types | ~21 |
| Global constraints | ~30 (F1–F10, C1–C7, cardinality, domain/range, acyclicity) |

The T-Box is **O(1)** in the number of diseases and crops — it does not grow with data. This is the single most important scalability property.

## 6.2 A-Box size for the current 18 conditions

Per-condition averages are read off the DKB field lengths (primary ~2–4, secondary ~1–3, diagnostic ~2–3, forbidden ~5–8, differentials ~3–4, references ~3).

| Node category | Per condition (avg) | Shared? | Total (18) |
|---------------|--------------------|---------|-----------|
| Condition | 1 | no | 18 |
| CausalAgent | ~1 | few shared | ~16 |
| PathogenFamily | ~0.7 | shared | ~9 |
| Symptom (observable) | ~6 | no | ~110 |
| Symptom (non-observable / forbidden) | ~6 | no | ~110 |
| Crop | — | shared | 2 |
| Color / Shape / Size / Texture / Distribution / Morphology value nodes | — | **shared** | ~70 (saturating) |
| LeafRegion | — | shared (closed) | 8 |
| PlantPart | — | shared (closed) | 7 |
| Severity / Extent / Confidence / Observability | — | shared (closed) | 3+5+3+2 = 13 |
| AgentCategory | — | shared (closed) | 8 |
| Evidence | — | shared | ~21 (= reference_registry) |
| EnvironmentalCondition | ~1 | some shared | ~14 |
| **Total nodes** | | | **≈ 400–460** |

| Edge category | Estimate (18) |
|---------------|--------------|
| `has_symptom` | ~220 (= #symptoms) |
| `has_sign_type` | ~220 |
| `has_{color,shape,size,texture,distribution,morphology}` | ~700 (avg ~3 quality edges per observable symptom) |
| `appears_on` + `has_observability` | ~440 |
| `caused_by` + `agent_in_category` + `member_of_family` + `affects` | ~70 |
| `has_extent` + `typical_at_severity` | ~120 |
| `differentiated_from` | ~65 |
| `favored_by` | ~40 |
| `part_of` (anatomy, fixed) | ~14 |
| `mutually_exclusive_with` (fixed) | ~8 |
| **Total edges** | **≈ 2,000–2,600** |

**Relationship-type usage** (how many of the ~21 relation types are exercised by 18 conditions): all except possibly `member_of_family` for pest classes — expect ~19–21 in use.

## 6.3 Coverage metrics (reported every build)

These are the paper's "did we faithfully encode the DKB?" instruments:

| Metric | Definition | Target |
|--------|-----------|--------|
| **DKB field coverage** | fraction of the 46 per-disease fields consumed into nodes/edges or explicitly declared non-structural | 100% (validator-enforced, [09](09_validation.md)) |
| **Symptom evidence coverage** | fraction of `has_symptom` edges with `evidence ≥ 1` | 100% (hard rule R5) |
| **Hallmark peer-review coverage** | fraction of `primary` `has_symptom` edges whose evidence includes a `PeerReviewed`/`ExtensionService` node | reported (a quality metric, not gated) |
| **Observable ratio** | observable symptoms / all symptoms per condition | reported per condition |
| **Differential coverage** | fraction of DKB `confused_with` entries realized as `differentiated_from` edges with a `via_symptom` qualifier | 100% |
| **Vocabulary reuse factor** | (Σ quality-edge count) / (distinct value nodes) — measures graph sharing | reported (higher ⇒ more graph-like) |
| **Determinism check** | rebuild `content_hash` == previous | must match |

## 6.4 Growth model (to "hundreds of diseases, dozens of crops")

Let `D` = number of conditions, `K` = number of crops. Then:

| Quantity | Growth | Reason |
|----------|--------|--------|
| Concept types (T-Box) | **O(1)** | schema is fixed |
| Relation types | **O(1)** | fixed |
| Condition / Symptom / Evidence nodes | **O(D)** | one condition, ~12 symptoms, few citations each |
| `has_*` edges | **O(D)** | edges per symptom is bounded |
| Crop nodes | **O(K)** | trivially |
| Quality value nodes (Color, Shape, …) | **O(1) asymptotically** | controlled vocabularies *saturate* — there are only so many leaf colors/shapes/textures; new diseases reuse existing value nodes |
| Anatomy / closed vocab nodes | **O(1)** | fixed enumerations |

**Projection.** For `D = 500` conditions across `K = 12` crops:
- Nodes ≈ `500 × ~12 symptoms` + `~500 conditions` + `~300 agents/families` + `~150 saturated value/closed nodes` + `~120 evidence` ≈ **~7,000–8,000 nodes**.
- Edges ≈ `~5×` node-ish along the spine ≈ **~35,000–45,000 edges**.

Both are trivial for an in-memory property graph (megabytes). The "millions of captions" figure is a *downstream* dataset scale, not an ontology scale — the ontology is queried O(1) times per disease, not per caption. **The design comfortably meets the stated long-term scale with no architectural change.**

## 6.5 Why the vocabulary saturates (the key scalability claim)

Quality value nodes are the only A-Box category that could in principle grow unboundedly. They do not, because they are drawn from *controlled* vocabularies (the DKB's `color_vocabulary`, `shape_vocabulary`, etc.), which are curated and small by construction. Empirically, leaf-disease description across all of plant pathology uses on the order of tens of colors, ~dozen shapes, ~dozen textures. Adding the 500th disease introduces essentially **zero** new value nodes. This is what keeps the graph dense (high reuse factor) and small even at large `D` — and it is a direct consequence of the "closed vocabulary" invariant inherited from the DKB and Caption Framework.
