# 7. Crop Independence

**Claim:** adding a crop (or a disease, or a whole new dataset) requires **only** new DKB entries. The ontology schema (T-Box), the builder, the closed vocabularies, and the anatomy model are untouched. This document proves the claim by construction.

## 7.1 The invariance principle

```
Ontology = f( DKB , GlobalPolicies )

where
  GlobalPolicies = T-Box schema + closed vocabularies + observability lexicon
                 + severity/extent split + sign-type keyword map + canonicalization tables
                 + evidence-tier map

Adding a crop  ⇒  only DKB grows (new condition records with the same 46-field schema)
               ⇒  GlobalPolicies unchanged, f unchanged
               ⇒  T-Box unchanged; A-Box gains instances of existing types
```

Because `f` reads the T-Box (it does not hard-code the type list — [02_schema.md](02_schema.md) §2.10) and the DKB schema is fixed at 46 fields, a new crop flows through the exact same code path as tomato and mango.

## 7.2 What is shared vs what differs

| Layer | Shared across all crops? | Populated from |
|-------|--------------------------|----------------|
| Concept types (T-Box) | ✓ identical | hand-designed once |
| Relation types | ✓ identical | hand-designed once |
| Constraints / rules | ✓ identical | hand-designed once |
| Anatomy (`Leaf`, `LeafRegion`, `PlantPart`) | ✓ identical | closed vocab (all leaves have margins, tips, veins, adaxial/abaxial) |
| `SignType`, `Severity`, `Extent`, `Confidence`, `Observability`, `AgentCategory`, `EvidenceTier` | ✓ identical | closed vocab |
| `Color`/`Shape`/`Texture`/`Size`/`Distribution` value nodes | ✓ **reused** (shared value nodes) | DKB controlled vocab, canonicalized |
| `Crop` nodes | differ (one per crop) | DKB `crop` field |
| `Condition`, `Symptom`, `CausalAgent`, `PathogenFamily`, `Evidence`, `EnvironmentalCondition` nodes | differ (per disease) | DKB records |

Everything in the "shared" rows is where redesign risk would live — and none of it depends on which crop we process.

## 7.3 Tomato and Mango share the ontology (worked comparison)

Both crops instantiate the **same concept types** and **reuse the same value nodes**:

```
crop:tomato ──affects─◄ condition:tomato_early_blight (Disease)
                              └ has_symptom → symptom → has_color → color:black ─┐
                                                        appears_on → region:margin│  SHARED
crop:mango  ──affects─◄ condition:mango_anthracnose (Disease)                    │  value nodes
                              └ has_symptom → symptom → has_color → color:black ─┘
                                                        appears_on → region:margin
```

- `color:black`, `region:margin`, `sign:lesion` are **one node each**, referenced by both a tomato disease and a mango disease. The graph literally *unifies* the two crops at the vocabulary level.
- The **only** structural difference between the tomato and mango subgraphs is *which conditions/symptoms exist* and *what they connect to* — i.e., the DKB content. The type system is identical.
- Condition-kind differences (mango has `SurfaceColonization` for sooty mould, `PestDamage` for cutting weevil/gall midge; tomato has `PestDamage` for spider mites) are handled by the **same four `Condition` subtypes** — already in the T-Box, not crop-specific.

## 7.4 Adding a third crop (procedure, zero redesign)

To add, say, **grape**:

1. Author a `grape` section in the DKB using the **existing 46-field schema** (the DKB's own extensibility, already proven for tomato+mango).
2. Ensure any *new* controlled-vocabulary surface strings map through the existing canonicalization tables; if grape introduces a genuinely new color/texture, add that value to the relevant open vocabulary policy (an additive, minor change — not a schema change).
3. Run `plantdx ontology build`. The builder produces grape's subgraph via the same path.

No new concept type, no new relation, no builder edit. If — and only if — grape required a *fundamentally new kind of observation* (e.g., a "bloom/wax" sign type absent from `SignType`), that would be a **schema extension** ([08_versioning.md](08_versioning.md), minor version): add one closed-vocab individual and, if needed, one compatibility-table row. This is the rare, bounded, additive case — never a redesign.

## 7.5 Why the anatomy generalizes (a subtle point)

One might worry that leaf anatomy differs across crops (a mango leaf ≠ a tomato leaflet). The ontology sidesteps this by modeling anatomy at the **functional-region** level that is universal to foliar disease description: every leaf has a lamina, margin, tip, midrib, veins, interveinal areas, and adaxial/abaxial surfaces. These are exactly the regions the DKB and pathology literature use. Crop-specific gross morphology (compound vs simple leaf, leaflet count) is **not** modeled because no caption/observability decision depends on it. This is a deliberate abstraction: model the regions that carry diagnostic meaning, not the botanical detail. It is why one `LeafRegion` vocabulary serves all crops.

## 7.6 Falsifiability of the claim

The crop-independence claim is testable, not aspirational (see [12_testing_strategy.md](12_testing_strategy.md)):
- **Regression test:** build with `{tomato}` only, then with `{tomato, mango}`; assert the *tomato subgraph is byte-identical* (adding mango does not perturb tomato). This proves additivity.
- **Schema-diff test:** assert `ontology_schema.json` is byte-identical before and after adding a crop (only `ontology_graph.json` changes).
- **New-crop smoke test:** a synthetic third crop DKB fragment builds and validates with **zero** schema edits.

If any of these fail, crop independence is violated and the design has a defect — the tests make that impossible to miss.
