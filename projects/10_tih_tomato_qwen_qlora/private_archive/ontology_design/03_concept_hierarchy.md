# 3. Concept Hierarchy

This is the complete `is_a` taxonomy of **concept types** (the T-Box). It is crop-independent and frozen except by a deliberate `schema_version` change. Instances (A-Box) are created only for non-abstract types.

Legend: **(abstract)** = no direct instances; **(closed)** = fixed enumeration owned by a policy file; **(open)** = populated from the DKB.

## 3.1 The full tree

```
Entity (abstract)
├── BiologicalTaxon (abstract)
│   ├── Crop                         (open)      — Tomato, Mango, …
│   └── CausalAgent (abstract)
│       ├── Pathogen (abstract)
│       │   ├── Bacterium            (open)
│       │   ├── Fungus               (open)
│       │   ├── Oomycete             (open)
│       │   └── Virus                (open)
│       ├── Pest (abstract)
│       │   ├── ArthropodPest        (open)
│       │   └── InsectPest           (open)
│       ├── Saprophyte               (open)      — e.g. sooty-mould complex
│       └── NoAgent                  (closed)    — the "healthy" placeholder agent
│
├── PathogenFamily                   (open)      — Pleosporaceae, Xanthomonadaceae, …
│
├── Condition (abstract)             — a diagnosable leaf state (one per DKB class)
│   ├── Disease                      (open)      — pathogen-caused
│   ├── PestDamage                   (open)      — arthropod/insect feeding or galling
│   ├── SurfaceColonization          (open)      — superficial (sooty mould)
│   └── HealthyState                 (open)      — the healthy class
│
├── Observation (abstract)
│   └── Symptom                      (open)      — a named, composite leaf observation
│
├── Quality (abstract)               — descriptive dimensions of a Symptom
│   ├── Color                        (open, bounded)
│   ├── Shape                        (open, bounded)
│   ├── Size                         (open, bounded)
│   ├── Texture                      (open, bounded)
│   ├── Distribution                 (open, bounded)   — spatial pattern/density
│   └── MorphologyModifier           (open, bounded)   — e.g. concentric, raised, shot-hole
│
├── Anatomy (abstract)
│   ├── PlantPart                    (closed)
│   │   ├── Leaf                     (closed)
│   │   ├── Fruit                    (closed)   ┐
│   │   ├── Stem                     (closed)   │  non-leaf parts:
│   │   ├── Twig                     (closed)   │  observable = false
│   │   ├── Flower                   (closed)   │  in the single-leaf
│   │   ├── Root                     (closed)   │  dataset context
│   │   └── WholePlant               (closed)   ┘
│   └── LeafRegion                   (closed)   — all observable
│       ├── Lamina                   (closed)
│       ├── Margin                   (closed)
│       ├── Tip                      (closed)
│       ├── Midrib                   (closed)
│       ├── Vein                     (closed)
│       ├── Interveinal              (closed)
│       ├── AdaxialSurface           (closed)   — upper
│       └── AbaxialSurface           (closed)   — underside
│
├── SignType                         (closed)   — the atomic visual kind of a Symptom
│   ├── Lesion                       (closed)
│   ├── Coating                      (closed)
│   ├── Gall                         (closed)
│   ├── Stippling                    (closed)
│   ├── Cut                          (closed)
│   ├── Deformation                  (closed)
│   ├── Mottle                       (closed)
│   └── HealthySurface               (closed)
│
├── Epistemic (abstract)
│   ├── Severity                     (closed)   — mild, moderate, severe
│   ├── Extent                       (closed)   — few, scattered, numerous, coalescing, extensive
│   ├── Confidence                   (closed)   — asserted, typical, hedged
│   └── Observability                (closed)   — observable, non_observable
│
├── AgentCategory                    (closed)   — none, bacterium, fungus, oomycete,
│                                                 virus, arthropod_pest, insect_pest,
│                                                 saprophytic_fungus  (mirrors DKB)
│
├── EnvironmentalCondition           (open)     — favouring conditions (temp/humidity/wetness)
│
└── Evidence (abstract)
    ├── PeerReviewed                 (open)
    ├── ExtensionService             (open)
    └── Textbook                     (open)
```

## 3.2 Design notes on the hierarchy

### Why `Condition` splits into four subtypes
The DKB distinguishes `is_pathogen_disease` and `agent_category`. This distinction is *scientifically load-bearing* and must be structural, because it changes what a caption may say:
- `Disease` → infection vocabulary (lesion, chlorosis, necrosis).
- `PestDamage` → mechanical/feeding vocabulary (stippling, cut, gall) — **never** "infection/lesion".
- `SurfaceColonization` → superficial-coating vocabulary; the underlying tissue is healthy — **never** "tissue necrosis".
- `HealthyState` → absence-of-signs vocabulary only.

Encoding this as four subtypes (rather than a property) lets relation domains and consistency rules be expressed per subtype (e.g., "a `PestDamage` MUST NOT `caused_by` a `Pathogen`").

### Why `SignType` is a closed class of individuals, not subclasses of `Symptom`
A `Symptom` *has a* sign type (`has_sign_type → SignType`); it *is not a* subclass of it. This keeps `Symptom` a single open class (one per DKB symptom phrase) while `SignType` stays a small closed vocabulary we can enumerate, index, and validate against. Making `Lesion`/`Coating`/… subclasses of `Symptom` would explode the class count and couple the taxonomy to the vocabulary. Trade-off accepted: we lose "a lesion is-a symptom" subsumption but gain a clean, queryable `has_sign_type`.

### Why `Quality` classes are "open but bounded"
`Color`, `Shape`, `Texture`, `Size`, `Distribution`, `MorphologyModifier` are populated from the DKB's controlled vocabularies (`color_vocabulary`, `shape_vocabulary`, …). They are **open** (new values can appear when a new disease/crop introduces a color) but **bounded** (a real controlled vocabulary saturates fast — there are only so many leaf colors). They are canonicalized into shared value nodes (`color:brown` is one node), which is what turns the structure into a graph (see [04_concept_graph.md](04_concept_graph.md)). See [06_statistics.md](06_statistics.md) for the saturation model.

### Why non-leaf `PlantPart`s exist at all
The dataset is single-leaf images. Fruit/twig/flower/etc. never appear. We still model them because the DKB's `forbidden_symptoms_not_leaf_observable` are exactly symptoms whose `appears_on` is a non-leaf part. Representing those parts lets us encode "not observable" *structurally* and lets a caption validator *prove* a forbidden symptom is forbidden. Non-leaf parts carry `observable=false`; leaf regions carry `observable=true`.

### Why `Evidence` is a three-way split
The DKB `references` object groups citations into `recent_research`, `extension_service`, `textbook`, and the `reference_registry` gives each a tier. `EvidenceTier` (closed) plus `Evidence` subtypes let the paper report *source quality* (e.g., "% of hallmark symptoms with peer-reviewed support") as a graph query.

## 3.3 Closed-vocabulary member enumeration (v1, complete)

These are owned by policy files, **not** the DKB. They are the same for every crop.

| Closed class | Members (individuals) |
|--------------|-----------------------|
| `AgentCategory` | none, bacterium, fungus, oomycete, virus, arthropod_pest, insect_pest, saprophytic_fungus |
| `SignType` | lesion, coating, gall, stippling, cut, deformation, mottle, healthy_surface |
| `LeafRegion` | lamina, margin, tip, midrib, vein, interveinal, adaxial_surface, abaxial_surface |
| `PlantPart` | leaf, fruit, stem, twig, flower, root, whole_plant |
| `Severity` | mild, moderate, severe |
| `Extent` | few, scattered, numerous, coalescing, extensive |
| `Confidence` | asserted, typical, hedged |
| `Observability` | observable, non_observable |
| `EvidenceTier` | peer_reviewed, extension_service, textbook |

## 3.4 Mapping the user's example hierarchy to this design

The task sketched: `Plant → Crop → Disease → Symptom → Morphology → Color → Distribution → Leaf Region → Confidence → Severity → Evidence`. That sketch conflates the `is_a` axis with the `has_*` (graph) axis. This design separates them:
- The **`is_a` axis** (this document) is a tree of *types*.
- The **`has_*` axis** ([04_concept_graph.md](04_concept_graph.md)) is the *graph* linking a `Condition` to its `Symptom`s, their `Quality`s, their `LeafRegion`s, and the `Confidence`/`Severity`/`Evidence` qualifiers.

Every level in the sketch is present: `Plant`→`Crop` (taxon branch), `Disease`/`Symptom` (condition + observation branches), `Morphology`/`Color`/`Distribution` (quality branch), `Leaf Region` (anatomy branch), `Confidence`/`Severity` (epistemic branch), `Evidence` (evidence branch). The relationships that connect them are the subject of the next document.
