# 01 — Caption Ontology Specification (Task 1) + Vocabulary Expansion (Task 3)

**Deliverable 2 of 8.** Defines the concept model, the **deterministic derivation of the per-disease caption ontology from `dkb.json`**, and the **vocabulary-expansion lattice** that produces lexical diversity without semantic drift.

Core principle: **the Caption Ontology is a pure function of the DKB.** `ontology = derive(dkb.json) + global_policy + optional_overrides`. Disease facts are never re-typed in Stage 2; the builder reads DKB fields and reshapes them. If a fact must change, it changes in the DKB and the ontology is rebuilt.

---

## 1. Artifacts produced by the Ontology Builder (component A)

Written to `ontology/` (doc 06):

| File | Content |
|------|---------|
| `concept_schema.json` | The global concept type registry (§2). Static; not disease-specific. |
| `global_policy.yaml` | Global defaults: register rules, severity split lexicon, hedging list, default salience (§2.3, §5). |
| `caption_ontology.json` | The derived per-disease ontology records (§3), one per `disease_id` (18 total). Generated. |
| `overrides/<disease_id>.yaml` | Optional manual overrides merged last (§6). Empty by default. |
| `ontology_build_report.md` | Auto-generated audit: for each disease, which DKB fields fed which concepts; unused DKB fields; warnings. |

## 2. Concept schema (`concept_schema.json`)

A caption is a realized set of **concepts**. A concept is an atomic, checkable unit of meaning with a fixed backing in the DKB. The registry below is **global** (same 20 concept types for all diseases); which concepts are *required/optional/forbidden* is per-disease and derived (§3).

### 2.1 Concept registry

| concept_id | Backing DKB field(s) | Observable? | Register(s) | Default cardinality | Notes |
|-----------|----------------------|-------------|-------------|---------------------|-------|
| `disease_identity` | `class_label`,`common_name`,`disease` | identity (from label) | all | 1 (required) | The one fact we are always licensed to state. |
| `host` | `crop`,`host_plant` | yes | all | 0–1 | "tomato leaf" / "mango leaf". |
| `agent_reference` | `scientific_name`,`pathogen_type`,`agent_category` | **no** (cannot see species) | clinical, educational | 0–1 | Gated by register; must obey pest/pathogen (§4). |
| `agent_category_descriptor` | `agent_category`,`pathogen_type` | partial | clinical, educational | 0–1 | "a fungal disease", "mite infestation", "insect feeding damage". |
| `primary_sign` | `primary_symptoms` ∪ `diagnostic_visual_features` | yes | all | ≥1 (required) | The hallmark leaf-observable sign(s). |
| `lesion_color` | `leaf_color`,`color_vocabulary` | yes | all | 0–1 | Color axis (§7). |
| `lesion_shape` | `lesion_shape`,`shape_vocabulary` | yes | all | 0–1 | Shape axis. |
| `lesion_size` | `lesion_size` | yes | all | 0–1 | Size axis. |
| `lesion_distribution` | `lesion_distribution` | yes | all | 0–1 | "scattered", "numerous", "coalescing". |
| `leaf_location` | `leaf_margin_changes`,`lesion_distribution`,`texture_changes` | yes | all | 0–1 | Location axis (margin/tip/interveinal/underside/lamina). |
| `texture` | `texture_changes`,`texture_vocabulary` | yes | all | 0–1 | Texture axis. |
| `chlorosis` | `chlorosis` | yes | all | 0–1 | Yellowing. |
| `necrosis` | `necrosis` | yes | all | 0–1 | Tissue death. |
| `leaf_deformation` | `leaf_curling` | yes | all | 0–1 | Curling/cupping/distortion. |
| `secondary_sign` | `secondary_symptoms` | yes but not guaranteed | descriptive, educational | 0–2 | **Hedged** register only (§5). |
| `extent` | `severity_vocabulary` (extent bucket) | yes | all | 0–1 | Visible density; NOT a stage claim (§5). |
| `severity_stage` | `severity_vocabulary` (stage bucket), `severity.*` | **no** (per image) | severity-conditioned only | 0–1 | **Gated OFF by default** (§5). |
| `differential` | `confused_with`,`key_differentiating_features` | comparative | educational | 0–1 | "distinguished from X by Y". |
| `healthy_state` | (healthy classes) `primary_symptoms` | yes | all | 1 (required, healthy only) | Uniform green intact leaf. |
| `management` | `management_practices`,`treatment_recommendations`,`prevention_recommendations` | **no** | educational-appendix only | 0–1 | **OFF** for visual-description captions (default). |

### 2.2 Observable vs non-observable (global)
- **Observable** (may appear in any register): `disease_identity` (as label), `host`, `primary_sign`, `lesion_color/shape/size/distribution`, `leaf_location`, `texture`, `chlorosis`, `necrosis`, `leaf_deformation`, `secondary_sign` (hedged), `extent`, `healthy_state`, `differential` (comparative).
- **Non-observable** (never a *visual* claim; allowed only in the noted non-visual register, and never for `is_pathogen_disease:false` where it would misattribute): `agent_reference`, `agent_category_descriptor` (partial), `severity_stage`, `management`.

The builder writes both lists per disease so the validator (doc 03 V7) can enforce "no non-observable concept in a visual-register caption".

### 2.3 Salience (for coverage-weighted sampling, doc 00 §4)
Default salience is assigned by source field, overridable in `global_policy.yaml`:
- `diagnostic_visual_features`, `key_differentiating_features` → salience 1.0
- `primary_symptoms` → 0.9
- `lesion_shape/color/distribution/leaf_location/texture` → 0.6
- `chlorosis/necrosis/lesion_size/leaf_deformation/extent` → 0.5
- `secondary_symptoms` → 0.3
- `differential/agent_*` → 0.2
Salience biases *which optional concepts get chosen*; it does not change required/forbidden status.

### 2.4 Co-selection constraints (global, in `concept_schema.json`)
- `lesion_color`,`lesion_shape`,`lesion_size`,`texture` may be selected only if a `primary_sign` whose `sign_type` is `lesion|coating|gall|stippling` is selected (you cannot color a sign that isn't there). `sign_type` is tagged per primary_sign at build time (§3.2).
- `necrosis` and `chlorosis` may co-occur.
- Global mutual-exclusion set (should never co-occur; cross-disease guard): {`white_powdery_coating`, `black_sooty_coating`}, {`raised_gall`, `flat_lesion`}, {`clean_cut_margin`, `intact_margin`}, {`healthy_state`, any disease sign}. Encoded as `mutex_groups`.
- `differential` requires that its counterpart disease is in `confused_with`; the realized "distinguished from X" X must come from `confused_with`.

## 3. Per-disease ontology record (`caption_ontology.json`)

### 3.1 Record schema (generated; one per disease)
```json
{
  "disease_id": "tomato_early_blight",
  "crop": "tomato",
  "is_pathogen_disease": true,
  "agent_category": "fungus",
  "register_policy": {"visual": true, "clinical": true, "educational": true, "severity_conditioned": false},
  "required_concepts": ["disease_identity", "primary_sign"],
  "optional_concepts": ["host","lesion_color","lesion_shape","lesion_distribution","leaf_location","chlorosis","necrosis","texture","extent","secondary_sign","differential","agent_category_descriptor","agent_reference"],
  "forbidden_concepts": ["severity_stage","management","leaf_deformation_cupping","white_powdery_coating","black_sooty_coating","raised_gall","stippling","clean_cut_margin"],
  "observable_concepts": ["disease_identity","host","primary_sign","lesion_color","lesion_shape","lesion_size","lesion_distribution","leaf_location","texture","chlorosis","necrosis","extent","secondary_sign","differential"],
  "non_observable_concepts": ["agent_reference","agent_category_descriptor","severity_stage","management"],
  "min_information": 2,
  "max_information": 8,
  "required_medical_terminology": ["concentric rings","lesion"],
  "optional_descriptive_terminology": ["target-like","bull's-eye","halo","brown","circular"],
  "vocab_axes": {
    "color": ["brown","dark-brown","yellow"],
    "shape": ["circular","concentric","target-like","ringed"],
    "size": ["3–12 mm","enlarging"],
    "texture": ["dry","papery","velvety-centered"],
    "location": ["on lower leaves","on older leaves","across the leaflet"],
    "extent": ["a few","scattered","coalescing","extensive"],
    "severity_stage": ["mild","moderate","severe"]
  },
  "never_appear": ["water-soaked","greasy","powdery","webbed","mosaic","mottled","pycnidia","white sporulation","fruit lesion","fruit","stem collar rot","tear-stain","gall","cut","sooty"],
  "concept_realizations": {
    "primary_sign": [
      {"phrase":"brown circular lesions with concentric rings","source_field":"primary_symptoms","sign_type":"lesion"},
      {"phrase":"target-like concentric-ringed lesions","source_field":"diagnostic_visual_features","sign_type":"lesion"}
    ],
    "chlorosis": [{"phrase":"a surrounding yellow halo","source_field":"chlorosis"}],
    "differential": [{"phrase":"distinguished from Septoria leaf spot by the concentric rings and absence of pycnidia","source_field":"key_differentiating_features + confused_with"}]
  },
  "salience": {"primary_sign":0.95,"lesion_shape":0.6,"...":0.5},
  "co_selection": {"requires":{"lesion_color":["primary_sign:lesion"]}, "mutex_groups":[["white_powdery_coating","black_sooty_coating"]]},
  "provenance": {"dkb_sha256":"...","built_from_fields":["primary_symptoms","diagnostic_visual_features","..."]}
}
```

### 3.2 Derivation rules (DKB field → ontology element)
The builder applies these deterministic rules. **This table is the contract; implement exactly.**

| Ontology element | Derivation rule from DKB |
|------------------|--------------------------|
| `required_concepts` | Always `["disease_identity"]`. Add `"primary_sign"` if `primary_symptoms` non-empty and disease ≠ healthy. For `healthy_*`: `["disease_identity","healthy_state"]` (no primary_sign). |
| `optional_concepts` | Every concept whose backing field is non-empty for this disease, minus required, minus forbidden, minus non-observable concepts disallowed by register policy. |
| `forbidden_concepts` | (a) `severity_stage` and `management` always (unless flags flip); (b) every concept in the global mutex complement of this disease's selected primary signs; (c) any concept whose *only* realization would require a term in `never_appear`. |
| `observable_concepts` / `non_observable_concepts` | Partition of the disease's concept set using §2.2 (global) intersected with what's present. |
| `min_information` | `len(required_concepts)`. |
| `max_information` | `len(required_concepts ∪ optional_concepts)` capped at 8. |
| `required_medical_terminology` | The head terms of `diagnostic_visual_features` + `key_differentiating_features` reduced to canonical noun phrases (e.g., "concentric rings", "pycnidia", "angular lesion", "windowpane", "raised gall", "sooty coating"). These are terms a *correct* caption of this disease should be able to use and that a validator treats as the disease's identity terms. Deduplicated. |
| `optional_descriptive_terminology` | `recommended_adjectives` ∪ `recommended_controlled_vocabulary` minus the required-terminology set. |
| `vocab_axes.color` | `color_vocabulary`. |
| `vocab_axes.shape` | `shape_vocabulary`. |
| `vocab_axes.size` | tokenized from `lesion_size` (ranges/qualifiers). |
| `vocab_axes.texture` | `texture_vocabulary`. |
| `vocab_axes.location` | Union of location phrases extracted from `lesion_distribution` + `leaf_margin_changes` + underside/upper-surface mentions in `texture_changes`, normalized against `vocabulary/location_axis.json`. |
| `vocab_axes.extent` | `severity_vocabulary` ∩ extent lexicon (`vocabulary/severity_axis.json:extent`). |
| `vocab_axes.severity_stage` | `severity_vocabulary` ∩ stage lexicon (kept for the gated mode; not used by default). |
| `never_appear` | `forbidden_terms` ∪ `forbidden_adjectives` ∪ head-noun surfaces of `forbidden_symptoms_not_leaf_observable` ∪ (for `is_pathogen_disease:false`) `{infection, infected, pathogen, lesion?}` per pest rule (§4). |
| `concept_realizations` | For each concept, split its backing field's list items into caption-ready phrases, tagging `source_field` and, for `primary_sign`, a `sign_type ∈ {lesion, coating, gall, stippling, cut, deformation, mottle, healthy}` inferred from a global keyword map (`vocabulary/sign_type_map.json`). |
| `salience` | Per §2.3. |
| `register_policy` | `severity_conditioned=false` always (until a severity label pipeline exists). `visual=clinical=educational=true`. |

### 3.3 `sign_type` inference (needed for co-selection and register)
A small global keyword map assigns each `primary_sign`/`diagnostic_visual_feature` phrase a `sign_type`:
- contains {spot, lesion, blotch, patch(necrotic)} → `lesion`
- contains {powdery, sooty, coating, film, mold on underside, velvety mold} → `coating`
- contains {gall, wart, pimple, bump} → `gall`
- contains {stippling, flecking, bronzing, webbing} → `stippling`
- contains {cut, notch, windowpane, trimmed} → `cut`
- contains {curl, cup, distort, fern-leaf, roll} → `deformation`
- contains {mosaic, mottle} → `mottle`
- else (healthy) → `healthy`
This drives which descriptive concepts (color/shape/size/texture) are eligible (you can size a `lesion` or `gall`, but "size" of a `mottle` is odd → co-selection blocks it).

## 4. Pest / pathogen register rule (enforced at derivation)
For `is_pathogen_disease: false` (spider_mites, cutting_weevil, gall_midge, sooty_mould):
- `agent_reference` may name the **organism** (mite/weevil/midge/fungus-on-honeydew) but must use `agent_category` framing (`arthropod_pest`,`insect_pest`,`saprophytic_fungus`), never "pathogen"/"infection".
- The words `infection, infected, pathogen` are added to `never_appear`.
- `lesion` is added to `never_appear` **only** where the DKB `forbidden_terms` already lists it (cutting_weevil, gall_midge as "flat lesion", sooty_mould "necrotic lesion") — follow the DKB, do not over-block. (Spider mites: DKB forbids "lesion" too.)
- `primary_sign.sign_type` for these classes is `stippling/cut/gall/coating` (not `lesion`), which automatically routes descriptive concepts correctly.
For `sooty_mould` specifically, add a **consistency assertion** to the record: captions must be compatible with "tissue healthy beneath" (no `necrosis`/`chlorosis` concepts are in its optional set because DKB necrosis/chlorosis = "none").

## 5. Severity handling in the ontology (implements doc 00 §5)
- `severity_vocabulary` is split by the builder using `vocabulary/severity_axis.json`, which classifies each term as `extent` or `stage`. Example classification: `a few, scattered, numerous, coalescing, extensive, dense, patchy, continuous, complete-coating` → **extent**; `mild, moderate, severe, early, advanced, faint(as stage), slight(as stage)` → **stage**.
- `extent` → concept `extent` (optional, allowed). `stage` → concept `severity_stage` (forbidden by default).
- `secondary_sign` concept is flagged `hedged: true`; templates that fill it must be from the hedged register (doc 02 §4), forcing "often/may/can" phrasing.

## 6. Overrides (`overrides/<disease_id>.yaml`)
A thin, auditable escape hatch for the rare case where pure derivation is imperfect (e.g., a `lesion_size` string that tokenizes poorly). Overrides can only: (a) re-bucket a vocab term (extent↔stage), (b) add a phrase to `never_appear`, (c) refine a `concept_realization` phrase's wording, (d) adjust salience. Overrides **cannot** introduce a concept or term absent from the DKB (the builder validates this and fails otherwise), preserving invariant #2. Default: all override files empty.

## 7. Vocabulary Expansion (Task 3) — controlled lexical diversity without drift

Expansion turns a realized phrase into lexically varied but semantically identical alternatives. It operates over **axes** and **equivalence classes**, all disease-filtered.

### 7.1 Equivalence classes (`vocabulary/synonyms.json`)
A global table of synonym sets, each tagged with the axis it belongs to and any disease-level restrictions. A synonym may be used for disease `d` only if none of its members are in `never_appear[d]`.
```json
{
  "HEAD_lesion":   {"members":["lesion","spot","blotch","mark"], "axis":"head_noun",
                    "restrict":{"forbid_for":["*any where 'spot' in never_appear*"]}},
  "HEAD_coating":  {"members":["coating","film","layer","growth"], "axis":"head_noun"},
  "HEAD_gall":     {"members":["gall","wart-like bump","pimple-like swelling","nodule"], "axis":"head_noun"},
  "VERB_present":  {"members":["shows","displays","exhibits","bears","is marked by"], "axis":"predicate"},
  "DET_many":      {"members":["numerous","many","multiple","abundant"], "axis":"extent"},
  "CONN":          {"members":["with","showing","featuring","accompanied by"], "axis":"connective"}
}
```
Rule: substitution replaces a member with another member **of the same class only**. Cross-class substitution is prohibited (prevents "lesion"→"gall" drift).

### 7.2 Modifier axes (`vocabulary/modifiers.json`)
Five ordered modifier axes attach to a head noun; each axis's allowed values for disease `d` come **only** from that disease's `vocab_axes` (§3.1):
`size → shape → color → texture → [head noun]`, plus a detachable `extent` quantifier before the noun phrase and a `location` PP after it.

Canonical realized order (English adjective order, fixed): `[extent] [size] [shape] [color] [texture] HEAD [location-PP]`.
Example slotting for early blight: `[scattered] [_] [concentric] [brown] [_] lesions [across the leaflet]` → "scattered concentric brown lesions across the leaflet".

### 7.3 The expansion lattice (typed edges)
Expansion is a walk on a DAG whose nodes are phrases and whose edges are **typed, meaning-preserving operations**:

| Edge type | Operation | Constraint |
|-----------|-----------|------------|
| `SUBST_syn` | replace a token with a same-class synonym | class member ∉ `never_appear[d]` |
| `ADD_color` | attach a color modifier | value ∈ `vocab_axes.color[d]`; slot empty; primary_sign.sign_type ∈ {lesion,coating,gall} |
| `ADD_shape` | attach a shape modifier | value ∈ `vocab_axes.shape[d]`; slot empty |
| `ADD_size` | attach a size modifier | value ∈ `vocab_axes.size[d]`; slot empty |
| `ADD_texture` | attach a texture modifier | value ∈ `vocab_axes.texture[d]`; slot empty |
| `ADD_extent` | prepend an extent quantifier | value ∈ `vocab_axes.extent[d]` (never a `stage` term) |
| `ADD_location` | append a location PP | value ∈ `vocab_axes.location[d]` |
| `REORDER_list` | permute a coordinated list of signs | permutation only; no lexical change |

**No-drift invariants (validator V6/V7 also enforce):**
1. Every edge preserves the head noun's concept (`SUBST_syn` stays within class; ADD_* never changes the head).
2. Every added modifier value is drawn from that disease's `vocab_axes` — i.e., from the DKB. Nothing is invented.
3. The phrase's concept set after expansion equals its concept set before (expansion changes surface form, not which concepts are asserted). Adding a color modifier does **not** add a new concept unless `lesion_color` was already selected by the concept selector; if it wasn't selected, `ADD_color` is disabled for this caption. **Expansion may only realize already-selected concepts more richly; it may never introduce an unselected concept.** This is the single most important rule and closes the drift loophole in the Task-3 example.
4. Max modifier depth per head noun = 3 (config `max_adjectives`) to avoid unnatural stacking.

### 7.4 Worked example (corrected vs the drift trap)
Task-3 prompt sketch was: `brown lesion → dark brown lesion → brown necrotic lesion → circular necrotic lesion`. The last step silently dropped "brown" and swapped the whole descriptor — that is exactly the drift we forbid. The disciplined lattice for early blight, assuming the concept selector chose `{primary_sign(lesion), lesion_color, lesion_shape, necrosis}`:

```
lesions                                            (head, from primary_sign)
 ├─ADD_color(brown)──►      brown lesions          (color ∈ vocab_axes.color)
 │                     ├─ADD_shape(circular)─► circular brown lesions
 │                     └─ADD_shape(concentric)─► concentric brown lesions
 ├─ADD_shape(target-like)─► target-like lesions
 └─SUBST_syn(HEAD_lesion: lesions→spots)  DISABLED for diseases where 'spot' ∉ never_appear? 
        (early blight: 'spot' allowed → enabled → "brown spots")
Final valid surface forms (all assert the SAME selected concept set):
  "brown circular lesions", "circular brown lesions", "concentric brown lesions",
  "brown target-like spots", "brown circular necrotic lesions"
Invalid (would drift): "brown greasy lesions" (greasy ∈ never_appear),
  "brown lesions with pycnidia" (pycnidia ∈ never_appear, and pycnidia is Septoria's sign),
  "circular necrotic lesions" as a REPLACEMENT that drops an asserted color while claiming necrosis
     was already selected — allowed only if 'brown' modifier is retained OR color concept was not selected.
```
Every arrow adds a modifier from the DKB's own axes for early blight; nothing new is introduced; the asserted concept set is invariant.

### 7.5 Determinism
Given the caption seed, the expander makes a fixed number of seeded edge choices. The chosen edges (as `(edge_type, value, source_field)`) are stored in provenance so the expansion is replayable and auditable (doc 04 §3, doc 05 hallucination trace).

## 8. Build-time self-checks (Ontology Builder must fail loudly on)
- Any disease with empty `required_medical_terminology` (except healthy) → error (DKB entry too thin or derivation bug).
- Any `never_appear` term that also appears in that disease's `recommended_*` vocabulary → error (DKB contradiction; fix DKB).
- Any concept whose realizations are empty but the concept is in `optional_concepts` → drop the concept and warn.
- Any `vocab_axes` value not traceable to a DKB field → error (invariant #2 breach).
- Cross-disease leakage check: for each disease, assert `never_appear[d]` ⊇ union of *other* diseases' `required_medical_terminology` that the DKB marks as confusable (so a caption can't accidentally use a rival disease's hallmark term). Warn on gaps.
