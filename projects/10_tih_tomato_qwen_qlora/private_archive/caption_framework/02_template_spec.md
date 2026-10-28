# 02 — Template Specification (Task 2)

**Deliverable 3 of 8.** Defines the slot grammar, a library of **52 reusable templates** across 8 style families, and the template-selection algorithm. Templates are the **response side** of the instruction pair; instruction (user-turn) variety is specified in doc 04 §5. **These are templates, not captions** — the example line under each is an illustration of one instantiation, not dataset content.

Contract: a template may only require slots that map to concepts the disease actually has; the selector guarantees `required_slots(template) ⊆ selected_concepts` before a template is used (doc 00 §3, step 8).

---

## 1. Slot grammar

Slots are placeholders filled by the Slot Realizer from **already-selected, already-expanded** concepts (doc 01 §7). A slot in `[[ ]]` is **optional**: if its backing concept was not selected, the realizer deletes the slot and repairs surrounding punctuation/articles. A slot in `{{ }}` is **required** for that template.

| Slot | Filled from concept | Example surface |
|------|--------------------|-----------------|
| `{{HOST}}` / `[[HOST]]` | `host` | "tomato leaf", "mango leaf" |
| `{{DISEASE}}` | `disease_identity` | "early blight", "bacterial spot", "sooty mould" |
| `[[AGENT]]` | `agent_reference` | "*Alternaria solani*" (clinical/educational only) |
| `[[AGENT_CAT]]` | `agent_category_descriptor` | "a fungal disease", "insect feeding damage", "a mite infestation" |
| `{{PRIMARY}}` | `primary_sign` (expanded NP) | "brown concentric-ringed lesions" |
| `[[COLOR]]`,`[[SHAPE]]`,`[[SIZE]]`,`[[TEXTURE]]` | resp. concepts | modifiers already folded into PRIMARY by the expander, or standalone |
| `[[EXTENT]]` | `extent` | "scattered", "numerous" |
| `[[DIST]]` | `lesion_distribution` | "coalescing across the leaflet" |
| `[[LOCATION]]` | `leaf_location` | "along the leaf margins", "on the underside" |
| `[[CHLOROSIS]]` | `chlorosis` | "a surrounding yellow halo", "interveinal yellowing" |
| `[[NECROSIS]]` | `necrosis` | "necrotic centers" |
| `[[DEFORM]]` | `leaf_deformation` | "upward cupping", "fern-leaf distortion" |
| `[[SECONDARY]]` | `secondary_sign` (**hedged**) | "and may develop shot-holes" |
| `[[DIFFERENTIAL]]` | `differential` | "unlike Septoria leaf spot, no pycnidia are present" |
| `{{HEALTHY}}` | `healthy_state` | "uniform green coloration and an intact margin" |
| `[[SIGN_LIST]]` | ordered list of selected descriptive signs | "brown lesions, a yellow halo, and necrotic centers" |

**Realizer responsibilities:** article selection (a/an by phonetics), singular/plural agreement with `EXTENT`, Oxford-comma list assembly for `SIGN_LIST`, capitalization, terminal punctuation, and *slot deletion repair*. Every template is authored so that deleting any `[[optional]]` slot still yields a grammatical sentence.

## 2. Template record schema (`templates/templates.json`)
```json
{
  "id": "T-TS-03",
  "style": "two_sentence",
  "length_band": "medium",           // short|medium|long
  "target_tokens": [18, 34],
  "register": "descriptive",         // visual|clinical|descriptive|educational
  "hedged": false,
  "required_slots": ["DISEASE", "PRIMARY"],
  "optional_slots": ["HOST","EXTENT","LOCATION","CHLOROSIS","DIST","SECONDARY"],
  "min_concepts": 2,
  "max_concepts": 6,
  "sign_type_allow": ["lesion","coating","gall","stippling","cut","deformation","mottle","healthy"],
  "pattern": "This {{HOST}} shows {{DISEASE}}. [[EXTENT]] {{PRIMARY}} are visible[[LOCATION]][[, CHLOROSIS]].",
  "example_disease": "tomato_early_blight",
  "example_render": "This tomato leaf shows early blight. Scattered brown concentric-ringed lesions are visible on the lower leaves, with a surrounding yellow halo.",
  "use_when": "Default workhorse for medium-information descriptive captions; disabled for healthy (no PRIMARY)."
}
```
`sign_type_allow` lets a template exclude sign types it reads badly with (e.g., a "lesions coalesce" template is not offered for `mottle`/`healthy`).

## 3. The template library (52 templates)

Notation: `{{req}}`, `[[opt]]`. Examples abbreviated. Full renders are illustrative only.

### 3.1 Short — bare noun phrase / minimal (style `short`, band `short`, 4–10 tok) — 6 templates
Use: highest-frequency, cheap grounding; the majority style for low-concept classes (healthy, cutting_weevil). Registers: visual.
- **T-S-01** `{{PRIMARY}} on a {{HOST}}.` → "Brown concentric lesions on a tomato leaf."
- **T-S-02** `A {{HOST}} with {{DISEASE}}.` → "A tomato leaf with early blight."
- **T-S-03** `[[EXTENT]] {{PRIMARY}}[[ LOCATION]].` → "Numerous small dark spots along the margins."
- **T-S-04** `{{DISEASE}}: {{PRIMARY}}.` → "Bacterial spot: small dark angular spots with yellow halos."
- **T-S-05** `{{HOST}} showing {{PRIMARY}}.` → "Mango leaf showing raised wart-like galls."
- **T-S-06 (healthy-safe)** `A healthy {{HOST}} with {{HEALTHY}}.` → "A healthy mango leaf with uniform green coloration and a glossy surface."

### 3.2 Single sentence (style `single_sentence`, band short/medium, 10–22 tok) — 10 templates
Use: the second-most-common style; one clause, one or two signs. Registers: visual/descriptive.
- **T-SS-01** `This {{HOST}} is affected by {{DISEASE}}, showing {{PRIMARY}}[[ LOCATION]].`
- **T-SS-02** `The {{HOST}} exhibits {{PRIMARY}}[[, CHLOROSIS]], consistent with {{DISEASE}}.`
- **T-SS-03** `{{DISEASE}} is present, characterized by [[EXTENT]] {{PRIMARY}}.`
- **T-SS-04** `On this {{HOST}}, {{PRIMARY}} [[DIST]] indicate {{DISEASE}}.`
- **T-SS-05** `A {{HOST}} displaying {{SIGN_LIST}}, typical of {{DISEASE}}.`
- **T-SS-06** `Symptoms of {{DISEASE}} appear as {{PRIMARY}}[[ LOCATION]].`
- **T-SS-07** `This {{HOST}} shows signs of {{DISEASE}}: {{PRIMARY}}.`
- **T-SS-08 (deformation-safe)** `The {{HOST}} shows [[DEFORM]] with {{PRIMARY}}, consistent with {{DISEASE}}.` (for TYLCV/ToMV)
- **T-SS-09 (pest register)** `This {{HOST}} shows {{PRIMARY}} caused by {{DISEASE}}.` (cutting_weevil/gall_midge/spider_mites — "caused by" + damage noun, no "infection")
- **T-SS-10 (healthy)** `This {{HOST}} appears healthy, with {{HEALTHY}} and no visible symptoms.`

### 3.3 Two sentence (style `two_sentence`, band medium, 18–34 tok) — 8 templates
Use: the descriptive workhorse; sentence 1 states disease/leaf, sentence 2 elaborates signs. Registers: descriptive/visual.
- **T-TS-01** `This is a {{HOST}} affected by {{DISEASE}}. {{PRIMARY}} are visible[[ LOCATION]][[, with CHLOROSIS]].`
- **T-TS-02** `The {{HOST}} shows {{DISEASE}}. [[EXTENT]] {{PRIMARY}} [[DIST]].`
- **T-TS-03** `This {{HOST}} shows {{DISEASE}}. {{PRIMARY}} are present[[, and SECONDARY]].` (hedged if SECONDARY)
- **T-TS-04** `{{DISEASE}} affects this {{HOST}}. The most evident sign is {{PRIMARY}}[[ LOCATION]].`
- **T-TS-05** `A {{HOST}} with {{DISEASE}}. {{SIGN_LIST}} can be seen across the lamina.`
- **T-TS-06** `The leaf is a {{HOST}} showing {{DISEASE}}. {{PRIMARY}}[[, accompanied by CHLOROSIS]].`
- **T-TS-07 (surface-coating safe, sooty/powdery)** `This {{HOST}} shows {{DISEASE}}. A {{PRIMARY}} covers the surface[[; LOCATION]].`
- **T-TS-08 (healthy)** `This {{HOST}} is healthy. It has {{HEALTHY}}, with no lesions, discoloration, or deformation.`

### 3.4 Clinical (style `clinical`, band medium, terse, terminology-dense) — 6 templates
Use: teach precise diagnostic terminology; present-tense, telegraphic, may name `AGENT`/`AGENT_CAT` (pathogen classes) or category (pest classes). Register: clinical. Fraction: small.
- **T-CL-01** `Diagnosis: {{DISEASE}}[[ (AGENT)]]. Findings: {{SIGN_LIST}}[[; DIST]].`
- **T-CL-02** `{{HOST}}, {{DISEASE}}. {{PRIMARY}}[[; LOCATION]][[; CHLOROSIS]].`
- **T-CL-03** `Presentation consistent with {{DISEASE}}: {{PRIMARY}}[[, NECROSIS]].`
- **T-CL-04** `{{DISEASE}}[[ — AGENT_CAT]]. Leaf findings: {{SIGN_LIST}}.`
- **T-CL-05 (differential)** `{{DISEASE}}; {{PRIMARY}}. [[DIFFERENTIAL]].`
- **T-CL-06 (healthy)** `{{HOST}}, no pathology. {{HEALTHY}}; margins intact.`

### 3.5 Descriptive / natural (style `descriptive`, band medium/long, fluent) — 8 templates
Use: natural-language variety; the main body of the library along with two-sentence. Register: descriptive.
- **T-DS-01** `Looking at this {{HOST}}, {{DISEASE}} is evident from {{PRIMARY}}[[ that appear LOCATION]].`
- **T-DS-02** `The {{HOST}} in this image is affected by {{DISEASE}}; {{SIGN_LIST}} are clearly visible.`
- **T-DS-03** `Across this {{HOST}}, {{EXTENT}} {{PRIMARY}} have developed[[, DIST]], indicating {{DISEASE}}.`
- **T-DS-04** `This {{HOST}} carries {{DISEASE}}, seen as {{PRIMARY}}[[ with CHLOROSIS]].`
- **T-DS-05** `The foliage shown is a {{HOST}} with {{DISEASE}}, recognizable by {{PRIMARY}}.`
- **T-DS-06 (deformation)** `This {{HOST}} is distorted by {{DISEASE}}, showing [[DEFORM]][[ and CHLOROSIS]].`
- **T-DS-07 (pest)** `This {{HOST}} has been damaged by {{DISEASE}}, leaving {{PRIMARY}}[[ LOCATION]].`
- **T-DS-08 (healthy)** `This {{HOST}} looks healthy and vigorous, with {{HEALTHY}} and no signs of disease or pests.`

### 3.6 Educational (style `educational`, band long, adds reasoning/differential) — 6 templates
Use: teach discrimination between confusable classes; always includes `DIFFERENTIAL` or a "because" clause; hedged where it uses `SECONDARY`. Register: educational. Fraction: modest, concentrated on classes with rich `confused_with`.
- **T-ED-01** `This {{HOST}} shows {{DISEASE}}. The diagnostic sign is {{PRIMARY}}[[; DIFFERENTIAL]].`
- **T-ED-02** `{{DISEASE}} can be identified on this {{HOST}} by {{PRIMARY}}. {{DIFFERENTIAL}}.`
- **T-ED-03** `The {{PRIMARY}} on this {{HOST}} point to {{DISEASE}} rather than a look-alike, because {{DIFFERENTIAL}}.`
- **T-ED-04** `On this {{HOST}}, {{DISEASE}} presents as {{PRIMARY}}[[, and may later SECONDARY]].` (hedged)
- **T-ED-05** `Note the {{PRIMARY}} on this {{HOST}}: a hallmark of {{DISEASE}}[[. DIFFERENTIAL]].`
- **T-ED-06 (pest/organism)** `The damage on this {{HOST}} is caused by {{DISEASE}}[[ (AGENT_CAT)]], producing {{PRIMARY}}; this is feeding damage, not a leaf infection.` (pest classes only)

### 3.7 Dense caption (style `dense`, band long, many concepts) — 4 templates
Use: rich supervision; only when `|selected_concepts| ≥ 5`; never for healthy/cutting_weevil. Register: descriptive/clinical.
- **T-DN-01** `This {{HOST}} is affected by {{DISEASE}}, showing {{SIGN_LIST}}[[, distributed DIST]][[, with CHLOROSIS]][[; NECROSIS]].`
- **T-DN-02** `{{DISEASE}} on this {{HOST}}: {{PRIMARY}}[[ that are SIZE and SHAPE]][[, LOCATION]][[, accompanied by CHLOROSIS]][[ and NECROSIS]].`
- **T-DN-03** `The {{HOST}} exhibits multiple signs of {{DISEASE}} — {{SIGN_LIST}} — [[EXTENT and DIST]] across the lamina.`
- **T-DN-04** `A detailed view of {{DISEASE}} on a {{HOST}}: {{PRIMARY}}[[, TEXTURE in texture]][[, LOCATION]][[, with CHLOROSIS]].`

### 3.8 Long narrative (style `long`, band long, multi-sentence, comprehensive) — 4 templates
Use: longest supervision; 3 short clauses/sentences; combine identity + signs + (hedged) elaboration and optionally differential. Register: descriptive/educational.
- **T-LN-01** `This image shows a {{HOST}}. It is affected by {{DISEASE}}, evident from {{PRIMARY}}[[ LOCATION]]. [[EXTENT]] signs [[DIST]][[; CHLOROSIS]].`
- **T-LN-02** `The subject is a {{HOST}} with {{DISEASE}}. {{SIGN_LIST}} are visible. [[The condition may also SECONDARY.]]` (hedged)
- **T-LN-03** `Here is a {{HOST}} displaying {{DISEASE}}. Characteristic {{PRIMARY}} can be seen[[, with CHLOROSIS]]. {{DIFFERENTIAL}}.`
- **T-LN-04 (healthy)** `This image shows a {{HOST}}. The leaf appears healthy, with {{HEALTHY}}. No lesions, discoloration, coatings, galls, or deformation are present.`

## 4. Hedged register (implements doc 00 §5, doc 01 §5)
Any template that fills `[[SECONDARY]]` is authored in **hedged** form ("may develop", "can later show", "often"), never asserted. The realizer selects the hedged connective from a small controlled set (`vocabulary/hedges.json`: {"may develop","can develop","often shows","may later show"}). Templates T-TS-03, T-ED-04, T-LN-02 are the canonical hedged carriers; the realizer refuses to place `SECONDARY` in a non-hedged template.

## 5. Template selection algorithm (component D↔E)
```
INPUT: spec = (style, length_band, register, hedged?), selected_concepts, disease sign_types, seed
1. pool ← templates WHERE style==spec.style
                        AND register compatible
                        AND required_slots ⊆ concepts_as_slots(selected_concepts)
                        AND sign_type(selected primary) ∈ template.sign_type_allow
                        AND (spec.hedged OR template not requiring SECONDARY-as-required)
2. IF pool empty: relax length_band → then style (to nearest neighbor per §6 fallback order); log.
3. Apply anti-domination: remove templates whose running share for this disease ≥ max_template_share.
4. weight each remaining template inversely to its running usage (coverage of templates) and by template.priority.
5. choose ← seeded weighted sample(pool).
```
`concepts_as_slots` maps selected concept_ids to the slot names a template expects (e.g., `primary_sign`→`PRIMARY`, `chlorosis`→`CHLOROSIS`).

## 6. Global style / length / register target distribution (config `style_distribution`)
Default target mix across the whole library (per doc 00 §7.4; jittered per image). Chosen to bias toward short/medium for trainability while retaining rich examples.

| Style | Share | Length band | Register |
|-------|-------|-------------|----------|
| short | 18% | short | visual |
| single_sentence | 26% | short/medium | visual/descriptive |
| two_sentence | 22% | medium | descriptive |
| descriptive | 14% | medium/long | descriptive |
| clinical | 7% | medium | clinical |
| educational | 7% | long | educational |
| dense | 3% | long | descriptive/clinical |
| long | 3% | long | descriptive/educational |

**Per-class adjustments (automatic):**
- `healthy`, `cutting_weevil` (low concept count): renormalize to {short 40%, single 40%, two_sentence 15%, descriptive 5%}; disable dense/long/clinical-differential.
- Classes with rich `confused_with` (early_blight, target_spot, septoria, bacterial_spot, anthracnose, bacterial_canker, powdery_mildew, sooty_mould): educational share raised to 12% to teach discrimination on the hardest pairs.
- Fallback order when a style pool is empty: `dense→descriptive→two_sentence→single_sentence→short`; `long→descriptive→two_sentence`; `clinical→single_sentence`; `educational→two_sentence`.

## 7. Authoring rules for adding templates later (governance)
- Every new template must declare `required_slots`, `sign_type_allow`, `register`, and pass a **round-trip test**: render it for one lesion class, one deformation class, one coating class, one pest class, and the healthy class; deleting any optional slot must remain grammatical.
- No template may hard-code a disease name, a color, a symptom, or any domain term — all domain content enters only through slots (invariants #2/#3). Templates carry *syntax*, the DKB carries *content*.
- Template set changes bump `template_set_version` (doc 00 §6) and require a diversity re-measurement (doc 05).
