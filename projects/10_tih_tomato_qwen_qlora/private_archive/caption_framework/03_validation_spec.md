# 03 — Validation Specification (Task 4)

**Deliverable 4 of 8.** Defines the **12-stage validator battery** (component G), its supporting lexicons, the regeneration/fallback loop, and the machine-readable report format. Every candidate caption must pass all **blocking** validators before it can enter the library. Validation is deterministic and produces a per-caption trace stored in provenance (doc 04 §3) so any acceptance/rejection is auditable.

Design stance: **defense in depth.** Multiple validators can catch the same defect from different angles; a hallucination that slips one check is caught by another. QA (doc 05) measures residual leakage assuming validators are imperfect.

---

## 1. Supporting lexicons (built once by component C, in `vocabulary/`)

| File | Content | Built from |
|------|---------|-----------|
| `symptom_lexicon.json` | Map: surface phrase / lemma → `concept_id` + `sign_type` + `owner_diseases[]`. Includes **forbidden-symptom surfaces** (fruit, twig, gummosis, tear-stain, pycnidia, star-shaped, etc.). | DKB `primary/secondary_symptoms`, `diagnostic_visual_features`, `forbidden_symptoms_not_leaf_observable`, `confused_with` targets' hallmark terms. |
| `allowed_terms/<disease_id>.txt` | The closed domain-term whitelist for the disease (all `recommended_*` vocab + `vocab_axes` values + host/disease names). | Ontology `required_medical_terminology` ∪ `optional_descriptive_terminology` ∪ `vocab_axes.*`. |
| `never_appear/<disease_id>.txt` | The disease's `never_appear` set (doc 01 §3.2). | Ontology. |
| `stage_terms.txt` | Global list of severity-**stage** tokens. | `vocabulary/severity_axis.json:stage`. |
| `hedges.json`, `synonyms.json`, `modifiers.json` | As in doc 01/02. | — |
| `function_words.txt` | Stop/function words excluded from the closed-vocab check (articles, prepositions, connectives, copulas). | Static English list + doc 02 connectives. |

Matching is **lemmatized, word-boundary, case-insensitive**, with multiword-phrase matching (longest-match first) so "fruit" in "dragonfruit-shaped" is not a false hit and "tear-stain" matches as a unit.

## 2. The 12 validators

Each entry: **purpose · input · method · pass criterion · blocking? · action on fail**. Execution order is V1→V12; a blocking failure short-circuits to the action (regenerate) but the full trace still records which validators had run.

### V1 — Ontology conformance
- **Purpose:** caption asserts only allowed concepts, and enough of them.
- **Input:** provenance `asserted_concepts` (the concept set the generator claims), disease ontology.
- **Method:** check `asserted_concepts ⊆ (required ∪ optional)`, `required ⊆ asserted_concepts`, and `min_information ≤ |asserted| ≤ max_information`.
- **Pass:** all three hold. **Blocking:** yes. **On fail:** regenerate (generator bug or over-selection).

### V2 — Forbidden-symptom detection (observability)
- **Purpose:** no non-leaf-observable / wrong-disease symptom is stated.
- **Input:** caption text, `symptom_lexicon.json`, disease `forbidden_symptoms_not_leaf_observable` + `never_appear`.
- **Method:** scan text; for every matched symptom surface, resolve to `concept_id`/owner; **fail** if the matched concept is in the disease's forbidden set, or is a symptom owned only by *other* diseases, or denotes a non-leaf structure (fruit/twig/flower/root/tree/vascular/insect-adult).
- **Pass:** zero forbidden-symptom matches. **Blocking:** yes. **On fail:** regenerate; log the offending span (this is the primary hallucination trap).

### V3 — Forbidden-vocabulary detection
- **Purpose:** no forbidden term/adjective appears.
- **Input:** caption, `never_appear/<disease_id>.txt`.
- **Method:** lemmatized boundary match against the never-appear set.
- **Pass:** zero matches. **Blocking:** yes. **On fail:** regenerate.

### V4 — Closed-vocabulary / terminology whitelist
- **Purpose:** every *domain* content word is licensed by the DKB (no invented terminology).
- **Input:** caption, `allowed_terms/<disease_id>.txt`, `function_words.txt`.
- **Method:** tokenize; remove function words, template scaffolding tokens (from the chosen template's fixed lexemes, e.g., "shows", "visible", "image", "leaf"), and numerals. Every remaining **content token/phrase** must be in the allowed-terms whitelist (or be a permitted synonym-class member resolvable to an allowed term). Any residual out-of-vocabulary content token → fail.
- **Pass:** OOV content set empty. **Blocking:** yes. **On fail:** regenerate. **Note:** template scaffolding vocabulary is itself a fixed, reviewed closed set (`templates/scaffold_lexicon.json`) so V4 does not reject ordinary sentence glue.

### V5 — Required-content presence
- **Purpose:** the caption actually states the disease identity and at least one primary sign (or healthy_state).
- **Input:** caption, ontology `required_concepts`, realization map.
- **Method:** confirm the surface realizations of each required concept are present in text (not just claimed in provenance) via the realization phrases / their synonym expansions.
- **Pass:** all required concepts textually present. **Blocking:** yes. **On fail:** regenerate.

### V6 — No-drift / realization integrity
- **Purpose:** expansion did not introduce or drop asserted concepts; every modifier traces to the DKB.
- **Input:** provenance expansion edges (doc 01 §7.5), pre/post-expansion concept sets.
- **Method:** assert `concepts(post) == concepts(pre)`; assert every `ADD_*` edge's value ∈ the disease `vocab_axes`; assert every `SUBST_syn` stayed within one synonym class; assert modifier depth ≤ `max_adjectives`.
- **Pass:** all hold. **Blocking:** yes. **On fail:** regenerate.

### V7 — Register & pest/pathogen consistency
- **Purpose:** no non-observable concept in a visual caption; correct pest vs pathogen language.
- **Input:** caption, template register, disease `is_pathogen_disease`, `agent_category`.
- **Method:** (a) if template register is `visual`, fail on any `non_observable` concept realized (agent_reference, agent_category_descriptor, severity_stage, management); (b) if `is_pathogen_disease==false`: fail on `{infection, infected, pathogen}` and on `lesion`/other terms present in that disease's `never_appear`; require organism references to use `agent_category` framing; (c) if `is_pathogen_disease==true`: fail if the caption calls it "pest damage"/"feeding damage"/"cut/gall/stippling" (wrong mechanism).
- **Pass:** register-consistent. **Blocking:** yes. **On fail:** regenerate.

### V8 — Cross-disease leakage
- **Purpose:** the caption does not use another disease's hallmark diagnostic term.
- **Input:** caption, global map `disease_id → required_medical_terminology`, this disease's `confused_with`.
- **Method:** fail if any *other* disease's hallmark term appears **unless** it occurs inside a legitimate `differential` span (e.g., "unlike Septoria, no pycnidia") — differential spans are whitelisted by matching the template's `DIFFERENTIAL` slot region.
- **Pass:** no unlicensed rival hallmark term. **Blocking:** yes. **On fail:** regenerate.

### V9 — Severity-claim guard
- **Purpose:** no per-image severity *stage* claim unless licensed.
- **Input:** caption, `stage_terms.txt`, `config.severity_conditioned`, optional `severity_label`.
- **Method:** if `severity_conditioned==false` (default), fail on any `stage` token. If `true`, allow only the stage token equal to the supplied `severity_label` and forbid others.
- **Pass:** no illicit stage token. **Blocking:** yes. **On fail:** regenerate. (Extent descriptors from doc 01 §5 are *not* stage terms and pass.)

### V10 — Consistency / contradiction
- **Purpose:** no internally contradictory statement.
- **Input:** caption, realized concept set, `mutex_groups`.
- **Method:** fail if two concepts from the same `mutex_group` co-occur (white-powdery + black-sooty; raised-gall + flat-lesion; healthy_state + any disease sign; "no lesions/intact margin" + a lesion/cut); fail on numeric/plurality contradictions ("a single lesion … are scattered"); for `sooty_mould`, fail if `necrosis`/`chlorosis`/tissue-death asserted (DKB: tissue healthy beneath); for `healthy`, fail on any symptom token.
- **Pass:** no contradiction. **Blocking:** yes. **On fail:** regenerate.

### V11 — Grammar & fluency
- **Purpose:** grammatical, natural English.
- **Input:** caption text.
- **Method:** (a) a grammar checker (LanguageTool or equivalent, pinned version) with a project rule profile — zero errors of categories {agreement, article, verb-form, run-on, capitalization, double-punctuation}; (b) article a/an phonetic check; (c) subject–verb and singular/plural agreement with `EXTENT`; (d) adjective-order check against the canonical order (doc 01 §7.2); (e) sentence-count matches the template's declared style (e.g., `two_sentence` ⇒ exactly 2 sentences); (f) terminal punctuation present.
- **Pass:** zero blocking grammar errors. **Blocking:** yes. **On fail:** regenerate. (Because templates are pre-vetted, V11 failures are rare and usually indicate a slot-deletion repair bug — surfaced with high priority.)

### V12 — Duplication
- **Purpose:** no duplicate/near-duplicate captions.
- **Input:** caption, per-image emitted set, per-disease MinHash index (doc 00 §7.5).
- **Method:** (a) **intra-caption**: no two sentences identical after normalization; (b) **exact**: normalized string not already emitted (globally within disease); (c) **near-dup within same image**: MinHash Jaccard < `dedup_jaccard` vs any accepted caption of the *same image*.
- **Pass:** all hold. **Blocking:** yes (a,b); (c) blocking within-image only. **On fail:** regenerate.

## 3. Non-blocking (soft) checks — recorded, not rejected
These raise warnings used by diversity QA (doc 05), not per-caption rejection:
- **S1 length band**: realized token count within the template's `target_tokens` (warn if outside).
- **S2 opening-trigram** frequency contribution (feeds anti-domination, doc 00 §7.3).
- **S3 readability** (e.g., Flesch band) for the descriptive/long styles.
- **S4 hedge presence** when `SECONDARY` realized (should already be enforced by realizer; warn if missing).

## 4. Regeneration & fallback loop (component G control)
```
attempts ← 0
loop:
    run V1..V12
    if all blocking pass: accept, break
    attempts += 1
    if attempts < MAX_ATTEMPTS:            # default 8
        reseed (fold attempt into caption seed); regenerate from concept-selection
        continue
    else:
        caption ← Fallback.minimal(ontology, seed)   # required concepts, shortest safe template (T-S-02 / T-SS-10 / T-S-06)
        run V1..V12 on fallback
        if pass: accept (mark provenance.fallback=true)
        else: HARD_ERROR(disease_id, failing_validator, offending_span)   # indicates ontology/lexicon bug
```
- The **fallback** uses only `required_concepts` and a template with the smallest slot footprint, so it is guaranteed to pass unless the ontology itself is inconsistent — in which case a hard error is the correct outcome (fix the DKB/derivation, not the caption).
- `MAX_ATTEMPTS`, `dedup_jaccard`, grammar profile version are in `validators/rules.yaml`.

## 5. Report format (`validators/reports/<library_version>/`)
Per caption (embedded in provenance) and aggregated:
```json
{
  "caption_id": "...", "disease_id": "tomato_early_blight",
  "verdict": "accept",              // accept | fallback | hard_error
  "attempts": 2,
  "validators": {
    "V1":{"pass":true}, "V2":{"pass":true},
    "V4":{"pass":false,"attempt":1,"oov":["reticulate"]},   // example rejected attempt
    "...":"..."
  },
  "soft":{"S1":{"in_band":true},"S2":{"opening_trigram":"this tomato leaf"}}
}
```
Aggregated report (per disease, per library): rejection counts by validator, mean attempts, fallback rate, hard-error list. **Acceptance gate:** fallback rate ≤ 2% per disease and **zero** hard errors, else the run is blocked and the ontology/lexicons are revisited before QA.

## 6. What the validators deliberately do NOT do
- They do not look at the image (invariant #1) — validation is against the DKB-derived ontology only.
- They do not "fix" a caption in place; they reject and regenerate, keeping generation and validation cleanly separated and every accepted caption fully reproducible.
- They do not judge *truth about the specific pixels* (impossible without image analysis); they judge *consistency with the disease label's licensed description*. The QA human review (doc 05) is where residual over-claim risk is sampled.
