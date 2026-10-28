# 00 — Caption Generation Methodology (Master Overview)

**Deliverable 1 of 8.** Owns: system architecture, the generation algorithm, the information-budget model, the severity-honesty policy, the reproducibility model, and the dataset diversity strategy (Task 5). Downstream detail lives in documents 01–07; this document is the spine that references them.

Upstream contract: [`../knowledge_base/dkb.json`](../knowledge_base/dkb.json) is FINAL and canonical. This framework derives, never restates. The seven design invariants in [`README.md`](README.md) are the acceptance contract for everything below.

---

## 1. Position in the pipeline

```
Stage 1  Disease Knowledge Base (DONE)  ──►  dkb.json  (single source of truth)
Stage 2  Caption Generation Framework (THIS SPEC)  ──►  caption_library.jsonl
Stage 3  Instruction-Tuning Dataset  ──►  per-model training files
Stage 4  QLoRA fine-tuning (MLX)  ──►  adapters
Stage 5  Evaluation (zero-shot vs fine-tuned)
Stage 6  IEEE paper
```

Stage 2 consumes `dkb.json` + the folder labels of PlantVillage (tomato, 10 classes) and MangoLeafBD (mango, 8 classes) and emits a validated, provenance-tracked **caption library**. It does **not** touch images beyond reading their path and folder-derived label.

## 2. System architecture

Nine components. Each is specified in a named document; this overview defines their contracts and data flow.

```
                         dkb.json  (FINAL, read-only)
                              │
          ┌───────────────────┴───────────────────┐
          ▼ (build-time, deterministic derivation) │
   ┌──────────────┐   ┌───────────────┐   ┌────────▼────────┐
   │ (A) Ontology │   │ (B) Vocabulary│   │ (C) Symptom     │
   │  Builder     │   │  Builder      │   │  Lexicon Builder│
   │  →ontology/  │   │  →vocabulary/ │   │  →vocabulary/   │
   └──────┬───────┘   └───────┬───────┘   └────────┬────────┘
          │  (per-disease ontology, vocab axes, forbidden sets)
          └───────────────────┼────────────────────┘
                              ▼
   image label ─► ┌──────────────────────────┐   ┌──────────────┐
   (folder GT)    │ (D) Concept Selector      │◄──│ (E) Template │
                  │  - info budget            │   │  Library     │
                  │  - required/optional pick  │   │  templates/  │
                  └────────────┬──────────────┘   └──────┬───────┘
                              ▼                          │
                  ┌──────────────────────────┐           │
                  │ (F) Slot Realizer +       │◄──────────┘
                  │  Vocabulary Expander      │
                  └────────────┬──────────────┘
                              ▼
                  ┌──────────────────────────┐
                  │ (G) Validator Battery     │  ── fail ──► regenerate (new sub-seed)
                  │  (12 stages, doc 03)      │              or fallback template
                  └────────────┬──────────────┘
                              ▼ pass
                  ┌──────────────────────────┐
                  │ (H) De-duplicator +       │
                  │  Diversity Controller     │
                  └────────────┬──────────────┘
                              ▼
                  ┌──────────────────────────┐
                  │ (I) Emitter → canonical   │
                  │  caption record + prov.   │
                  └──────────────────────────┘
                              ▼
                     caption_library.jsonl  ──►  QA (doc 05)  ──►  converters (doc 04)
```

Components **A, B, C** run once at build time (they transform the DKB). Components **D–I** run per caption. Nothing in the runtime path calls a neural model.

## 3. The generation algorithm

Deterministic given a seed. Pseudocode (specification, not implementation):

```
INPUT:  image_id, crop, disease_label, global_seed, config
OUTPUT: 0..K validated caption records for this image

1.  disease_id  ← label_map[crop][disease_label]          # e.g. "Tomato___Early_blight" → "tomato_early_blight"
2.  onto        ← ontology[disease_id]                    # derived, doc 01
3.  base_seed   ← H(global_seed ‖ image_id)               # stable per-image seed (H = SHA-256 → int)
4.  budget_plan ← plan_captions_for_image(disease_id, config, base_seed)
        # decides how many captions and their (style,length,register) mix for this image
5.  FOR n, spec IN enumerate(budget_plan):
6.      seed_n  ← H(base_seed ‖ n)
7.      concepts ← ConceptSelector.select(onto, spec, seed_n)      # required ∪ sampled optional, within info budget
8.      template ← TemplateLibrary.choose(spec, concepts, seed_n)  # required_slots(template) ⊆ concepts
9.      draft    ← SlotRealizer.realize(template, concepts, onto, seed_n)   # fills slots
10.     caption  ← VocabExpander.expand(draft, onto, seed_n)       # synonym/adjective variation, no drift
11.     verdict  ← Validator.run(caption, onto, concepts)          # doc 03
12.     IF verdict.fail AND attempts < MAX_ATTEMPTS: goto 6 with seed_n ← H(seed_n ‖ attempt)
13.     IF verdict.fail AND attempts == MAX_ATTEMPTS: caption ← Fallback.minimal(onto, seed_n); re-validate
14.     IF Deduplicator.is_duplicate(caption): goto 6 (bounded); else register
15.     record   ← Emitter.build(image_id, disease_id, caption, template, concepts, seeds)
16.     yield record
```

Key properties:
- **Determinism**: identical `(global_seed, config, image set, DKB)` ⇒ identical library, byte-for-byte. Enables exact reproduction in the paper and diffing across DKB revisions.
- **Bounded regeneration**: `MAX_ATTEMPTS` (default 8) then a guaranteed-valid **fallback** template (minimal safe caption using only required concepts). The pipeline never emits an invalid caption and never hangs.
- **No hidden state**: everything that influenced a caption is captured in provenance (doc 04 §3).

## 4. Concept selection & the information budget

A caption is a set of **concepts** (doc 01 §2) realized as text. Concept selection is governed by an **information budget** so captions range from terse to dense without ever exceeding the allowed concept set.

For disease `d`, let:
- `R_d` = required concepts (always included),
- `O_d` = optional concepts (sampled),
- `F_d` = forbidden concepts (never; enforced by validator as defense-in-depth).

Each caption request carries a target **information level** `L ∈ {minimal, low, medium, high, dense}`, which maps to a target concept count `c(L)`:

| Level | Target concepts | Typical use |
|-------|-----------------|-------------|
| minimal | \|R_d\| (required only) | short captions, healthy/pest classes with few signs |
| low | \|R_d\| + 1 | single-sentence captions |
| medium | \|R_d\| + 2–3 | two-sentence / descriptive |
| high | \|R_d\| + 4–5 | long / clinical |
| dense | all compatible O_d (capped) | dense captions (only when `\|R_d ∪ O_d\| ≥ 6`) |

Selection rule (component D):
1. Start with `R_d`.
2. Sample `k = clip(c(L) − |R_d|, 0, |O_d|)` optional concepts from `O_d` **without replacement**, weighted by `concept.salience` (doc 01 §2.3; primary/diagnostic concepts weighted higher than secondary).
3. Enforce **co-selection constraints** (doc 01 §2.4): e.g., `lesion_shape` may only be selected if `primary_sign` is a lesion-type sign; `chlorosis` and `necrosis` may co-occur; mutually exclusive concepts (e.g., `white_powdery_coating` vs `black_sooty_coating`) can never co-occur (they belong to different diseases anyway, but the guard is global).
4. If a class has very few concepts (e.g., `healthy`, `cutting_weevil`), levels `high`/`dense` are **clamped** to the max available; the budget planner (below) avoids requesting them.

The point: information level controls *length and richness*; it never unlocks concepts outside `R_d ∪ O_d`.

## 5. Severity-honesty policy (critical)

Severity is the sharpest scientific-integrity risk in this stage, and it is handled explicitly.

**Problem.** The DKB defines `severity.{mild,moderate,severe}`, but the datasets label only *disease identity*, not *severity*. Asserting "severe early blight" for an arbitrary early-blight image is an unsupported per-image claim — the same class of error as claiming a non-observable symptom.

**Policy.**
1. **Stage labels are gated.** The tokens *mild / moderate / severe / early-stage / advanced* and any `severity_vocabulary` entry that denotes a disease **stage** are **forbidden by default**. They may appear only in a **severity-conditioned mode** that requires a per-image `severity_label` supplied by a separate (future, optional) annotation task. Absent that label, the severity-claim guard (doc 03, V9) rejects them.
2. **Extent vs stage are separated.** `severity_vocabulary` is split at ontology-build time into:
   - **extent descriptors** — describe *visible density/coverage* not a clinical stage (*a few, scattered, numerous, coalescing, extensive coverage*). These are **allowed** because they describe what is in the frame, not a diagnosis of stage. They are still optional concepts, never required.
   - **stage descriptors** — denote clinical progression (*mild, moderate, severe, early, advanced*). **Gated** as in (1).
   The split is specified per disease in doc 01 §5; the builder classifies each `severity_vocabulary` term into one bucket using the global lexicon in `vocabulary/severity_axis.json`.
3. **Default register is "disease-typical".** Because PlantVillage/MangoLeafBD images are *curated class exemplars*, captions describe the **characteristic leaf-observable presentation of the labeled disease**, prioritizing `primary_symptoms`, `diagnostic_visual_features`, and `key_differentiating_features` (features reliably present in class exemplars). Rare/stage-specific `secondary_symptoms` are optional and, when used, are **hedged** ("often", "may show", "can develop") rather than asserted as certainly present in the specific frame. The hedging register is a template attribute (doc 02 §4).

This policy is a deliberate research contribution (doc 07 §E): the framework distinguishes *label-supported claims* (disease identity + its characteristic signs) from *image-specific claims it cannot license* (exact severity, exact count) and structurally prevents the latter.

## 6. Reproducibility model

- **Seeds.** One `global_seed` in `generation/config.yaml`. Per-image seed `= SHA256(global_seed ‖ image_id)`; per-caption seed `= SHA256(base_seed ‖ index)`; per-attempt seed folds in the attempt number. All RNG draws use these; no wall-clock, no unseeded PRNG.
- **Provenance.** Every emitted record stores: `global_seed`, `base_seed`, `caption_seed`, `template_id`, ordered list of selected `concept_ids`, every vocabulary choice (`slot → chosen_surface_form → source_field`), validator version, DKB version hash. Schema in doc 04 §3.
- **Pinning.** The library header records `dkb_sha256`, `ontology_build_id`, `template_set_version`, `vocabulary_version`, `config_hash`. Any change to these is a new library version; libraries are immutable once QA-accepted.
- **Regeneration test.** CI requirement: regenerating any record from its provenance must reproduce the identical caption string; a mismatch is a build failure.

## 7. Dataset diversity strategy (Task 5)

Goal: **maximize linguistic diversity subject to scientific correctness**. Diversity is engineered at five loci and measured against explicit gates.

### 7.1 Five loci of controlled variation (all seeded)
1. **Template variation** — random template choice from the compatible set (doc 02), with anti-domination caps (§7.3).
2. **Concept-subset variation** — which optional concepts are chosen (component D), via coverage-guided sampling (§7.2), so different captions of the same image foreground different valid signs.
3. **Lexical variation** — controlled synonym/adjective substitution within disease-filtered equivalence classes (doc 01 §7). No semantic drift.
4. **Syntactic variation** — template-level alternations: active/passive, NP-fronting, clause order, "there-is" vs attributive, listing order of multi-sign enumerations (seeded shuffle of enumerable slots).
5. **Register/length variation** — the per-image style/length mix from the budget planner (§7.4).

### 7.2 Coverage-guided concept sampling (not pure random)
Pure random sampling over-represents high-salience concepts and under-covers rare valid combinations. Instead, the selector maintains a per-disease **coverage table** of `(concept, concept-pair)` usage counts and biases sampling toward under-covered concepts/pairs (ε-greedy: with prob ε sample to maximize coverage, else salience-weighted). Target: every valid concept appears in ≥ `min_concept_coverage` fraction of that disease's captions, and every valid concept *pair* appears at least once (subject to co-selection constraints).

### 7.3 Anti-domination and anti-overfitting caps
- **Template cap**: no single `template_id` may realize more than `max_template_share` (default 8%) of a disease's captions.
- **Skeleton cap**: after masking slot fillers, no caption *skeleton* (template + syntactic-variant) may exceed `max_skeleton_share` (default 12%).
- **Exact-instantiation cap**: any exact `(template, concept-set, vocab-choices)` tuple may be emitted at most once per disease (i.e., no two captions of the same disease are identical strings; enforced by dedup, §7.5).
- **Opening-n-gram balance**: monitor the distribution of caption-initial trigrams; if any exceeds `max_opening_trigram_share` (default 15%), down-weight templates that produce it. Prevents "The leaf shows…" monotony.

### 7.4 Per-image budget planner
`plan_captions_for_image` decides, per image, how many captions (`K`) and their `(style, length, register)` mix, drawn from a **global target distribution** (doc 02 §5) but jittered per-image (seeded) so images are not stylistically identical. `K` is set by class-balancing (§7.6). Classes with few concepts (healthy, cutting_weevil) get a distribution skewed to short/single styles (their dense templates are unavailable).

### 7.5 Duplicate prevention (two-level)
- **Exact**: normalized (lowercased, whitespace/punct-collapsed) string hash set; collisions rejected.
- **Near-duplicate**: token-shingle MinHash (k=5 shingles) with Jaccard threshold `dedup_jaccard` (default 0.9); a candidate too similar to an accepted caption *of the same image* is rejected and regenerated. Across different images of the same disease, near-duplicates are allowed but counted and capped by the skeleton/template shares (§7.3).

### 7.6 Class balancing
PlantVillage classes are imbalanced (≈14k images across 10 classes, uneven); MangoLeafBD is balanced (500/class). Two supported policies in `config.yaml` (`balance_mode`):
- `per_image_fixed` (default): `K` captions per image (default `K=3`), so caption counts track image counts. Simpler; downstream sampler balances at training time.
- `per_class_target`: choose `K_image` so each class reaches a target caption count `T_class`, capping/oversampling captions (never oversampling *images*). Use when a balanced caption library is wanted directly.
The recommendation is `per_image_fixed` with training-time class weighting (doc 04 §6), to avoid manufacturing artificial redundancy.

### 7.7 Diversity acceptance gates (measured; doc 05 uses these)
Computed per disease and globally; a library that fails a hard gate is rejected.

| Metric | Definition | Target |
|--------|-----------|--------|
| distinct-1 / distinct-2 / distinct-3 | unique n-grams / total n-grams | ≥ 0.10 / 0.45 / 0.70 (global) |
| self-BLEU (↓) | mean BLEU of each caption vs sample of others (same disease) | ≤ 0.35 |
| template entropy | Shannon entropy over `template_id` usage (per disease) | ≥ 0.85 × log₂(#compatible templates) |
| concept coverage | fraction of valid concepts appearing ≥ once (per disease) | = 1.0 |
| concept-pair coverage | fraction of valid concept pairs appearing ≥ once | ≥ 0.9 |
| max template share | largest single-template fraction (per disease) | ≤ 0.08 |
| exact-dup rate | identical caption strings | 0 within disease |
| TTR (type-token ratio) | unique tokens / tokens (per disease, sampled) | ≥ baseline in config |

## 8. Failure handling
- Validation fail → regenerate (new sub-seed) up to `MAX_ATTEMPTS` → guaranteed-valid minimal fallback (required concepts only, shortest safe template) → if even fallback fails, **hard error** surfaced with the disease_id and offending concept (indicates an ontology/derivation bug, not a data issue).
- Dedup exhaustion (cannot find a novel caption for an image at level L within bounded tries) → lower the level L by one and retry; if minimal is exhausted, emit fewer than `K` captions for that image and log it (expected only for `healthy`/`cutting_weevil` at high `K`).

## 9. Configuration surface (single file: `generation/config.yaml`)
All tunables live in one config so the pipeline has no magic numbers:
`global_seed, captions_per_image (K), balance_mode, T_class, style_distribution, length_distribution, register_distribution, MAX_ATTEMPTS, dedup_jaccard, max_template_share, max_skeleton_share, max_opening_trigram_share, min_concept_coverage, epsilon (coverage sampler), hedging_probability, severity_conditioned (bool, default false), diversity_gates{...}`. Defaults are given inline in doc 06 §4.

## 10. Cross-references
- Concepts, DKB→ontology derivation, vocabulary expansion → doc 01.
- Templates & selection → doc 02.
- Validator battery → doc 03.
- Canonical record, instructions, per-model conversion → doc 04.
- QA & acceptance → doc 05.
- Runtime folders & configs → doc 06.
- Paper methodology & justification → doc 07.
