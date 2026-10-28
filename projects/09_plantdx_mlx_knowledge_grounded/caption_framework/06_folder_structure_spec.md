# 06 — Folder Structure Specification (Task 7)

**Deliverable 7 of 8.** The exact runtime directory layout the implementation must create under `experiments/`, the purpose of every file, the data-flow ordering, naming conventions, and default config values. An engineer should be able to create empty stubs from this tree and fill them in dependency order.

---

## 1. Full tree

```
experiments/
├── tomato/                              # EXISTING — images (do not modify)
│   └── PlantVillage/<Class_Folder>/*.JPG
├── mango/                               # EXISTING — images (do not modify)
│   └── MangoLeafBD/<Class_Folder>/*.jpg
│
├── knowledge_base/                      # EXISTING — Stage 1, FINAL, single source of truth
│   ├── dkb.json                         #   canonical DKB (read-only input to Stage 2)
│   └── DKB_report.md                    #   human-readable DKB (used by QA review UI)
│
├── caption_framework/                   # THIS SPEC (design docs 00–07 + README). Read-only reference.
│
├── ontology/                            # (A) Ontology Builder outputs — DERIVED from dkb.json
│   ├── concept_schema.json              #   global concept registry (doc 01 §2)
│   ├── global_policy.yaml               #   defaults: register rules, severity split, hedges, salience
│   ├── caption_ontology.json            #   per-disease ontology records ×18 (doc 01 §3) — GENERATED
│   ├── overrides/<disease_id>.yaml      #   optional manual overrides (empty by default; doc 01 §6)
│   └── ontology_build_report.md         #   audit: DKB-field→concept trace, warnings (doc 01 §1)
│
├── vocabulary/                          # (B)(C) Vocabulary + Lexicon Builder outputs — DERIVED
│   ├── synonyms.json                    #   equivalence classes (doc 01 §7.1)
│   ├── modifiers.json                   #   color/shape/size/texture/extent/location axes (doc 01 §7.2)
│   ├── adjective_order.json             #   canonical adjective ordering
│   ├── location_axis.json               #   normalization table for location phrases
│   ├── severity_axis.json               #   {extent[], stage[]} classification (doc 00 §5)
│   ├── sign_type_map.json               #   keyword→sign_type map (doc 01 §3.3)
│   ├── hedges.json                      #   hedged connectives (doc 02 §4)
│   ├── symptom_lexicon.json             #   surface→concept, incl. forbidden symptoms (doc 03 §1)
│   ├── function_words.txt               #   stop/function words excluded from closed-vocab check
│   ├── allowed_terms/<disease_id>.txt   #   closed whitelist per disease (doc 03 §1) — GENERATED
│   ├── never_appear/<disease_id>.txt    #   forbidden set per disease — GENERATED
│   └── stage_terms.txt                  #   global severity-stage tokens
│
├── templates/                           # (E) Template + instruction libraries — AUTHORED (syntax only, no domain content)
│   ├── templates.json                   #   the 52 caption templates (doc 02 §3)
│   ├── template_index.json              #   index by style/length/register/required-slots (doc 02 §5)
│   ├── instructions.json                #   instruction bank, ≥6 paraphrases × task_type (doc 04 §4)
│   └── scaffold_lexicon.json            #   fixed template glue words allowed by V4 (doc 03 V4)
│
├── generation/                          # (D,F) Generation configuration + provenance
│   ├── config.yaml                      #   the single config surface (doc 00 §9; defaults §4 below)
│   └── provenance/<library_version>/    #   full per-caption provenance (regeneration + audit)
│
├── validators/                          # (G) Validation configuration + reports
│   ├── rules.yaml                       #   thresholds, grammar profile version, MAX_ATTEMPTS
│   └── reports/<library_version>/       #   per-run validation aggregates (doc 03 §5)
│
├── captions/                            # (H,I) Generated caption library
│   ├── raw/<library_version>.jsonl      #   pre-dedup candidates (debug; optional retention)
│   └── caption_library.jsonl            #   canonical records, ACCEPTED (doc 04 §1) — the deliverable of Stage 2
│
├── datasets/                            # Stage 3 — converters (doc 04 §6)
│   ├── splits/
│   │   ├── train_image_ids.txt
│   │   ├── val_image_ids.txt
│   │   ├── test_image_ids.txt
│   │   └── diagnostic_image_ids.txt     #   confusable-pair held-out set (doc 04 §5)
│   ├── qwen2_5_vl/{train,val}.jsonl + manifest.json
│   ├── qwen3_vl/{train,val}.jsonl + manifest.json
│   ├── internvl3/{train,val}.jsonl + manifest.json
│   ├── gemma3/{train,val}.jsonl + manifest.json
│   └── mlx_vlm/{train,val}.jsonl + manifest.json + README.md   # mlx-vlm version pin (doc 04 §6.4)
│
├── metadata/                            # Dataset-level bookkeeping
│   ├── label_map.json                   #   folder→disease_id (doc 04 §2) — AUTHORED, immutable
│   ├── dataset_card.md                  #   dataset card (provenance, license, stats, intended use)
│   ├── manifest.json                    #   library_version, dkb_sha256, ontology_build_id, versions, counts
│   └── stats/<library_version>/         #   diversity metrics, concept coverage, length dists (doc 00 §7.7)
│
└── qa/                                  # Stage-2 QA (doc 05)
    ├── checklist.md                     #   the reviewer checklist (doc 05 §3), print/reference form
    ├── review_samples/<library_version>.jsonl   #   seeded audit manifest
    ├── review_results/<reviewer_id>.jsonl       #   per-reviewer verdicts (independent)
    └── acceptance_<library_version>.md          #   final sign-off (counts, κ, gates)
```

## 2. Purpose of each top-level directory (one line)
- `ontology/` — the DKB **projected** into caption concepts; what may be said per disease.
- `vocabulary/` — the closed word inventories, synonym classes, modifier axes, and forbidden/lexicon files the realizer, expander, and validators consume.
- `templates/` — reusable **syntax** (caption + instruction), carrying zero domain content.
- `generation/` — the one config file + full provenance for reproducibility.
- `validators/` — validation thresholds and run reports.
- `captions/` — the accepted canonical caption library (the Stage-2 product).
- `datasets/` — per-model training files derived by pure adapters + the fixed splits.
- `metadata/` — label map, dataset card, version manifest, and computed stats.
- `qa/` — the human-review artifacts and acceptance record.

## 3. Data-flow / build order (dependency DAG)
```
dkb.json ─► (build) ontology/  +  vocabulary/          [components A,B,C — run once per DKB version]
templates/  (authored, DKB-independent)                [reviewed once; versioned]
generation/config.yaml (authored)
      │
      ▼
[generate]  captions/raw → validators/ (doc 03) → captions/caption_library.jsonl   [components D–I]
      │
      ▼
[measure]   metadata/stats/  (diversity gates, doc 00 §7.7)
      │
      ▼
[QA]        qa/  (doc 05)  ──accept──►  freeze library_version
      │
      ▼
[split]     datasets/splits/  (by image, stratified)
      │
      ▼
[convert]   datasets/{qwen2_5_vl,qwen3_vl,internvl3,gemma3,mlx_vlm}/   [pure adapters, doc 04 §6]
```
Rule: nothing downstream is produced until its upstream is version-frozen. `caption_library.jsonl` is immutable once QA-accepted; any change forces a new `library_version`.

## 4. `generation/config.yaml` — defaults (authoritative)
```yaml
global_seed: 20260711
captions_per_image: 3            # K (doc 00 §7.6)
balance_mode: per_image_fixed    # per_image_fixed | per_class_target
T_class: null                    # used only if per_class_target
MAX_ATTEMPTS: 8
max_adjectives: 3
hedging_probability: 0.9         # prob. that a SECONDARY sign is hedged (must stay high)
severity_conditioned: false      # NEVER true unless per-image severity labels exist
dedup_jaccard: 0.90
epsilon_coverage: 0.30           # ε-greedy coverage sampler (doc 00 §7.2)
style_distribution:  { short: 0.18, single_sentence: 0.26, two_sentence: 0.22,
                       descriptive: 0.14, clinical: 0.07, educational: 0.07,
                       dense: 0.03, long: 0.03 }
task_distribution:   { describe: 0.34, identify: 0.14, signs: 0.16, color_qa: 0.08,
                       location_qa: 0.08, crop_qa: 0.05, differential: 0.10, healthy_check: 0.05 }
anti_domination:     { max_template_share: 0.08, max_skeleton_share: 0.12, max_opening_trigram_share: 0.15 }
diversity_gates:     { distinct_1: 0.10, distinct_2: 0.45, distinct_3: 0.70,
                       self_bleu_max: 0.35, concept_coverage: 1.0, concept_pair_coverage: 0.90 }
splits:              { train: 0.80, val: 0.10, test: 0.10, group_by: image, stratify_by: disease_id }
```
Per-class automatic overrides (healthy/cutting_weevil skew; educational boost on confusable classes) are applied by the budget planner per doc 02 §6 and need no manual config.

## 5. Naming & versioning conventions
- `disease_id`: `{crop}_{snake_case_class}` exactly as in `dkb.json` (e.g., `tomato_early_blight`, `mango_sooty_mould`).
- `library_version`: `L{n}` (e.g., `L1`), bumped on any change to DKB, ontology, templates, vocabulary, or config. The version string embeds a short `config_hash`.
- `caption_id`: `cap_` + first 16 hex of `SHA256(image_id ‖ caption_seed ‖ template_id)`.
- All generated files carry a header/manifest with: `dkb_sha256`, `ontology_build_id`, `template_set_version`, `vocabulary_version`, `config_hash`, `generator_version`.
- Immutability: `knowledge_base/` is read-only to Stage 2; `templates/` and `metadata/label_map.json` are change-controlled (bump versions + re-QA on edit); `captions/caption_library.jsonl` is immutable per version.

## 6. What lives where — quick rules for the implementer
- Domain facts → **only** `knowledge_base/` (never hard-code a symptom/term elsewhere).
- Concept/vocab derivations → `ontology/` + `vocabulary/` (generated, never hand-edited except thin `overrides/`).
- Syntax → `templates/` (no domain content).
- Tunables → `generation/config.yaml` + `validators/rules.yaml` (no magic numbers in code).
- The one folder-string coupling to the filesystem → `metadata/label_map.json` (reconciled to real folders at build time).
