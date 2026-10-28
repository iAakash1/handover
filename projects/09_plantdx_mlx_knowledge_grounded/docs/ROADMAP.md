# Roadmap

Milestones and their acceptance criteria. Each milestone is a separate,
reviewable body of work; scope PRs to a single milestone.

## M1 — Repository scaffolding ✅ (this milestone)
- Production structure, packaging, tooling, configs, typed public APIs, tests, docs.
- **Acceptance:** `make check` green; every module imports; CLI surface parses;
  registries complete; stubs raise `NotImplementedError`. No pipeline logic.

## M2 — Dataset Audit Engine ✅ (implemented)
- CPU-only audit that inventories the configured datasets and writes a
  reproducibility report (`reports/`). See [AUDIT.md](AUDIT.md). Also implemented
  the M2-tagged infrastructure it depends on: config loader, `utils.io`,
  `utils.hashing`, `utils.logging`.
- **Acceptance:** `plantdx audit` runs on CPU and emits `dataset_card.md`,
  per-dataset summaries, CSVs, and a deterministic `audit_manifest.json`; unit
  tests cover discovery, corrupt handling, duplicates, checksum determinism, and
  report generation.

## M2.1 — Dataset Normalization Engine ✅ (implemented)
- CPU-only, filesystem-only. Copies the tomato + mango classes from the immutable
  raw datasets into `datasets/<crop>/processed/<canonical_class>/`, with a class
  map, manifest, dataset card, and combined run report. See [NORMALIZATION.md](NORMALIZATION.md).
- **Acceptance:** `plantdx normalize` extracts only mapped classes, normalizes names,
  merges train/val (preserving split in the manifest), verifies every copied file's
  checksum, and never modifies `raw/`. Unit tests cover layout detection, extraction,
  dedup/collision, checksum verification, and report generation.

## M2.2 — Domain Ontology Compiler ✅ (implemented)
- CPU-only, deterministic `Ontology = f(DKB, Policies)`. Compiles the DKB into a
  typed knowledge graph (`src/plantdx/ontology/domain/`) and writes six artifacts
  to `artifacts/ontology/`. Design: [`ontology_design/`](../ontology_design/); usage:
  [ONTOLOGY.md](ONTOLOGY.md). Distinct from the caption-concept model in
  `plantdx.ontology` (component A, M3 below).
- **Acceptance:** `plantdx ontology` builds + validates (fail-closed) the 18-condition
  graph; repeated builds are byte-identical (content-hash pinned); unit tests cover
  hierarchy, graph generation, ordering, checksum, and every validation-failure mode.

## M2b — Vocabulary Builder + Symptom Lexicon Compiler ✅ (implemented)
- CPU-only, deterministic `Vocabulary = f(Ontology, Policies)`. Projects the compiled
  domain ontology (never the DKB directly) into a flat controlled vocabulary
  (`VocabularyBuilder`, component B) and a bounded symptom lexicon (`SymptomLexiconBuilder`,
  component C) — `src/plantdx/vocabulary/domain/` — and writes six artifacts to
  `artifacts/vocabulary/`. Usage: [VOCABULARY.md](VOCABULARY.md). Re-founds components B/C
  directly onto the domain ontology substrate (`ontology_design/01_architecture.md` §1.5)
  rather than waiting on the still-unimplemented caption-concept model (component A, below).
- **Acceptance:** `plantdx vocabulary` builds + validates (fail-closed) both artifacts
  from the real 18-condition ontology; every item traces to an ontology node and, through
  its grounding relation, to the DKB disease(s) and evidence that licensed it; the lexicon
  is linear in attached quality values (never a Cartesian product) and deterministically
  deduplicates cross-axis word collisions; repeated builds are byte-identical
  (content-hash pinned); unit tests cover category coverage, bounded symptom realizations,
  dedup, traceability, and every `V-VOC-*` validation-failure mode.

## M3 — Caption Concept Model + Template Engine + Corpus + Exporters ✅ (implemented)
- CPU-only, deterministic, **image-independent** language layer. Implements the Caption
  Concept Model (component A, `src/plantdx/concepts/`, `plantdx concepts`), the Template
  Engine (`src/plantdx/templates/`, authored `assets/templates/templates.json`,
  `plantdx templates`), the Sentence Planner + Caption Generator + Caption Validator +
  Corpus Builder (`src/plantdx/corpus/`, `plantdx generate|validate|corpus`), and the
  Dataset Exporters (`src/plantdx/exporters/`). Usage: [CONCEPTS.md](CONCEPTS.md),
  [CORPUS.md](CORPUS.md). Produces a per-disease caption corpus (`artifacts/corpus/`)
  reshaped into `generic`/`llava`/`paligemma`/`blip2`/`messages` formats.
- **Design note.** This milestone deliberately scopes to the *disease-level* corpus (a pure
  function of ontology + vocabulary + lexicon + templates). Image cross-join, instruction
  pairing, and the image-grounded per-model VLM converters move to M4. The concept model
  is derived from the DKB cross-linked to the ontology/vocabulary (doc 01), not a pure
  ontology view, because the domain ontology does not preserve the fine concept typing.
- **Acceptance:** deterministic bit-for-bit corpus (content-hash pinned); zero forbidden
  terms and zero severity-stage tokens in any accepted caption; every disease yields ≥1
  valid caption or the build hard-errors; unit + integration tests cover the concept model,
  templates, planner/generator, every `V-CON-*`/`V-TPL-*`/`V-CAP-*` failure mode, exporters,
  and the CLI; the two M3 generation/reproducibility integration tests are unskipped.

## M4 — Image grounding + Instruction Dataset Builder + splits + converters + QA
- Cross-join the disease-level corpus with the normalized image datasets; implement the
  instruction bank + `(instruction, image, response)` pairing, `Emitter` (I),
  `SplitBuilder`, `LabelResolver`, the image-grounded per-model converters
  (`CONVERTER_REGISTRY`: Qwen2.5-VL/Qwen3-VL/InternVL3/Gemma-3/MLX), and QA.
- **Acceptance:** image-grouped stratified splits; all converters preserve
  response text and validate their lines; QA acceptance rule + kappa computed;
  converter/split integration tests unskipped.

## M7 — Training pipeline (QLoRA on Qwen2.5-VL, MLX) ✅ (implemented)
- Config-driven, deterministic fine-tuning **workflow** for **one model**
  (**Qwen2.5-VL-7B-Instruct-4bit** via mlx-vlm, M4 Pro / 24 GB) across **tomato and
  mango** (`configs/train/qwen25vl_{tomato,mango}.yaml`). Cross-joins the frozen
  corpus (response pool) with the normalized images for the configured crop into
  mlx-vlm JSONL with image-grouped, disease-stratified splits; builds the exact
  `mlx_vlm.lora` command; pre-flight plan + report; checkpoints/resume; CSV/JSON/Markdown
  logging; single/folder/batch inference. `configs/{train,models,lora}/`, `training/` +
  `training/data/`, `assets/{metadata/label_map,training/instructions{,_mango}}.json`.
- **Acceptance:** `prepare-training` / `train --dry-run` build the dataset + report
  and print the exact one launch command without training; DoRA fails closed; the
  frozen golden hashes are unchanged. See `docs/TRAINING.md`. Note: this milestone
  narrows the original M5 (four models) to the single requested Qwen2.5-VL run,
  now covering both tomato and mango; extending to the other three models reuses
  the same config/command machinery.

## M5 — QLoRA fine-tuning (MLX) — superseded for tomato by M7
- Implement `QLoRASettings` resolution and `MLXVLMRunner`; fine-tune all four
  models on the identical library/splits (M4 pin the mlx-vlm version). The tomato +
  Qwen2.5-VL slice is done (M7); the remaining three models follow the same pattern.
- **Acceptance:** each model trains to completion on M4 Pro / 24 GB; adapters
  produced; runs reproducible from a training config.

## M6 — Evaluation (base vs. fine-tuned) ✅ (implemented)
- Two-stage (`--stage inference|analyze|all`) base-vs-fine-tuned comparison on
  the frozen test split of whichever crop's dataset is configured (crop is read
  from the dataset's own `manifest.json`, never a hardcoded assumption — see
  `docs/EVALUATION.md`): official BLEU/ROUGE/METEOR/CIDEr/BERTScore,
  full classification metrics + confusion matrices, per-disease breakdown,
  DKB-grounded hallucination + clinical-correctness detection, response
  quality, latency, paired statistical significance (t-test/Wilcoxon/
  bootstrap), publication-quality figures, and a reproducibility manifest.
  `evaluation/`, `configs` via CLI flags (no YAML layer needed at this size),
  `pyproject.toml`'s `[eval]` extra + `scripts/setup_eval_env.sh`.
- **Acceptance:** `plantdx evaluate` reproduces byte-identical predictions
  given the same adapter/seed; every required output file is written; split
  leakage is a hard failure; 119 tests pass with 0 failures (BERTScore tests
  skip cleanly, not silently, where a pre-existing environment conflict
  blocks it — verified 0 skips in a clean environment). See
  `docs/EVALUATION.md`. Note: this milestone covers the single Qwen2.5-VL model
  this project actually trained (M7), now for both tomato and mango, not the
  four-model zero-shot comparison matrix originally scoped here — extending to
  additional models reuses the same two-stage machinery.
