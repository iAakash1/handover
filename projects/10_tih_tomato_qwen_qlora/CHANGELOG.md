# Changelog

All notable changes to PlantDx are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Streamlit demo app (`app/`, `streamlit_app.py`)
A presentation-only inference app over the trained tomato/mango QLoRA adapters,
reusing `plantdx.training.inference` / `plantdx.evaluation` unchanged (no
training/evaluation/CI/test behavior modified). Upload leaf images → grounded
diagnosis with a **real** generation confidence (mean probability of the model's
own emitted tokens via `stream_generate`, replacing the degenerate last-token
logprob), a configurable confidence threshold with low-confidence/unknown
handling for out-of-distribution images, robust crop-scoped disease
classification (wraps the frozen extractor with normalized/alias fallbacks),
adapter verification (LoRA rank/params/attachment), and the crop's held-out
evaluation results — all in three tabs. Uploads are filed immutably under
`uploads/<crop>/<predicted_class|unknown>/` with a JSON sidecar; every run is
recorded to `predictions/` and `logs/predictions.jsonl`. Model loads once via
`st.cache_resource` (one crop resident at a time); MLX buffer cache is bounded
and cleared per run (memory stays flat). Uses the current `width="stretch"`
Streamlit API (no deprecation warnings). Tests under `tests/unit/app/` (45,
Streamlit-guarded where needed). Docs: `docs/DEMO_APP.md`; README overhauled as
the public entry point with real evaluation results + figures.

### Added — Mango training config + crop-agnostic evaluation
M6 (evaluation) and M7 (training), both below, originally shipped tomato-only.
This extends them to mango without changing either milestone's architecture.
- **`configs/train/qwen25vl_mango.yaml`**: the mango run config — identical
  LoRA/QLoRA, optimizer, schedule, and checkpoint/logging cadence to the tomato
  config; only crop-specific values differ (`data.crop`, `data.classes`,
  `data.image_glob`, `data.instructions_path`, `run_name`; output directories
  auto-derive from `run_name`). `assets/training/instructions_mango.json` (mango-
  worded instruction paraphrases) and a `mango` section in
  `assets/metadata/label_map.json` were added alongside the existing tomato ones.
- **Crop-agnostic evaluation** (`evaluation/config.py`, `evaluation/
  inference_runner.py`, `evaluation/report.py`, `evaluation/hallucination.py`):
  crop is now read from `<dataset_dir>/manifest.json` (the same manifest
  `plantdx train`/`prepare-training` writes) instead of a hardcoded tomato
  default — no `--crop` flag needed. Stage 1 records the resolved crop into
  `metadata.json`; stage 2 reads it back to build the correct disease/
  hallucination/clinical lexicons, falling back to the dataset manifest for
  predictions produced before this change. `--output-dir` now defaults to
  `reports/<run_name>/evaluation`, derived from `--adapter`'s own directory
  name, instead of a hardcoded `reports/qwen25vl_tomato_qlora/evaluation`.
  Hallucinated-crop detection now derives "other crops" from the DKB's own
  crop set minus the crop being evaluated, instead of a static crop-name list.
- Verified against the real, previously-mislabeled mango run: predictions now
  carry correct `mango_*` disease IDs (not `unknown:*` or a tomato ID from a
  folder-name collision), and evaluating a mango dataset no longer produces
  tomato disease IDs anywhere in the output. Tomato evaluation re-verified
  unaffected.
- New regression tests targeting each reported symptom directly: output-dir
  derivation from the adapter path, fail-closed crop resolution, mango
  disease-ID extraction, and per-crop hallucinated-crop true/false positives.
- `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest` (495
  passed, 47 skipped, 0 failed) all green.

### Added — M6: Evaluation pipeline (base vs. fine-tuned, tomato)
A two-stage, deterministic evaluation comparing the fine-tuned tomato QLoRA
adapter against the base Qwen2.5-VL-7B-Instruct-4bit model on the frozen
`test.jsonl` split (910 images), with identical prompts and temperature-0
decoding for both. Never retrains, never regenerates data, never touches the
frozen DKB/ontology/vocabulary/concepts/templates/corpus/training pipeline.
- **Two-stage architecture** (`--stage inference|analyze|all`): stage 1 (lazy
  mlx-vlm) writes `predictions.jsonl`; stage 2 (lazy metrics stack) reads only
  that frozen artifact. The two dependency sets (mlx-vlm vs. matplotlib/
  scikit-learn/scipy/nltk/pycocoevalcap/rouge-score/bert-score+torch) are never
  installed together — `pyproject.toml`'s new `[eval]` extra + `make
  install-eval` (`scripts/setup_eval_env.sh`) install and one-time-cache
  WordNet + the BERTScore backbone so the analyze stage never touches the
  network.
- **Split-integrity guard** (`evaluation/integrity.py`): reads `train.jsonl` +
  the target split read-only and hard-fails (`InvariantViolation`) on any
  image-path overlap; 0 overlap on the real frozen tomato dataset.
- **Official reference metric implementations** (never approximated):
  BLEU-1..4 + CIDEr via `pycocoevalcap` (CIDEr scored over the full batch —
  its TF-IDF degenerates on a single pair), ROUGE-L via Google's
  `rouge-score`, METEOR via `nltk.translate.meteor_score` + WordNet, BERTScore
  via `bert-score`. Disease-label extraction, hallucination detection
  (other-disease/pathogen/treatment/crop/impossible-symptom), and clinical
  correctness (severity-honesty, forbidden terminology) are deterministic
  lexicon matching grounded in the frozen DKB and compiled
  `artifacts/vocabulary/` (never an LLM judge).
- **Full metrics suite**: classification (accuracy/precision/recall/macro-
  weighted-micro-F1/balanced accuracy + confusion matrices) via scikit-learn;
  per-disease breakdown; response-quality heuristics (fluency/redundancy/
  repetition/lexical diversity — no `language-tool-python` dependency);
  latency/throughput/memory aggregation; paired statistical comparison
  (t-test, Wilcoxon, bootstrap CI via `scipy.stats`, NaN-free on degenerate
  zero-variance inputs); a reproducibility manifest (git commit, package
  versions, adapter/corpus/ontology/vocabulary checksums, hardware).
- **Publication-quality figures** (`evaluation/visualize.py`, matplotlib
  only): a small CVD-validated color system (fixed categorical hue order, a
  one-hue sequential ramp for the confusion-matrix heatmap), PNG+SVG for
  every chart, metrics grouped by comparable scale (CIDEr/BERTScore never
  share an axis with 0–1-bounded metrics).
- **CLI**: `plantdx evaluate` (`--stage`, `--adapter`, `--dataset`, `--split`,
  `--model`, `--output-dir`, `--batch-size`, `--max-samples`, `--seed`,
  `--device`), replacing the M6 stub. Every output file the milestone
  requires is written under `<output_dir>/`.
- **Tests/docs**: `tests/unit/evaluation/` (119 tests; BERTScore-dependent
  tests skip cleanly with a clear reason where the environment's numba/NumPy
  ABI conflict blocks it, and were separately verified to pass with 0 skips
  in a clean environment) + `docs/EVALUATION.md` + README setup section.
- **A real, pre-existing environment bug found and documented, not
  patched-around silently**: `bert-score`'s `AutoModel.from_pretrained` path
  transitively imports `librosa` → `numba`, and this shared environment's
  `numba==0.56.4` predates NumPy 2.x, causing `ImportError:
  numpy.core.multiarray failed to import`. Verified unrelated to PlantDx (a
  clean venv with no numba/librosa installed works correctly); `text_metrics.py`
  detects this specific failure and reports the exact fix
  (`pip install -U "numba>=0.59" "llvmlite>=0.42"`) instead of a raw traceback.

### Added — M7: Training pipeline (tomato QLoRA on Qwen2.5-VL-4bit, MLX)
A config-driven, deterministic fine-tuning **workflow** for **tomato only** (10 classes) on
**Qwen2.5-VL-7B-Instruct-4bit** via **mlx-vlm** on Apple Silicon (M4 Pro, 24 GB). It orchestrates
mlx-vlm's tested LoRA trainer rather than hand-rolling an MLX loop; the frozen pipeline
(ontology/vocabulary/concepts/templates/generator/validator/corpus/exporters), the DKB, and the
templates are **untouched** — their golden hashes are unchanged.
- **Config system (`configs/{train,models,lora}/`, `training/config.py`):** three composable YAML
  layers (run × model × LoRA method) into a typed, fail-closed `TrainingConfig`. LoRA/QLoRA/DoRA
  selectable by config; the backend guard rejects DoRA (unsupported by mlx-vlm 0.6.x) with a clear
  message rather than silently running LoRA.
- **Data pipeline (`training/data/`):** cross-joins the frozen caption corpus (the **response
  pool**, used verbatim) with normalized tomato images to emit mlx-vlm JSONL rows
  `{"image","question","answer"}` + `manifest.json`. Image-grouped, disease-stratified splits;
  deterministic SHA-256 fanout; reads image **paths + folder labels only** (no pixels, no LLM/VLM).
  New authored assets: `assets/metadata/label_map.json`, `assets/training/instructions.json`.
- **Orchestration (`training/{command,planner,checkpoints,metrics,scheduler,seeds,reports,models,
  lora,runner}.py`):** exact `mlx_vlm.lora` command builder; pre-flight plan (iters, effective
  batch, time/memory/disk estimates, LR-schedule preview); best/latest/per-epoch checkpoints +
  resume + prune; CSV/JSON/Markdown (+ optional TensorBoard) metrics; a pre-flight report ending in
  the one launch command. MLX is imported lazily so the package imports cleanly in CI.
- **Inference (`training/inference.py`):** single image / folder / batch, programmatic + CLI, lazy MLX.
- **CLI:** `plantdx prepare-training` (build dataset + report, never launches), `plantdx train`
  (`--dry-run` previews; without it, launches mlx-vlm as a subprocess of the current interpreter),
  `plantdx infer`.
- **Tests/docs:** `tests/unit/training/` (config, data determinism + image-grouped splits + row
  shape, command flags + DoRA guard, planner/scheduler/seeds/checkpoints/metrics, runner-prepares-
  without-launching, CLI, inference helpers) + `docs/TRAINING.md`. `logs/` added to `.gitignore`.

### Changed — RC1: caption corpus hardening (deterministic realization-engine improvements)
A publication-grade audit + hardening pass over the disease-level caption corpus, before
freezing the caption pipeline. No architecture/schema/API/validation-strictness changes;
all improvements are in the realization engine and are measurable. Golden `content_hash`es
were deliberately bumped (reviewed) and determinism re-verified; the ontology and vocabulary
hashes are unchanged.
- **Grammar / naturalness:** strip DKB disambiguation parentheticals from quality realizations
  ("yellow (halo)" → "yellow") and spelling notes from disease names ("sooty mould (sooty mold)"
  → "sooty mould"); extend the noun-phrase filter to reject trailing finite verbs ("young leaves
  distort"); collapse adjacent duplicate words/spans in the generator ("raised raised", "on the
  lamina on the lamina"); suppress redundant modifiers already conveyed by the primary sign and
  agent references that restate the disease name (viruses).
- **Healthy class (W5):** decompose `healthy_state` into multiple atomic, evidence-supported
  observations from the DKB healthy fields (diagnostic features, texture, margins) — healthy
  captions **12 → 32**; added 3 healthy templates with varied openers.
- **Balance + diversity (W6):** bounded per-(template, subset) realization variants (2) draw on
  each disease's multiple DKB phrasings — captions **1,070 → 1,966**, per-disease minimum
  **6 → 15**, unique bigrams/trigrams up ~12%, all still unique + validated. `severity_stage`
  tokens are now filtered from realizations at the source, so the `V-CAP-11` rejection count
  drops **17 → 0** (validators retained as defense in depth).
- **Metrics:** `statistics.json` gains a full lexical-diversity block (distinct-1/2/3, unique
  n-gram counts, entropy, opener diversity, mean reuse) and a per-disease balance block.
- Regression tests for every fix; docs (`docs/CORPUS.md`, `docs/CONCEPTS.md`) updated.

### Added — M3: Caption Concept Model + Template Engine + Caption Corpus + Exporters (CPU-only, deterministic, image-free)
- `src/plantdx/concepts/` — the **Caption Concept Model** compiler (component A):
  `ConceptModels = f(DKB, Ontology, Vocabulary)`. Per disease, derives the
  mandatory/optional/forbidden concept sets (20-concept taxonomy), canonical ordering,
  information budget, register policy, sign type, per-concept controlled realizations +
  evidence, and the `never_appear` set. Fail-closed `V-CON-1..11` battery. Entry point
  `plantdx.concepts.build_concept_models`; CLI `plantdx concepts`; artifacts in
  `artifacts/concepts/`. `lesion_size` is omitted (no controlled DKB vocabulary) and
  `severity_stage`/`management` are always forbidden (severity/observability honesty).
- `src/plantdx/templates/` + `assets/templates/templates.json` — the **Template Engine**
  (component E): 33 authored templates across the 8 doc-02 families, using a structured
  segment schema (`lit`/`slot`/`opt`/`list`) so optional-slot deletion is grammatical by
  construction. Fail-closed `V-TPL-1..8` battery; `compatible()` routing; CLI
  `plantdx templates`; `artifacts/templates/template_index.json`.
- `src/plantdx/corpus/` — the **Sentence Planner**, **Caption Generator**, **Caption
  Validator** (independent 12-check `V-CAP-1..12` battery), and **Corpus Builder**.
  Deterministically enumerates a bounded, de-duplicated, validated per-disease caption
  corpus (1,966 captions from the real DKB); drops-and-records failing candidates and
  hard-errors on a disease with zero valid captions. CLI `plantdx generate|validate|corpus`
  (with `--condition`/`--crop`/`--format`/`--all`); artifacts `captions.{json,jsonl,csv}` +
  stats + validation report + checksum in `artifacts/corpus/`. Every caption carries its
  full source-checksum pin (ontology, vocabulary, concepts, templates) and evidence chain.
- `src/plantdx/exporters/` — **Dataset Exporters**: pure reshapers of the one corpus into
  `generic`/`llava`/`paligemma`/`blip2`/`messages` formats (deterministic, byte-identical),
  each with a `manifest.json`. CLI `plantdx corpus --format <F>` / `--all`.
- `configs/paths.yaml`: new `artifacts.{concepts_dir,corpus_dir,exports_dir}` keys.
- Tests under `tests/unit/{concepts,templates,corpus,exporters}/` (build, validation-failure,
  determinism/golden-hash, generation grammar, CLI end-to-end) plus two unskipped M3
  integration tests (forbidden-term absence, bit-for-bit reproducibility). Docs:
  `docs/CONCEPTS.md`, `docs/CORPUS.md`.
- **Scope decision (deliberate):** this milestone builds the *disease-level*,
  image-independent corpus. Image cross-join, instruction pairing, image-based splits, and
  the image-grounded per-model VLM converters (`CONVERTER_REGISTRY`) remain M4; the M1
  image-grounded stubs (`generation/*`, `validation/*`, `diversity/*`,
  `dataset/{emitter,converters}`, `core.types.CaptionRecord`) are untouched. The ontology
  and vocabulary golden `content_hash`es were verified unchanged.

### Added — Vocabulary Builder + Symptom Lexicon Compiler (CPU-only, deterministic)
- `src/plantdx/vocabulary/domain/` — a deterministic `Vocabulary = f(Ontology, Policies)`
  compiler that projects the compiled domain ontology (never the DKB directly) into a
  flat controlled vocabulary (component B: color, shape, texture, extent, leaf region,
  sign type, agent name, disease name, environment, observability + confidence modifiers)
  and a bounded symptom lexicon (component C: one verbatim base realization per symptom,
  plus one linear single-modifier realization per attached quality value on primary,
  modifiable-sign-type symptoms — never a Cartesian product across axes). Modules:
  `models` (shared `LexicalItem` schema), `policies` (category → relation mappings),
  `graph_queries` (shared read-only ontology queries), `builder`, `lexicon`, `validator`
  (fail-closed `V-VOC-1..9` battery), `statistics`, `serialization`, `checksum`. Entry
  point `plantdx.vocabulary.domain.build_vocabulary_result`.
- `plantdx vocabulary` CLI (flags `--config`, `--output`, `--validate-only`,
  `--stats-only`; also `python -m plantdx vocabulary`). Writes six artifacts to
  `artifacts/vocabulary/` (`vocabulary.json`, `symptom_lexicon.json`, `concept_index.json`,
  `statistics.json`, `checksum.txt`, `validation_report.json`) — byte-identical across runs.
- Every item carries the full traceability schema (`id, surface_form, canonical_form,
  concept, concept_id, confidence, source, ontology_node, dkb_reference, evidence,
  language, part_of_speech`), grounding it back through its ontology node to the DKB
  disease(s) and evidence that licensed it.
- Discovered and handled a real cross-axis word collision in the DKB (a few conditions,
  e.g. `mango_gall_midge`, use the same word — "raised" — as both a shape and a texture
  value): the lexicon builder deterministically keeps only the highest-priority axis
  (`MODIFIER_RELATIONS` order) per symptom, so no duplicate realizations are ever emitted.
- Tests under `tests/unit/vocabulary/` (`test_vocabulary_domain_*`, `test_vocabulary_cli.py`):
  category coverage, agent-name labeling, shared-value traceability, bounded (non-combinatorial)
  lexicon realizations, cross-axis dedup, every fail-closed validation mode, determinism,
  a real-DKB golden-hash regression, and real CLI end-to-end runs. Docs: `docs/VOCABULARY.md`.

### Fixed — CI stabilization pass (Ruff / mypy / pytest green; no behavior change)
- Fixed a circular import (`dataset/emitter.py` ⇄ `generation/engine.py`) with a
  `TYPE_CHECKING`-guarded import; zero runtime effect (`from __future__ import annotations`
  already made the annotation lazy).
- Fixed a pytest module-cache collision: two unrelated test fixture files shared the basename
  `_dataset.py` (`tests/unit/audit/` and `tests/unit/normalization/`), and two test modules
  shared `test_discovery.py`, silently shadowing each other under pytest's default import mode.
  Renamed `tests/unit/normalization/_dataset.py` → `_sample_raw_datasets.py` and
  `tests/unit/normalization/test_discovery.py` → `test_layout_detection.py`.
- Updated 3 tests that had drifted from already-implemented reality: `test_cli.py` still
  exercised the pre-domain-compiler `ontology build` stub subcommand; `test_stub_contracts.py`
  still asserted `load_config`/`sha256_hex` raise `NotImplementedError` (both are implemented);
  `test_repo_structure.py` read `paths.yaml` at its pre-nesting key path.
- 120 Ruff findings resolved (docstrings, import order, unicode-in-docstrings, `Path.open()`
  over `open()`, broad `pytest.raises(Exception)` narrowed to the actual observed exception,
  a `policies as P` import alias removed in favor of the qualified name). Two lint rules
  (`N801` on `Qwen2_5VLConverter`, `N818` on `InvariantViolation`) were suppressed with scoped,
  justified `# noqa` rather than renamed, since renaming would break the public API.
- 1 mypy error fixed in `audit/report.py` (an explicit `list[list[object]]` annotation,
  matching an existing sibling function's pattern).
- **Ontology `content_hash` verified unchanged (`sha256:25ae0f6d9692a6d00a8968dc916a3665001bba29dd45616afe5e9b3c49bf2ca4`)
  across every step of this pass** — all changes were formatting, imports, docstrings, and
  type annotations, never semantic.

### Added — Domain Ontology Compiler (CPU-only, deterministic)
- `src/plantdx/ontology/domain/` — a deterministic `Ontology = f(DKB, Policies)` compiler
  that turns the DKB into a typed knowledge graph: `models`, `policies` (fixed T-Box +
  classification maps), `graph`, `builder`, `validator` (fail-closed V-ONT battery),
  `statistics`, `serialization` (canonical JSON), `checksum` (content-only). Entry point
  `plantdx.ontology.domain.compile_ontology`.
- `plantdx ontology` CLI (flags `--config`, `--output`, `--validate-only`, `--stats-only`;
  also `python -m plantdx ontology`). Writes six artifacts to `artifacts/ontology/`
  (`ontology.json`, `concept_graph.json`, `concept_index.json`, `ontology_statistics.json`,
  `ontology_checksum.txt`, `ontology_build.log`) — byte-identical across runs.
- Tests under `tests/unit/ontology/` (`test_domain_*`): DKB loading, hierarchy/inheritance,
  graph generation, ordering, checksum determinism, real-DKB golden-hash regression, and
  every fail-closed validation mode. Docs: `docs/ONTOLOGY.md`.

### Changed
- Repurposed the `plantdx ontology` CLI command from a stub (`ontology build`) to the real
  domain-ontology compiler. The M1 caption-ontology stubs (`plantdx.ontology.{builder,models,
  concept_schema}`) are preserved unchanged; the domain compiler lives in the
  `plantdx.ontology.domain` subpackage so no other package is affected.

### Added — Milestone 2.1: Dataset Normalization Engine (CPU-only, filesystem)
- `src/plantdx/normalization/` — copies the tomato + mango classes from the immutable
  raw datasets into `datasets/<crop>/processed/<canonical_class>/`; structure-agnostic
  layout detection (flat and train/val), canonical class naming, SHA-256 verification,
  duplicate/collision handling, and writers for `class_mapping.json`, `manifest.json`,
  `dataset_card.md`, and a combined `normalization_report.json`. Entry point
  `plantdx.normalization.run_normalization`.
- `plantdx normalize` CLI command (and `python -m plantdx normalize`).
- `configs/normalization.yaml` (+ `NormalizationConfig`, `SourceSpec`, `paths.processed_dir`);
  wired into `config.yaml` includes.
- Tests under `tests/unit/normalization/` (layout detection, extraction, dedup/collision,
  checksum verification, raw immutability, report generation). Docs: `docs/NORMALIZATION.md`.
- `.gitignore`: `/datasets/` (generated normalized output).

### Added — Milestone 2: Dataset Audit Engine (CPU-only)
- `src/plantdx/audit/` — image discovery, metadata inspection (Pillow, no full decode),
  SHA-256 hashing, exact + optional average-hash near-duplicate detection, folder
  validation, per-dataset summaries, deterministic checksums, and report writers
  (CSV/JSON/Markdown + manifest + log). Entry point `plantdx.audit.run_audit`.
- `plantdx audit` CLI command (and `python -m plantdx audit`).
- `configs/audit.yaml` (+ `AuditConfig`, `paths.reports_dir`); wired into `config.yaml` includes.
- Implemented the M2 infrastructure the audit depends on: config loader
  (`config/loader.py`), `utils/io.py`, `utils/hashing.py`, `utils/logging.py`.
- Tests under `tests/unit/audit/` (discovery, corrupt handling, duplicates, checksum
  determinism, report/JSON/CSV/manifest generation). Docs: `docs/AUDIT.md`.

### Changed
- `configs/paths.yaml` nested under a top-level `paths:` key so the loader maps it to
  `PlantDxConfig.paths` (consistent with the other included config files).
- Pillow moved from the `train` extra to a core runtime dependency (needed by the audit).
- `reports/` added to `.gitignore` (generated audit output).

### Changed — post-review simplification (no methodology change)
- Removed `core/interfaces.py` (unused generic Protocols that duplicated the concrete ABCs).
- Refactored `core/seeding.py` from a `SeedDeriver` class to pure module-level functions
  (`image_seed`, `caption_seed`, `attempt_seed`); updated `CaptionEngine` to take `global_seed`.
- Consolidated the `dataset/converters/` package (7 files) into a single `dataset/converters.py`
  (base ABC + 5 converters + `CONVERTER_REGISTRY`); public API and registry unchanged.
- Renamed `dataset/schema.py` → `dataset/serialization.py` (disambiguates from `config/schema.py`).
- Removed unused dependency `typing-extensions`; moved the M3 generation/diversity stack
  (`numpy`, `tqdm`, `datasketch`, `sacrebleu`) to a `generate` extra.
- Trimmed `__init__.py` re-exports to true public entry points across all subpackages.
- Removed speculative config: per-validator enable/blocking toggles, duplicated `max_attempts`,
  `reproducibility.hash_algorithm`, and the duplicated eval output dir.

### Added — Milestone 1: repository scaffolding
- Production repository structure (`src/` layout, `configs/`, `tests/`, `docs/`, `assets/`, `artifacts/`).
- Packaging and tooling: `pyproject.toml`, `requirements*.txt`, `.pre-commit-config.yaml`,
  `.editorconfig`, `.gitignore`, `Makefile`, `MANIFEST.in`, Apache-2.0 `LICENSE`, `NOTICE`, `CITATION.cff`.
- Typed public APIs and module responsibilities for every pipeline component
  (ontology, vocabulary, generation, validation, diversity, dataset, qa, training, evaluation) —
  **interfaces only**, no algorithm implementations.
- Configuration files: `config.yaml`, `paths.yaml`, `generation.yaml`, `validation.yaml`, `training.yaml`.
- Test scaffolding (unit / integration / benchmark) and developer documentation.
- CLI surface (`plantdx …`) with stubbed subcommands.

### Notes
- No pipeline logic is implemented in this milestone; stage entry points raise
  `NotImplementedError` pointing to their target milestone.
- Stage 1 (DKB) and the Stage 2 specification were completed prior to this changelog
  and live in `knowledge_base/` and `caption_framework/`.

<!-- Milestone template:
## [X.Y.Z] - YYYY-MM-DD
### Added
### Changed
### Fixed
-->
