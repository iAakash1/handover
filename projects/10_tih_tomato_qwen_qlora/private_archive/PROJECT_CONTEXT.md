# PlantDx — Project Context

A short orientation for collaborators and contributors. For the full,
exhaustive engineering history and rationale, see the (gitignored,
AI-assistant-local) `CLAUDE.md` — this document is the public,
always-current summary.

## What this project is

PlantDx builds **knowledge-grounded instruction-tuning datasets** for
fine-tuning open-weight Vision-Language Models (VLMs) to describe and
diagnose diseases in tomato and mango leaves. Every caption the pipeline will
eventually produce is required to trace back to a cited, peer-reviewed or
extension-service scientific source — never to an LLM's or VLM's own guess.
This is a direct response to an earlier finding: a zero-shot benchmark of
general-purpose VLMs on crop-disease diagnosis performed poorly, so no model
is ever used to *generate or judge* training captions in this pipeline.

The end goal is an IEEE paper demonstrating that this knowledge-grounded,
deterministic approach produces better instruction-tuning supervision than
rule-based templates alone, LLM-generated captions, or VLM-self-captioned
images — backed by a released, reproducible artifact (not just a claim).

## Architecture at a glance

```
Disease Knowledge Base (FINAL, cited)
        │
        ▼
Domain Ontology Compiler ──► typed knowledge graph (deterministic, content-hashed)
        │
        ▼
Vocabulary + Symptom Lexicon Compiler ──► controlled vocabulary + bounded lexicon
        │
        ▼
Caption Concept Model ──► Template Engine ──► Sentence Planner ──► Caption Generator
        │                                                              │
        ▼                                                              ▼
Caption Validator ──► Corpus Builder ──► caption corpus ──► Dataset Exporters
        │
        ▼
QLoRA Fine-tuning (Qwen2.5-VL, tomato or mango) ──► Evaluation (base vs. fine-tuned)
        │
        ▼
[ next: image grounding + instruction pairing → per-model VLM converters
  (Qwen3-VL/InternVL3/Gemma-3) — the four-model comparison this narrows from ]

Raw datasets ──► Dataset Audit Engine ──► Dataset Normalization Engine
                                            ──► canonical datasets/<crop>/processed/
```

Eight independent, CPU-only pipeline stages are implemented today, plus a
QLoRA training workflow (`plantdx train`) and evaluation pipeline
(`plantdx evaluate`) for Qwen2.5-VL on tomato and mango; image grounding,
instruction pairing, and the multi-model VLM converters remain a typed
interface stub (`raise NotImplementedError`), by design — see
[Current Status](#current-status). The M3 caption corpus is **disease-level
and image-independent** — a pure function of the ontology, vocabulary,
lexicon, and templates.

## Implemented today

| Stage | Command | What it does |
|-------|---------|---------------|
| **Dataset Audit Engine** | `plantdx audit` | Inventories the raw datasets: counts, class distribution, corrupt/duplicate detection, deterministic checksums. Read-only. |
| **Dataset Normalization Engine** | `plantdx normalize` | Extracts and canonicalizes tomato/mango classes from the raw datasets into `datasets/<crop>/processed/<class>/`. Raw datasets are never modified. |
| **Domain Ontology Compiler** | `plantdx ontology` | Compiles `knowledge_base/dkb.json` into a typed, evidence-linked knowledge graph. Fail-closed validation; deterministic, content-hashed output. |
| **Vocabulary + Symptom Lexicon Compiler** | `plantdx vocabulary` | Projects the domain ontology into a controlled vocabulary and a bounded symptom lexicon, every item traceable back to its ontology node, DKB disease(s), and evidence. Fail-closed validation; deterministic, content-hashed output. |
| **Caption Concept Model** | `plantdx concepts` | Derives, per disease, the mandatory/optional/forbidden concept sets, ordering, information budget, register policy, and per-concept controlled realizations + evidence. Fail-closed `V-CON-*` validation; deterministic, content-hashed. |
| **Template Engine** | `plantdx templates` | Loads, validates, and indexes the authored caption templates (syntax only; slots name concepts). Fail-closed `V-TPL-*` validation. |
| **Caption corpus** | `plantdx generate` / `validate` / `corpus` | Deterministically plans → generates → independently validates (`V-CAP-*`) → assembles a per-disease, image-free caption corpus, and reshapes it into `generic`/`llava`/`paligemma`/`blip2`/`messages` export formats. |
| **Training (M7)** | `plantdx train` / `prepare-training` / `infer` | Config-driven QLoRA fine-tuning of Qwen2.5-VL-7B-Instruct-4bit via mlx-vlm, for tomato or mango (`configs/train/qwen25vl_{tomato,mango}.yaml`). Cross-joins the frozen caption corpus with the crop's normalized images directly (paths + folder labels only) — a narrower path than the still-stubbed image-grounded Instruction Dataset Builder below. |
| **Evaluation (M6)** | `plantdx evaluate` | Two-stage base-vs-fine-tuned comparison on the frozen test split. Crop is read from the dataset's own `manifest.json`, never hardcoded — see `docs/EVALUATION.md`. |

The first eight are CPU-only, fully deterministic (byte-identical output from
the same inputs); training and evaluation additionally require mlx-vlm
(Apple Silicon) and, for evaluation's metrics stack, a separate `[eval]`
environment. All are covered by CI (`ruff check`, `ruff format --check`,
`pytest`, `mypy` — all green).

## Not yet implemented

The originally-planned **image-grounded** path: cross-joining captions with the
normalized image datasets through a formal Instruction Dataset Builder,
instruction pairing, image-based splits, and the per-model VLM converters for
the other three target models (Qwen3-VL/InternVL3/Gemma-3, `CONVERTER_REGISTRY`).
These exist today only as typed package interfaces with `NotImplementedError`
bodies — the public API shape is fixed, the logic is not written. QLoRA
training and evaluation were built directly (see above), reading image paths
and folder labels straight from the normalized datasets rather than going
through this still-stubbed path. **Image grounding + the Instruction Dataset
Builder is next.**

## Repository layout

```
src/plantdx/            the Python package (see table above for what's real vs. stubbed)
knowledge_base/          Stage 1 — the Disease Knowledge Base (FINAL; 18 diseases, cited)
caption_framework/       Stage 2 — the caption-generation design spec (FINAL; no code)
ontology_design/         Stage 3 — the domain-ontology design spec (FINAL; no code)
configs/                 all pipeline configuration (YAML)
tests/                   mirrors src/plantdx/
docs/                    developer docs: AUDIT.md, NORMALIZATION.md, ONTOLOGY.md, VOCABULARY.md, CONCEPTS.md, CORPUS.md, KNOWN_ISSUES.md
artifacts/, datasets/    generated outputs (gitignored; regenerable from source + config)
tomato/raw/, mango/raw/  the two raw datasets (gitignored; immutable, never edited by the pipeline)
```

## Datasets

- **Tomato** — raw source is the *full* PlantVillage dataset (multi-crop,
  ~54k images, 38 classes) — normalization extracts the 10 tomato-relevant
  classes and canonicalizes their names.
- **Mango** — MangoLeafBD, 4,000 images, 8 classes, already flat/clean.

Both are treated as immutable ground truth; the pipeline never relabels or
infers from pixels. See `docs/AUDIT.md` and `docs/NORMALIZATION.md`.

## Disease Knowledge Base

`knowledge_base/dkb.json` — 18 hand-curated, cited disease records (10
tomato + 8 mango), each with 46 fields covering identity, causal agent,
symptoms, controlled vocabularies, and references resolving into a shared
`reference_registry`. This file is the **single source of scientific truth**
for the entire project — every downstream fact must trace back to it.

## Core design principles

1. **Deterministic** — every stage is a pure function of its declared
   inputs; no wall-clock, no unseeded randomness.
2. **No hallucination by construction** — a caption can only ever assert
   what's structurally licensed by the ontology (observable, evidenced,
   within the closed vocabulary) — not filtered after the fact, made
   impossible up front.
3. **Observability-honest** — both datasets are single-leaf images; the
   system can never claim a symptom that wouldn't be visible in one (no
   fruit/twig/whole-plant claims), enforced as a structural graph property.
4. **Severity-honest** — no per-image severity claims, since no such label
   exists in the source data (only class-level severity knowledge is
   retained).
5. **DKB-first, ontology-first, vocabulary-first, template-first** — a
   strict layering where scientific fact (DKB) → formal knowledge (ontology)
   → controlled language (vocabulary) → sentence structure (templates) are
   kept separate and each is independently reviewable.

## CI

```bash
ruff check .            # lint
ruff format --check .   # formatting
pytest                  # tests (unit / integration / benchmark markers)
mypy                    # strict type checking
```
All four are green as of the latest commit. GitHub Actions
(`.github/workflows/ci.yml`) runs this matrix across Python 3.10–3.12 on
every push/PR to `main`.

## Getting started

```bash
git clone git@github.com:iAakash1/experimentation.git && cd experimentation
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

plantdx audit                    # inventory the raw datasets
plantdx normalize --dataset mango  # build the canonical normalized mango dataset
plantdx ontology                  # compile the DKB into the knowledge graph
plantdx vocabulary                 # derive the vocabulary + symptom lexicon from the ontology
plantdx concepts                   # derive the per-disease Caption Concept Model
plantdx corpus                     # build the caption corpus + dataset exporters
```

Training (`plantdx train`, Apple Silicon + `[train]` extra) and evaluation
(`plantdx evaluate`, a separate `[eval]` extra) each need their own environment
— see `docs/TRAINING.md` and `docs/EVALUATION.md`.

See `docs/AUDIT.md`, `docs/NORMALIZATION.md`, `docs/ONTOLOGY.md`,
`docs/VOCABULARY.md`, `docs/CONCEPTS.md`, and `docs/CORPUS.md` for full usage of
each implemented stage, and `docs/ROADMAP.md` for the complete milestone plan.

## Known issues

See [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — currently one deferred
issue: re-running `plantdx normalize` on an already-normalized dataset causes
its manifest/report to show 0 images (the actual normalized files are
unaffected; only that report is stale until the crop's `processed/` output
is cleared and rebuilt). Not dangerous, understood, deliberately deferred
rather than fixed as part of an unrelated CI-stabilization pass.

## Contributing

See `CONTRIBUTING.md` and `docs/DEVELOPMENT.md`. The short version: the DKB
and both design specifications (`caption_framework/`, `ontology_design/`) are
**FINAL** — implementation must follow them exactly, not redesign them. Every
change should preserve determinism, reproducibility, and the observability/
severity-honesty invariants described above. When in doubt, ask before
deviating from a written spec.
