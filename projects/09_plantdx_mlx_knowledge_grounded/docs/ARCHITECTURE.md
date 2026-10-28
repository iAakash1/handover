# Architecture

How the `src/plantdx/` code maps to the caption-framework specification
(`caption_framework/`). The spec labels the runtime components **A–I**
(`00_methodology_overview.md §2`); each maps to a package here.

## Component → package map

| Spec component | Package / module | Interface class | Milestone |
|----------------|------------------|-----------------|-----------|
| — DKB loader | `knowledge_base/loader.py` | `DKBLoader` | M4 |
| — Domain ontology substrate | `ontology/domain/` | `compile_ontology` | M2.2 ✅ |
| — Domain vocabulary substrate | `vocabulary/domain/` | `build_vocabulary_result` | M2b ✅ |
| (A) Caption Concept Model | `concepts/` | `build_concept_models` | M3 ✅ |
| (B) Vocabulary Builder | `vocabulary/domain/` | `build_vocabulary_result` | M2b ✅ |
| (C) Symptom Lexicon Builder | `vocabulary/domain/lexicon.py` | `build_lexicon` | M2b ✅ |
| (E) Template Engine | `templates/` | `load_library`, `compatible` | M3 ✅ |
| (D/F) Sentence Planner + Generator | `corpus/planner.py`, `corpus/generator.py` | `plan_caption`, `generate` | M3 ✅ |
| (G) Caption Validator | `corpus/validator.py` | `validate_caption` (`V-CAP-1..12`) | M3 ✅ |
| (H/I) Corpus Builder | `corpus/builder.py` | `build_corpus` | M3 ✅ |
| Dataset Exporters | `exporters/` | `write_all` (generic/llava/paligemma/blip2/messages) | M3 ✅ |
| Image grounding + Emitter | `dataset/emitter.py` | `Emitter` | M4 |
| Splits | `dataset/splits.py` | `SplitBuilder` | M4 |
| Per-model VLM converters | `dataset/converters.py` | `BaseConverter` + 5 subclasses + `CONVERTER_REGISTRY` | M4 |
| QA | `qa/*` | `AuditSampler`, `ReviewStore`, `AcceptanceEvaluator` | M4 |
| Training | `training/*` | `MLXVLMRunner`, `QLoRASettings` | M5 |
| Evaluation | `evaluation/*` | `ZeroShotEvaluator`, `ComparisonReporter` | M6 |

> **M3 built a disease-level, image-independent language layer.** The Caption
> Concept Model (A) is derived from the DKB cross-linked to the ontology/vocabulary
> (doc 01), and the corpus is a pure function of concept models + templates — no
> images, no instruction pairing, no splits. Those (and the image-grounded per-model
> VLM converters in the tested `CONVERTER_REGISTRY`) are M4. The M1 stub modules
> `generation/*`, `validation/*`, `diversity/*`, `dataset/{emitter,converters,...}`
> and the image-grounded `core.types.CaptionRecord` remain untouched, awaiting M4 —
> the new work lives in the fresh `concepts/`, `templates/`, `corpus/`, `exporters/`
> packages (the same "new package coexists with old stub" pattern as
> `ontology/domain/` and `vocabulary/domain/`).

Supporting layers: `core/` (types, enums, exceptions, seeding — a leaf package
depending only on stdlib), `config/` (typed schema + loader), `utils/` (hashing,
io, logging, versioning).

## Dependency direction

```
core  ◄─ everything          (leaf; no PlantDx deps)
config ◄─ (uses core)
knowledge_base ◄─ ontology ◄─ vocabulary
                     └────────────┐
generation (D,E,F, engine) ─► validation, diversity, dataset(emitter)
dataset (schema, splits, converters, instructions, label_map)
qa, training, evaluation depend on dataset/core
```

Rules:
- `core` never imports another PlantDx package.
- Builders (A/B/C) run once at build time and only read the DKB + assets.
- The generation path (D–I) never imports a neural model and never reads pixels
  (invariant #1). Training/evaluation are the only packages that touch models.

## Data flow (runtime)

`dkb.json` → (A/B/C) `ontology/` + `vocabulary/` artifacts → per image the
`CaptionEngine` runs D→E→F→G→H→I → `captions/caption_library.jsonl` → diversity
metrics + QA → splits → per-model converters → training → evaluation. Full
diagram: `caption_framework/00_methodology_overview.md §2`.

## Determinism & provenance

All randomness derives from one `global_seed` via the pure functions in
`core.seeding` (`image_seed`/`caption_seed`/`attempt_seed`, SHA-256 fan-out).
Every emitted record carries a `Provenance` object (`core/types.py`) sufficient
to regenerate it bit-for-bit (doc 00 §6).
