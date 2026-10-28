# PlantDx Documentation

Developer-facing documentation. The **research/design specification** (the source
of truth) lives in [`../caption_framework/`](../caption_framework/) and
[`../knowledge_base/`](../knowledge_base/); the docs here explain how the code
implements it.

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the `src/plantdx/` packages map to spec components A–I and the pipeline stages. |
| [REPO_LAYOUT.md](REPO_LAYOUT.md) | Repository layout and its correspondence to the spec's artifact tree (doc 06). |
| [CONFIGURATION.md](CONFIGURATION.md) | Configuration reference (`configs/`), precedence, and tunables. |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Dev environment, workflow, testing, coding standards. |
| [AUDIT.md](AUDIT.md) | The Dataset Audit Engine (`plantdx audit`): usage, artifacts, guarantees. |
| [NORMALIZATION.md](NORMALIZATION.md) | The Dataset Normalization Engine (`plantdx normalize`): usage, artifacts, guarantees. |
| [ONTOLOGY.md](ONTOLOGY.md) | The Domain Ontology Compiler (`plantdx ontology`): usage, artifacts, exit codes. |
| [VOCABULARY.md](VOCABULARY.md) | The Vocabulary + Symptom Lexicon Compiler (`plantdx vocabulary`): usage, artifacts, validation battery. |
| [CONCEPTS.md](CONCEPTS.md) | The Caption Concept Model compiler (`plantdx concepts`): usage, artifacts, `V-CON-*` battery. |
| [CORPUS.md](CORPUS.md) | The language layer (`plantdx templates/generate/validate/corpus`): planner, generator, validators, exporters. |
| [TRAINING.md](TRAINING.md) | The training pipeline (`plantdx train`/`prepare-training`/`infer`): QLoRA fine-tuning of Qwen2.5-VL for tomato or mango via mlx-vlm. |
| [EVALUATION.md](EVALUATION.md) | The evaluation pipeline (`plantdx evaluate`): base vs. fine-tuned comparison, crop derived from the dataset manifest. |
| [DEMO_APP.md](DEMO_APP.md) | The Streamlit inference/demo app (`app/`, `streamlit_app.py`): launch, features, confidence/unknown handling, adapter verification. |
| [ROADMAP.md](ROADMAP.md) | Milestones and their acceptance criteria. |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Known, intentionally deferred issues. |
| [adr/](adr/) | Architecture Decision Records. |

## The seven design invariants (every change must preserve these)

1. Label-only grounding · 2. DKB is the single source of truth · 3. Closed
vocabulary · 4. Observability (single-leaf) · 5. Pest/pathogen register integrity
· 6. Severity honesty · 7. Reproducibility.

See [`../caption_framework/README.md`](../caption_framework/README.md) for the
authoritative statement.
