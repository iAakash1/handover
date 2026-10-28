# Caption Concept Model

A CPU-only, **deterministic** compiler: `ConceptModels = f(DKB, Ontology, Vocabulary)`.
The **Caption Concept Model** is the intermediate representation between the
knowledge layers (ontology + vocabulary + symptom lexicon) and language. It is
**not** text, **not** a template, **not** a sentence — it is, per disease, the
deterministic answer to *"which concepts may a caption of this disease assert, in
what order, with what confidence and observability, and which are forbidden — and
what controlled surface phrases and evidence back each one?"* Every caption is
generated from this model.

Faithful to `caption_framework/01_caption_ontology_spec.md`: the model is derived
from the DKB (its designed input) and cross-linked to the compiled domain
ontology and vocabulary for evidence, confidence, sign types, and controlled
realizations. It uses no images, no datasets, no LLM/VLM, and no randomness —
repeated builds are byte-identical. This is component **A** of the caption
framework; consumed downstream by the Template Engine, Sentence Planner, Caption
Generator, and Caption Validator.

## What each disease model contains

Per disease (18 total): `mandatory` / `optional` / `forbidden` concept sets (from
the 20-concept `ConceptId` taxonomy), a canonical `ordering`, the min/max
information budget, the `register_policy`, the effective `sign_type`, a
`never_appear` term set, and one `CaptionConcept` per concept carrying its
`status`, `observable` flag, `confidence`, `sign_type`, controlled
`realizations`, legal `modifiers`, `evidence`, and source `dkb_fields`.

Two concepts from the doc-01 taxonomy are deliberately **omitted / gated**:
- **`lesion_size`** — the DKB has no controlled size vocabulary (only free-text
  prose like "~3–12 mm"); synthesizing one would violate no-invention, so size is
  never available.
- **`severity_stage`** and **`management`** — gated (severity-honesty policy) and
  non-visual respectively; always forbidden in this milestone's corpus.

## Run it

```bash
plantdx concepts                       # build + validate + write artifacts
plantdx concepts --validate-only       # build + validate; write nothing
plantdx concepts --stats-only          # build + validate; print statistics JSON
plantdx concepts --output some/dir     # override output directory
python -m plantdx concepts             # equivalent, without the console script
```

## Artifacts

Written to `artifacts/concepts/` (gitignored):

| File | Contents |
|------|----------|
| `concept_models.json` | One model per disease: concept sets, ordering, budget, per-concept detail + evidence. |
| `statistics.json` | Counts by condition type + concept availability, checksum, validation status. |
| `validation_report.json` | The `V-CON-*` battery's pass report. |
| `checksum.txt` | The content-only SHA-256 (`sha256:…`); identifies the build. |

## Validation battery (fail closed)

| Rule | Guards against |
|------|-----------------|
| `V-CON-1` | Missing/duplicate disease coverage (one model per ontology condition). |
| `V-CON-2` | Unknown concept ids or statuses. |
| `V-CON-3` | A mandatory concept with no realization. |
| `V-CON-4` | A forbidden concept carrying realizations; missing always-forbidden concepts. |
| `V-CON-5` | An inconsistent information budget. |
| `V-CON-6` | An observability flag that disagrees with the fixed non-observable set. |
| `V-CON-7` | A quality/modifier concept offered when the primary sign is not modifiable. |
| `V-CON-8` | An evidence-required concept with realizations but no evidence (traceability). |
| `V-CON-9` | A `never_appear` set missing the severity-stage tokens, or forbidding a mandatory phrase. |
| `V-CON-10` | Healthy/disease routing (healthy must require healthy_state and forbid disease signs). |
| `V-CON-11` | An ordering that is not the canonical concept order. |

## Guarantees

- **Deterministic.** No timestamps, no randomness; the checksum depends only on
  the model content. Golden hash pinned in
  `tests/unit/concepts/test_concepts_determinism.py`.
- **Traceable.** Every scientific concept carries `evidence` (citation ids) and
  `dkb_fields`; nothing is invented.
- **Observability- and severity-honest by construction** — non-observable
  concepts are flagged; severity stages are always forbidden.
- **Fail closed.** Any rule violation aborts the build with a specific `V-CON-*` error.
