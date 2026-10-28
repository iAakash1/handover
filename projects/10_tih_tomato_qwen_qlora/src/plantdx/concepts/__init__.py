"""Caption Concept Model compiler (``ConceptModels = f(DKB, Ontology, Vocabulary)``).

The **Caption Concept Model** is the intermediate representation between the
knowledge layers (ontology + vocabulary + symptom lexicon) and language. It is
NOT text and NOT a template: it is, per disease, the deterministic answer to
"which concepts may a caption assert, in what order, with what confidence and
observability, and which are forbidden — and what controlled surface phrases and
evidence back each one."

Faithful to ``caption_framework/01_caption_ontology_spec.md``: the concept model
is derived from the DKB (its designed input) and cross-linked to the compiled
domain ontology and vocabulary for evidence, confidence, sign types, and
controlled realizations. It never reads an image, never calls an LLM/VLM, and is
fully deterministic — the same inputs yield byte-identical artifacts. Component A
of the caption framework; consumed downstream by the Template Engine, Sentence
Planner, Caption Generator, and Caption Validator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plantdx.concepts import builder, checksum, serialization, statistics, validator
from plantdx.concepts.models import ConceptModelSet
from plantdx.concepts.validator import ConceptValidationError
from plantdx.ontology.domain.models import Ontology
from plantdx.utils.io import ensure_dir
from plantdx.vocabulary.domain.models import VocabularyResult

__all__ = [
    "ARTIFACT_NAMES",
    "ConceptModelSet",
    "ConceptValidationError",
    "build_concept_models",
    "compute_statistics",
    "validate_concept_models",
    "write_artifacts",
]

ARTIFACT_NAMES = (
    "concept_models.json",
    "statistics.json",
    "validation_report.json",
    "checksum.txt",
)


def build_concept_models(
    dkb: dict[str, Any], ontology: Ontology, vocabulary: VocabularyResult
) -> ConceptModelSet:
    """Derive the per-disease caption concept models (no I/O)."""
    result = builder.build_concept_models(dkb, ontology, vocabulary)
    result.provenance["content_hash"] = checksum.content_hash(result)
    return result


def validate_concept_models(
    result: ConceptModelSet, ontology: Ontology, vocabulary: VocabularyResult
) -> dict[str, Any]:
    """Run the validator battery; raise on any error, else return the report."""
    violations = validator.collect_violations(result, ontology, vocabulary)
    if violations:
        raise ConceptValidationError(violations)
    return {
        "kind": "plantdx.concepts.validation_report",
        "status": "valid",
        "checks_run": validator.CHECK_COUNT,
        "violation_count": 0,
        "disease_count": len(result.disease_models),
    }


def compute_statistics(result: ConceptModelSet, validation_status: str) -> dict[str, Any]:
    """Compute the statistics document."""
    return statistics.compute(result, validation_status)


def write_artifacts(
    result: ConceptModelSet,
    out_dir: str | Path,
    stats: dict[str, Any],
    validation_report: dict[str, Any],
) -> list[Path]:
    """Write all four artifacts deterministically. Returns the written paths."""
    out = ensure_dir(out_dir)
    written: list[Path] = []

    def _write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    _write(
        "concept_models.json",
        serialization.canonical_json(serialization.concept_models_document(result)),
    )
    _write("statistics.json", serialization.canonical_json(stats))
    _write("validation_report.json", serialization.canonical_json(validation_report))
    _write("checksum.txt", result.provenance["content_hash"] + "\n")
    return written
