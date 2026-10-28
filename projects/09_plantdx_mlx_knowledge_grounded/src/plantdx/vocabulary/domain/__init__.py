"""Vocabulary + symptom lexicon compiler (``Vocabulary = f(Ontology, Policies)``).

Deterministic: building from the same ontology with the same policies yields
byte-identical artifacts (no timestamps, no randomness, everything sorted).
Consumes only the compiled domain ontology (:mod:`plantdx.ontology.domain`) —
never the DKB directly, never an LLM/VLM, never randomness. Component (B) is
the flat concept vocabulary; component (C) is the bounded symptom lexicon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plantdx.ontology.domain.models import Ontology
from plantdx.utils.io import ensure_dir
from plantdx.vocabulary.domain import (
    builder,
    checksum,
    lexicon,
    policies,
    serialization,
    statistics,
    validator,
)
from plantdx.vocabulary.domain.models import VocabularyResult
from plantdx.vocabulary.domain.validator import VocabularyValidationError

__all__ = [
    "ARTIFACT_NAMES",
    "VocabularyResult",
    "VocabularyValidationError",
    "build_vocabulary_result",
    "compute_statistics",
    "validate_vocabulary_result",
    "write_artifacts",
]

ARTIFACT_NAMES = (
    "vocabulary.json",
    "symptom_lexicon.json",
    "concept_index.json",
    "statistics.json",
    "checksum.txt",
    "validation_report.json",
)


def build_vocabulary_result(ontology: Ontology) -> VocabularyResult:
    """Derive the vocabulary + symptom lexicon from a compiled ontology (no I/O)."""
    result = VocabularyResult(
        vocabulary_items=builder.build_vocabulary(ontology),
        lexicon_items=lexicon.build_lexicon(ontology),
        provenance={
            "ontology_content_hash": ontology.provenance.get("content_hash", ""),
            "builder": "plantdx.vocabulary.domain",
            "vocabulary_version": policies.VOCABULARY_VERSION,
        },
    )
    result.provenance["content_hash"] = checksum.content_hash(result)
    return result


def validate_vocabulary_result(result: VocabularyResult, ontology: Ontology) -> dict[str, Any]:
    """Run the validator battery; raise on any error, else return the validation report."""
    violations = validator.collect_violations(result, ontology)
    if violations:
        raise VocabularyValidationError(violations)
    return {
        "kind": "plantdx.vocabulary.validation_report",
        "vocabulary_version": policies.VOCABULARY_VERSION,
        "status": "valid",
        "checks_run": 9,
        "violation_count": 0,
        "vocabulary_item_count": len(result.vocabulary_items),
        "lexicon_item_count": len(result.lexicon_items),
    }


def compute_statistics(result: VocabularyResult, validation_status: str) -> dict[str, Any]:
    """Compute the statistics document."""
    return statistics.compute(result, validation_status)


def write_artifacts(
    result: VocabularyResult,
    out_dir: str | Path,
    stats: dict[str, Any],
    validation_report: dict[str, Any],
) -> list[Path]:
    """Write all six artifacts deterministically. Returns the written paths."""
    out = ensure_dir(out_dir)
    written: list[Path] = []

    def _write(name: str, text: str) -> None:
        path = out / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    _write(
        "vocabulary.json", serialization.canonical_json(serialization.vocabulary_document(result))
    )
    _write(
        "symptom_lexicon.json", serialization.canonical_json(serialization.lexicon_document(result))
    )
    _write(
        "concept_index.json",
        serialization.canonical_json(serialization.concept_index_document(result)),
    )
    _write("statistics.json", serialization.canonical_json(stats))
    _write("checksum.txt", result.provenance["content_hash"] + "\n")
    _write("validation_report.json", serialization.canonical_json(validation_report))
    return written
