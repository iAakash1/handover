"""Deterministic concept-model statistics (``statistics.json``)."""

from __future__ import annotations

from typing import Any

from plantdx.concepts.models import ConceptModelSet
from plantdx.concepts.policies import CONCEPT_ORDER


def compute(result: ConceptModelSet, validation_status: str) -> dict[str, Any]:
    """Return the statistics document (all values deterministic)."""
    models = result.disease_models
    by_condition_type: dict[str, int] = {}
    concept_availability: dict[str, int] = dict.fromkeys(CONCEPT_ORDER, 0)
    total_mandatory = total_optional = total_forbidden = 0
    for model in models:
        by_condition_type[model.condition_type] = by_condition_type.get(model.condition_type, 0) + 1
        total_mandatory += len(model.mandatory)
        total_optional += len(model.optional)
        total_forbidden += len(model.forbidden)
        for cid in model.mandatory + model.optional:
            concept_availability[cid] += 1

    return {
        "concepts_version": result.provenance.get("concepts_version", ""),
        "build_checksum": result.provenance.get("content_hash", ""),
        "validation_status": validation_status,
        "disease_count": len(models),
        "condition_types": {k: by_condition_type[k] for k in sorted(by_condition_type)},
        "total_mandatory": total_mandatory,
        "total_optional": total_optional,
        "total_forbidden": total_forbidden,
        "mean_optional_per_disease": round(total_optional / len(models), 4) if models else 0.0,
        "concept_availability": {
            c: concept_availability[c] for c in CONCEPT_ORDER if concept_availability[c] > 0
        },
    }
