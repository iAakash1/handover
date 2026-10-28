"""Canonical, deterministic serialization of the concept-model artifacts.

Reuses :func:`plantdx.ontology.domain.serialization.canonical_json` so every
compiler in the stack shares one JSON convention (sorted keys, 2-space indent,
UTF-8, trailing newline). No timestamps, no machine-dependent ordering.
"""

from __future__ import annotations

from typing import Any

from plantdx.concepts import policies
from plantdx.concepts.models import CaptionConcept, ConceptModel, ConceptModelSet
from plantdx.ontology.domain.serialization import canonical_json

__all__ = [
    "canonical_json",
    "concept_dict",
    "concept_models_document",
    "model_dict",
    "semantic_content",
]


def concept_dict(concept: CaptionConcept) -> dict[str, Any]:
    """One :class:`CaptionConcept` as a canonical dict."""
    return {
        "concept_id": concept.concept_id,
        "status": concept.status,
        "observable": concept.observable,
        "confidence": concept.confidence,
        "sign_type": concept.sign_type,
        "realizations": list(concept.realizations),
        "modifiers": list(concept.modifiers),
        "evidence": list(concept.evidence),
        "dkb_fields": list(concept.dkb_fields),
    }


def model_dict(model: ConceptModel) -> dict[str, Any]:
    """One disease :class:`ConceptModel` as a canonical dict."""
    return {
        "disease_id": model.disease_id,
        "crop": model.crop,
        "condition_type": model.condition_type,
        "sign_type": model.sign_type,
        "is_pathogen_disease": model.is_pathogen_disease,
        "agent_category": model.agent_category,
        "register_policy": {k: model.register_policy[k] for k in sorted(model.register_policy)},
        "mandatory": list(model.mandatory),
        "optional": list(model.optional),
        "forbidden": list(model.forbidden),
        "ordering": list(model.ordering),
        "min_information": model.min_information,
        "max_information": model.max_information,
        "never_appear": list(model.never_appear),
        "concepts": [concept_dict(c) for c in sorted(model.concepts, key=lambda c: c.concept_id)],
    }


def _sorted_models(result: ConceptModelSet) -> list[dict[str, Any]]:
    return [model_dict(m) for m in sorted(result.disease_models, key=lambda m: m.disease_id)]


def concept_models_document(result: ConceptModelSet) -> dict[str, Any]:
    """The complete ``concept_models.json`` document."""
    return {
        "kind": "plantdx.concepts",
        "schema_version": policies.SCHEMA_VERSION,
        "concepts_version": policies.CONCEPTS_VERSION,
        "provenance": result.provenance,
        "models": _sorted_models(result),
    }


def semantic_content(result: ConceptModelSet) -> dict[str, Any]:
    """The full semantic content excluding provenance (basis of the content hash)."""
    return {
        "schema_version": policies.SCHEMA_VERSION,
        "concepts_version": policies.CONCEPTS_VERSION,
        "models": _sorted_models(result),
    }
