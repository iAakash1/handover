"""Canonical, deterministic serialization of vocabulary/lexicon documents.

Every document is stable, pretty-printed, UTF-8, sorted (keys via ``sort_keys``,
lists sorted explicitly). No timestamps, no machine-dependent ordering. Reuses
:func:`plantdx.ontology.domain.serialization.canonical_json` so the two
compilers share one JSON convention.
"""

from __future__ import annotations

from typing import Any

from plantdx.ontology.domain.serialization import canonical_json
from plantdx.vocabulary.domain import policies
from plantdx.vocabulary.domain.models import LexicalItem, VocabularyResult

__all__ = [
    "canonical_json",
    "concept_index_document",
    "item_dict",
    "lexicon_document",
    "semantic_content",
    "vocabulary_document",
]


def item_dict(item: LexicalItem) -> dict[str, Any]:
    """The exact metadata schema mandated for every lexical item."""
    return {
        "id": item.id,
        "surface_form": item.surface_form,
        "canonical_form": item.canonical_form,
        "concept": item.concept,
        "concept_id": item.concept_id,
        "confidence": item.confidence,
        "source": item.source,
        "ontology_node": item.ontology_node,
        "dkb_reference": list(item.dkb_reference),
        "evidence": list(item.evidence),
        "language": item.language,
        "part_of_speech": item.part_of_speech,
    }


def _sorted_items(items: list[LexicalItem]) -> list[dict[str, Any]]:
    return [item_dict(i) for i in sorted(items, key=lambda i: i.id)]


def vocabulary_document(result: VocabularyResult) -> dict[str, Any]:
    """The complete ``vocabulary.json`` document."""
    return {
        "kind": "plantdx.vocabulary",
        "schema_version": policies.SCHEMA_VERSION,
        "vocabulary_version": policies.VOCABULARY_VERSION,
        "provenance": result.provenance,
        "items": _sorted_items(result.vocabulary_items),
    }


def lexicon_document(result: VocabularyResult) -> dict[str, Any]:
    """The complete ``symptom_lexicon.json`` document."""
    return {
        "kind": "plantdx.vocabulary.symptom_lexicon",
        "schema_version": policies.SCHEMA_VERSION,
        "vocabulary_version": policies.VOCABULARY_VERSION,
        "provenance": result.provenance,
        "items": _sorted_items(result.lexicon_items),
    }


def concept_index_document(result: VocabularyResult) -> dict[str, Any]:
    """Lookup indices (``concept_index.json``): by concept, by ontology node, by disease."""
    by_concept: dict[str, list[str]] = {}
    by_ontology_node: dict[str, list[str]] = {}
    by_dkb_reference: dict[str, list[str]] = {}
    for item in result.vocabulary_items + result.lexicon_items:
        by_concept.setdefault(item.concept, []).append(item.id)
        if item.ontology_node:
            by_ontology_node.setdefault(item.ontology_node, []).append(item.id)
        for disease_id in item.dkb_reference:
            by_dkb_reference.setdefault(disease_id, []).append(item.id)
    for index in (by_concept, by_ontology_node, by_dkb_reference):
        for ids in index.values():
            ids.sort()

    return {
        "kind": "plantdx.vocabulary.concept_index",
        "schema_version": policies.SCHEMA_VERSION,
        "vocabulary_version": policies.VOCABULARY_VERSION,
        "provenance": result.provenance,
        "by_concept": {k: by_concept[k] for k in sorted(by_concept)},
        "by_ontology_node": {k: by_ontology_node[k] for k in sorted(by_ontology_node)},
        "by_dkb_reference": {k: by_dkb_reference[k] for k in sorted(by_dkb_reference)},
    }


def semantic_content(result: VocabularyResult) -> dict[str, Any]:
    """The full semantic content excluding provenance (basis of the content hash)."""
    return {
        "schema_version": policies.SCHEMA_VERSION,
        "vocabulary_version": policies.VOCABULARY_VERSION,
        "vocabulary_items": _sorted_items(result.vocabulary_items),
        "lexicon_items": _sorted_items(result.lexicon_items),
    }
