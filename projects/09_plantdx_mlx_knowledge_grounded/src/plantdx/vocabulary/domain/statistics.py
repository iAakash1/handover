"""Deterministic vocabulary + lexicon statistics (``statistics.json``)."""

from __future__ import annotations

from typing import Any

from plantdx.vocabulary.domain.models import VocabularyResult


def _by_field(items: list[Any], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = getattr(item, field)
        counts[key] = counts.get(key, 0) + 1
    return {k: counts[k] for k in sorted(counts)}


def compute(result: VocabularyResult, validation_status: str) -> dict[str, Any]:
    """Return the statistics document (all values deterministic)."""
    vocab, lexicon = result.vocabulary_items, result.lexicon_items
    base_count = sum(1 for i in lexicon if ":base" in i.id)
    modifier_count = len(lexicon) - base_count

    return {
        "vocabulary_version": result.provenance.get("vocabulary_version", ""),
        "build_checksum": result.provenance.get("content_hash", ""),
        "validation_status": validation_status,
        "vocabulary_item_count": len(vocab),
        "lexicon_item_count": len(lexicon),
        "total_item_count": len(vocab) + len(lexicon),
        "vocabulary_by_concept": _by_field(vocab, "concept"),
        "vocabulary_by_confidence": _by_field(vocab, "confidence"),
        "lexicon_by_confidence": _by_field(lexicon, "confidence"),
        "lexicon_base_realizations": base_count,
        "lexicon_modifier_realizations": modifier_count,
        "unique_ontology_nodes_covered": len({i.ontology_node for i in vocab if i.ontology_node}),
        "unique_diseases_referenced": len({d for i in (vocab + lexicon) for d in i.dkb_reference}),
    }
