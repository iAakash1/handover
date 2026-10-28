"""Canonical serialization of the caption corpus (JSON / JSONL / CSV).

Reuses :func:`plantdx.ontology.domain.serialization.canonical_json`. Every
caption record carries the full source-checksum pin (ontology, vocabulary,
concepts, templates) so no provenance is lost — but the corpus *content hash* is
computed over caption content only, so upstream provenance never perturbs it.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from plantdx.corpus.models import Caption, Corpus
from plantdx.ontology.domain.serialization import canonical_json

__all__ = [
    "canonical_json",
    "captions_document",
    "csv_text",
    "jsonl_text",
    "record",
    "semantic_content",
    "semantic_record",
]

_SOURCE_KEYS = (
    "ontology_content_hash",
    "vocabulary_content_hash",
    "concepts_content_hash",
    "template_content_hash",
)


def semantic_record(c: Caption) -> dict[str, Any]:
    """The content fields of one caption (basis of the corpus content hash)."""
    return {
        "caption_id": c.caption_id,
        "concept_model_id": c.disease_id,
        "condition": c.disease_id,
        "crop": c.crop,
        "condition_type": c.condition_type,
        "template_id": c.template_id,
        "family": c.family,
        "register": c.register,
        "hedged": c.hedged,
        "confidence": c.confidence,
        "observable": c.observable,
        "text": c.text,
        "concepts": list(c.concepts),
        "evidence": list(c.evidence),
        "language": c.language,
    }


def record(c: Caption, source: dict[str, str]) -> dict[str, Any]:
    """A full caption record: content + the source-checksum pin (no metadata lost)."""
    doc = semantic_record(c)
    doc["source"] = {k: source.get(k, "") for k in _SOURCE_KEYS}
    return doc


def _sorted_captions(corpus: Corpus) -> list[Caption]:
    return sorted(corpus.captions, key=lambda c: (c.disease_id, c.template_id, c.caption_id))


def semantic_content(corpus: Corpus) -> dict[str, Any]:
    """The full semantic content excluding provenance (basis of the content hash)."""
    return {
        "schema_version": "1.0.0",
        "captions": [semantic_record(c) for c in _sorted_captions(corpus)],
    }


def captions_document(corpus: Corpus) -> dict[str, Any]:
    """The complete ``captions.json`` document."""
    source = {k: corpus.provenance.get(k, "") for k in _SOURCE_KEYS}
    return {
        "kind": "plantdx.corpus",
        "schema_version": "1.0.0",
        "provenance": corpus.provenance,
        "captions": [record(c, source) for c in _sorted_captions(corpus)],
    }


def jsonl_text(corpus: Corpus) -> str:
    """One JSON caption record per line (``captions.jsonl``), deterministically ordered."""
    import json

    source = {k: corpus.provenance.get(k, "") for k in _SOURCE_KEYS}
    lines = [
        json.dumps(record(c, source), sort_keys=True, ensure_ascii=False)
        for c in _sorted_captions(corpus)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def csv_text(corpus: Corpus) -> str:
    """A flat CSV view (``captions.csv``); list fields are ``|``-joined."""
    source = {k: corpus.provenance.get(k, "") for k in _SOURCE_KEYS}
    columns = [
        "caption_id",
        "concept_model_id",
        "crop",
        "condition_type",
        "template_id",
        "family",
        "register",
        "hedged",
        "confidence",
        "observable",
        "language",
        "concepts",
        "evidence",
        "text",
        *_SOURCE_KEYS,
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for c in _sorted_captions(corpus):
        writer.writerow(
            [
                c.caption_id,
                c.disease_id,
                c.crop,
                c.condition_type,
                c.template_id,
                c.family,
                c.register,
                c.hedged,
                c.confidence,
                c.observable,
                c.language,
                "|".join(c.concepts),
                "|".join(c.evidence),
                c.text,
                *[source.get(k, "") for k in _SOURCE_KEYS],
            ]
        )
    return buffer.getvalue()
