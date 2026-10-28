"""Content-only checksum for the vocabulary + lexicon (determinism guarantee).

The checksum depends solely on the item content of both artifacts, via the
canonical serialization — never on timestamps, output location, machine, OS,
or hash randomization. Running the builder twice on the same ontology must
produce an identical checksum.
"""

from __future__ import annotations

from plantdx.utils.hashing import sha256_bytes
from plantdx.vocabulary.domain import serialization
from plantdx.vocabulary.domain.models import VocabularyResult


def content_hash(result: VocabularyResult) -> str:
    """``sha256:<hex>`` over the full semantic content (excludes provenance)."""
    payload = serialization.canonical_json(serialization.semantic_content(result))
    return "sha256:" + sha256_bytes(payload.encode("utf-8"))
