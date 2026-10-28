"""Content-only checksum for the concept models (determinism guarantee).

The checksum depends solely on the semantic content of the models via the
canonical serialization — never on timestamps, output location, machine, OS, or
hash randomization. Running the builder twice on the same inputs must produce an
identical checksum.
"""

from __future__ import annotations

from plantdx.concepts import serialization
from plantdx.concepts.models import ConceptModelSet
from plantdx.utils.hashing import sha256_bytes


def content_hash(result: ConceptModelSet) -> str:
    """``sha256:<hex>`` over the full semantic content (excludes provenance)."""
    payload = serialization.canonical_json(serialization.semantic_content(result))
    return "sha256:" + sha256_bytes(payload.encode("utf-8"))
