"""Content-only checksum for the caption corpus (determinism guarantee).

Computed over caption *content* only (via the canonical serialization) — never
timestamps, output paths, machine, OS, or the upstream source-checksum pins. Two
runs on the same concept models + templates produce an identical corpus hash.
"""

from __future__ import annotations

from plantdx.corpus import serialization
from plantdx.corpus.models import Corpus
from plantdx.utils.hashing import sha256_bytes


def content_hash(corpus: Corpus) -> str:
    """``sha256:<hex>`` over the full caption content (excludes provenance)."""
    payload = serialization.canonical_json(serialization.semantic_content(corpus))
    return "sha256:" + sha256_bytes(payload.encode("utf-8"))
