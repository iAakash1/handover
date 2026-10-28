"""Content-only checksum for the template library (determinism guarantee)."""

from __future__ import annotations

from plantdx.templates import serialization
from plantdx.templates.models import TemplateLibrary
from plantdx.utils.hashing import sha256_bytes


def content_hash(library: TemplateLibrary) -> str:
    """``sha256:<hex>`` over the full template content (excludes provenance)."""
    payload = serialization.canonical_json(serialization.semantic_content(library))
    return "sha256:" + sha256_bytes(payload.encode("utf-8"))
