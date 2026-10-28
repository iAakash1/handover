"""Content-only checksums for the ontology (determinism guarantee).

The checksum depends solely on ontology content (concept types, relation types,
nodes, edges) via the canonical serialization — never on timestamps, output
location, machine, OS, or hash randomization.
"""

from __future__ import annotations

from plantdx.ontology.domain import serialization
from plantdx.ontology.domain.models import Ontology
from plantdx.utils.hashing import sha256_bytes


def content_hash(ontology: Ontology) -> str:
    """``sha256:<hex>`` over the full semantic content (excludes provenance)."""
    payload = serialization.canonical_json(serialization.semantic_content(ontology))
    return "sha256:" + sha256_bytes(payload.encode("utf-8"))


def schema_hash(ontology: Ontology) -> str:
    """``sha256:<hex>`` over the T-Box only (concept + relation types)."""
    payload = serialization.canonical_json(serialization.schema_content(ontology))
    return "sha256:" + sha256_bytes(payload.encode("utf-8"))
