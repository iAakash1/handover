"""Artifact versioning and manifests (interface) (doc 00 §6, doc 06 §5).

Computes ``library_version`` and writes/reads the artifact manifest that pins
``dkb_sha256``, ``ontology_build_id``, ``template_set_version``,
``vocabulary_version``, ``config_hash``, and ``generator_version``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Version pins for a generated artifact family (doc 06 §5)."""

    library_version: str
    dkb_sha256: str
    ontology_build_id: str
    template_set_version: str
    vocabulary_version: str
    config_hash: str
    generator_version: str


def compute_library_version(config_hash: str, dkb_sha256: str) -> str:
    """Derive the ``L{n}`` library-version string embedding a short config hash."""
    raise NotImplementedError("Milestone 2: library version derivation")


def write_manifest(path: str | Path, manifest: ArtifactManifest) -> None:
    """Write an artifact manifest JSON."""
    raise NotImplementedError("Milestone 2: manifest write")


def read_manifest(path: str | Path) -> ArtifactManifest:
    """Read an artifact manifest JSON."""
    raise NotImplementedError("Milestone 2: manifest read")
