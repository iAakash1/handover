"""Loader for the global concept registry (``concept_schema.json``).

The concept schema is disease-independent (doc 01 §2). It is authored/derived
once and consumed by the ontology builder and the concept selector.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.ontology.models import ConceptSchema


class ConceptSchemaLoader:
    """Loads the global :class:`ConceptSchema` from disk."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the loader with the path to the concept schema file."""
        self.path = Path(path)

    def load(self) -> ConceptSchema:
        """Parse and return the concept registry and co-selection constraints."""
        raise NotImplementedError("Milestone 2: concept schema loading")
