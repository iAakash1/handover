"""Ontology Builder — component (A) (doc 01).

Derives the per-disease caption ontology from the DKB by applying the
derivation-rules table (doc 01 §3.2). This is the mechanism that keeps the DKB
the single source of truth: no disease fact is re-authored, only reshaped.

Milestone 2 implements the derivation. Milestone 1 fixes the public interface.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.knowledge_base.models import KnowledgeBase
from plantdx.ontology.models import ConceptSchema, DiseaseOntology, Ontology


class OntologyBuilder:
    """Builds :class:`Ontology` from the DKB + global concept schema + overrides.

    Args:
        knowledge_base: The loaded, verified DKB.
        concept_schema: The global concept registry.
        overrides_dir: Directory of optional per-disease override files (doc 01 §6).
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        concept_schema: ConceptSchema,
        overrides_dir: str | Path | None = None,
    ) -> None:
        """Initialize the builder with the DKB, concept schema, and optional overrides."""
        self.kb = knowledge_base
        self.concept_schema = concept_schema
        self.overrides_dir = Path(overrides_dir) if overrides_dir else None

    def build(self) -> Ontology:
        """Derive the full ontology for all diseases and persist it.

        Applies the doc-01 §3.2 derivation rules per disease, merges any thin
        overrides (doc 01 §6), then runs the build-time self-checks (doc 01 §8).

        Raises:
            plantdx.core.exceptions.DerivationError: On a derivation/consistency fault.
        """
        raise NotImplementedError("Milestone 2: DKB -> ontology derivation")

    def build_disease(self, disease_id: str) -> DiseaseOntology:
        """Derive the ontology record for a single disease."""
        raise NotImplementedError("Milestone 2: per-disease ontology derivation")

    def self_check(self, ontology: Ontology) -> None:
        """Run build-time invariant checks (doc 01 §8).

        Raises:
            plantdx.core.exceptions.DerivationError: If any check fails
                (e.g., a ``never_appear`` term also in ``recommended_*``, or a
                vocab value not traceable to a DKB field).
        """
        raise NotImplementedError("Milestone 2: ontology self-checks")
