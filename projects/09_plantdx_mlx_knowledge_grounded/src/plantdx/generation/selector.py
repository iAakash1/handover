"""Concept Selector — component (D) (doc 00 §4, §7.2).

Selects the concept set for one caption: required concepts plus a budgeted,
coverage-guided, salience-weighted sample of optional concepts, subject to the
co-selection constraints and the instruction task's response constraint.
"""

from __future__ import annotations

from plantdx.core.types import SelectedConcepts
from plantdx.generation.models import CaptionRequest
from plantdx.ontology.models import DiseaseOntology


class ConceptSelector:
    """Chooses which concepts a caption will assert.

    Args:
        epsilon: Coverage-sampler exploration rate (``configs/generation.yaml``).
    """

    def __init__(self, epsilon: float = 0.30) -> None:
        """Initialize the selector with the coverage-sampler exploration rate."""
        self.epsilon = epsilon

    def select(
        self,
        ontology: DiseaseOntology,
        request: CaptionRequest,
        seed: str,
    ) -> SelectedConcepts:
        """Return the concept set (and realizations) for one caption.

        Enforces ``required ⊆ C ⊆ required union optional``, the information budget,
        co-selection constraints, and the task-type response mask (doc 04 §4.2).
        """
        raise NotImplementedError("Milestone 3: coverage-guided concept selection")

    def reset_coverage(self, disease_id: str) -> None:
        """Reset the per-disease coverage table (start of a generation run)."""
        raise NotImplementedError("Milestone 3: coverage bookkeeping")
