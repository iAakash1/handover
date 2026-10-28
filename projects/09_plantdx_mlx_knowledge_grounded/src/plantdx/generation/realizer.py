"""Slot Realizer — component (F) (doc 02 §1).

Fills a template's slots from selected+expanded concepts, then repairs the
surface: article selection (a/an), singular/plural agreement with the extent
quantifier, Oxford-comma list assembly, capitalization, terminal punctuation,
and optional-slot deletion repair (every template must stay grammatical when an
optional slot is dropped).
"""

from __future__ import annotations

from plantdx.core.types import SelectedConcepts, VocabChoice
from plantdx.generation.models import Template
from plantdx.ontology.models import DiseaseOntology
from plantdx.vocabulary.expander import VocabularyExpander


class SlotRealizer:
    """Realizes a template into caption text.

    Args:
        expander: The vocabulary expander applied to slot fillers.
    """

    def __init__(self, expander: VocabularyExpander) -> None:
        """Initialize the realizer with the vocabulary expander."""
        self.expander = expander

    def realize(
        self,
        template: Template,
        concepts: SelectedConcepts,
        ontology: DiseaseOntology,
        seed: str,
    ) -> tuple[str, tuple[VocabChoice, ...]]:
        """Return the realized caption text and the vocabulary choices made.

        Raises:
            plantdx.core.exceptions.GenerationError: If a required slot cannot be filled.
        """
        raise NotImplementedError("Milestone 3: slot realization and surface repair")
