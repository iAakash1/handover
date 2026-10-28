"""Symptom Lexicon Builder — component (C) (doc 03 §1).

Compiles the surface-phrase -> concept lexicon used by the observability and
cross-disease-leakage validators (V2, V8). Includes forbidden-symptom surfaces
(fruit, twig, gummosis, tear-stain, pycnidia, star-shaped, …) drawn from every
disease's ``forbidden_symptoms_not_leaf_observable`` and rivals' hallmark terms.
"""

from __future__ import annotations

from plantdx.knowledge_base.models import KnowledgeBase
from plantdx.ontology.models import Ontology
from plantdx.vocabulary.models import SymptomLexicon


class SymptomLexiconBuilder:
    """Builds the :class:`SymptomLexicon` from the DKB and derived ontology.

    Args:
        knowledge_base: The loaded DKB.
        ontology: The derived ontology (provides required medical terminology).
    """

    def __init__(self, knowledge_base: KnowledgeBase, ontology: Ontology) -> None:
        """Initialize the builder with the DKB and derived ontology."""
        self.kb = knowledge_base
        self.ontology = ontology

    def build(self) -> SymptomLexicon:
        """Compile and persist the symptom lexicon.

        Raises:
            plantdx.core.exceptions.DerivationError: On an inconsistent mapping.
        """
        raise NotImplementedError("Milestone 2: symptom lexicon compilation")
