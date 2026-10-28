"""Vocabulary Builder — component (B) (doc 01 §7, doc 03 §1).

Derives per-disease closed vocabularies (``allowed_terms``, ``never_appear``),
the extent/stage severity split, and the modifier axes from the DKB and the
authored static vocabulary assets. Deterministic; rebuilt when the DKB changes.
"""

from __future__ import annotations

from plantdx.knowledge_base.models import KnowledgeBase
from plantdx.ontology.models import Ontology
from plantdx.vocabulary.models import VocabularyBundle


class VocabularyBuilder:
    """Builds the :class:`VocabularyBundle` from the DKB, ontology, and assets.

    Args:
        knowledge_base: The loaded DKB.
        ontology: The derived ontology (provides per-disease vocab axes).
        assets: Mapping of asset name to file path (synonyms, modifiers, …).
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        ontology: Ontology,
        assets: dict[str, str],
    ) -> None:
        """Initialize the builder with the DKB, ontology, and asset file paths."""
        self.kb = knowledge_base
        self.ontology = ontology
        self.assets = assets

    def build(self) -> VocabularyBundle:
        """Derive and persist all vocabulary artifacts.

        Raises:
            plantdx.core.exceptions.DerivationError: On a derivation fault
                (e.g., a synonym-class member appearing in a disease's ``never_appear``).
        """
        raise NotImplementedError("Milestone 2: vocabulary derivation")
