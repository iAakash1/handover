"""Vocabulary Expander — part of component (F) (doc 01 §7.3).

Produces lexical diversity by walking the typed, meaning-preserving expansion
lattice (synonym substitution + modifier attachment) under strict no-drift
constraints:

    * every modifier value comes from the disease's DKB-derived ``vocab_axes``;
    * synonym substitution stays within one equivalence class;
    * the asserted concept set is invariant — expansion realizes already-selected
      concepts more richly and NEVER introduces an unselected concept.

Every expansion step is recorded as a :class:`plantdx.core.types.ExpansionEdge`
so the result is replayable and auditable (doc 04 §3).
"""

from __future__ import annotations

from plantdx.core.types import ExpansionEdge
from plantdx.ontology.models import DiseaseOntology
from plantdx.vocabulary.models import VocabularyBundle


class VocabularyExpander:
    """Expands a realized phrase into a varied but semantically identical form.

    Args:
        vocabulary: The compiled vocabulary bundle (synonym classes, axes).
        max_adjectives: Modifier-stack depth cap (``configs/generation.yaml``).
    """

    def __init__(self, vocabulary: VocabularyBundle, max_adjectives: int = 3) -> None:
        """Initialize the expander with the vocabulary bundle and modifier-depth cap."""
        self.vocabulary = vocabulary
        self.max_adjectives = max_adjectives

    def expand(
        self,
        phrase: str,
        ontology: DiseaseOntology,
        seed: str,
    ) -> tuple[str, tuple[ExpansionEdge, ...]]:
        """Return an expanded phrase and the ordered edges that produced it.

        The returned phrase asserts exactly the same concept set as ``phrase``.

        Raises:
            plantdx.core.exceptions.InvariantViolation: If an expansion would
                introduce an out-of-axis modifier or cross a synonym class.
        """
        raise NotImplementedError("Milestone 3: constrained vocabulary expansion")
