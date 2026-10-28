"""Value objects for the vocabulary + symptom lexicon compiler.

One shared dataclass, :class:`LexicalItem`, is used for both artifacts
(``vocabulary.json`` and ``symptom_lexicon.json``) since both are the same kind
of thing: a lexical surface form with full traceability back to an ontology
node, and through it to the DKB and its evidence. Plain dataclasses, no
behavior. Determinism is the caller's responsibility (everything is sorted
before output), exactly as in ``plantdx.ontology.domain.models``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LexicalItem:
    """One vocabulary or symptom-lexicon entry, fully traceable to the ontology.

    Traceability chain: ``ontology_node`` (+ ``source`` relation) -> the DKB
    disease(s) named in ``dkb_reference`` -> the citations named in ``evidence``.
    """

    id: str
    surface_form: str
    canonical_form: str
    concept: str  # human-readable category, e.g. "color", "symptom_realization"
    concept_id: str  # the ontology ConceptType.id this traces to, e.g. "Color"
    confidence: str  # "asserted" | "typical" | "hedged"
    source: str  # the ontology relation type this item was derived through
    ontology_node: str  # the grounding ontology node id
    dkb_reference: tuple[str, ...]  # disease_id(s) (Condition nodes) that use this node
    evidence: tuple[str, ...]  # evidence node ids (empty for T-Box/closed-vocab items)
    language: str = "en"
    part_of_speech: str = "noun"


@dataclass
class VocabularyResult:
    """The complete build output: both artifacts' items plus provenance."""

    vocabulary_items: list[LexicalItem]
    lexicon_items: list[LexicalItem]
    provenance: dict[str, str] = field(default_factory=dict)
