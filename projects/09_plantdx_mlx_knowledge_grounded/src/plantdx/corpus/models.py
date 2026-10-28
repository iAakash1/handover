"""Value objects for the caption corpus layer.

Three stages, three shapes: the Sentence Planner emits a :class:`SentencePlan`
(structured, ordered slot fillings — NOT English); the Caption Generator turns a
plan into a :class:`Caption` (English + traceable metadata); the Corpus Builder
assembles a :class:`Corpus`. Plain dataclasses, no behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PIECE_LIT = "lit"
PIECE_CONCEPT = "concept"
PIECE_LIST = "list"


@dataclass(frozen=True)
class PlanPiece:
    """One resolved piece of a sentence plan (still structured, not a string)."""

    kind: str  # lit | concept | list
    text: str = ""  # lit
    concept: str = ""  # concept id (concept kind)
    phrase: str = ""  # the chosen controlled realization (concept kind)
    glue: str = ""  # concept | list prefix
    suffix: str = ""  # concept suffix (paired delimiters)
    conj: str = "and"  # list conjunction
    items: tuple[tuple[str, str], ...] = ()  # list: (concept_id, phrase) pairs


@dataclass(frozen=True)
class SentencePlan:
    """The ordered, still-structural plan for one caption (Sentence Planner output)."""

    disease_id: str
    template_id: str
    family: str
    register: str
    hedged: bool
    asserted_concepts: tuple[str, ...]  # concept ids actually realized (sorted)
    pieces: tuple[PlanPiece, ...]


@dataclass(frozen=True)
class Caption:
    """One finished, validated caption with full traceability metadata."""

    caption_id: str
    disease_id: str
    crop: str
    condition_type: str
    template_id: str
    family: str
    register: str
    hedged: bool
    confidence: str  # weakest-link confidence across asserted concepts
    observable: bool  # every asserted concept is leaf-observable
    text: str
    concepts: tuple[str, ...]  # asserted concept ids (sorted)
    evidence: tuple[str, ...]  # union of asserted concepts' evidence (sorted)
    language: str = "en"


@dataclass
class Corpus:
    """The complete caption corpus plus provenance."""

    captions: list[Caption]
    provenance: dict[str, str] = field(default_factory=dict)
