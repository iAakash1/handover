"""Vocabulary and lexicon data models (doc 01 §7, doc 03 §1)."""

from __future__ import annotations

from dataclasses import dataclass

from plantdx.core.enums import ConceptId, SignType


@dataclass(frozen=True, slots=True)
class SynonymClass:
    """A synonym equivalence class tagged with its axis (doc 01 §7.1)."""

    name: str
    members: tuple[str, ...]
    axis: str
    forbid_for: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModifierAxis:
    """One ordered modifier axis (size/shape/color/texture/extent/location)."""

    axis: str
    order_rank: int


@dataclass(frozen=True, slots=True)
class SymptomLexiconEntry:
    """Maps a surface phrase to a concept, sign type, and owning diseases (doc 03 §1)."""

    surface: str
    concept_id: ConceptId | None
    sign_type: SignType | None
    owner_diseases: tuple[str, ...]
    is_forbidden_structure: bool  # non-leaf structure (fruit/twig/…)


@dataclass(frozen=True, slots=True)
class SymptomLexicon:
    """The compiled symptom lexicon consumed by validators V2/V8."""

    entries: tuple[SymptomLexiconEntry, ...]


@dataclass(frozen=True, slots=True)
class DiseaseVocabulary:
    """Per-disease closed vocabulary artifacts (doc 03 §1)."""

    disease_id: str
    allowed_terms: frozenset[str]
    never_appear: frozenset[str]


@dataclass(frozen=True, slots=True)
class VocabularyBundle:
    """All vocabulary artifacts produced by the builders (B, C)."""

    vocabulary_version: str
    synonym_classes: tuple[SynonymClass, ...]
    modifier_axes: tuple[ModifierAxis, ...]
    symptom_lexicon: SymptomLexicon
    per_disease: dict[str, DiseaseVocabulary]
    stage_terms: frozenset[str]
    function_words: frozenset[str]
