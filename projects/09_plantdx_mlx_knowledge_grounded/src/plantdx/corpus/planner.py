"""Sentence Planner (component D->F, structural).

Consumes a disease :class:`ConceptModel`, a :class:`Template`, and a chosen
concept subset; produces a :class:`SentencePlan` — the ordered, still-structural
filling of the template's slots with controlled realizations. It does NOT emit
English (that is the Caption Generator's job). Every realization is chosen
deterministically from the concept model's controlled phrase set; nothing is
invented. Required slots must be fillable (the corpus builder guarantees this via
template compatibility); an unfillable required slot is a hard error.
"""

from __future__ import annotations

from plantdx.concepts.models import CaptionConcept, ConceptModel
from plantdx.corpus.models import (
    PIECE_CONCEPT,
    PIECE_LIST,
    PIECE_LIT,
    PlanPiece,
    SentencePlan,
)
from plantdx.corpus.seeds import choice_index
from plantdx.templates.models import SEG_LIST, SEG_LIT, SEG_OPT, SEG_SLOT, Template


def plan_caption(
    model: ConceptModel, template: Template, selected: frozenset[str], variant: str
) -> SentencePlan:
    """Build one sentence plan for ``template`` filled with ``selected`` concepts.

    ``selected`` is the concept-id subset to realize (already includes every
    required slot). ``variant`` is a stable string that diversifies realization
    choices across captions of the same (disease, template).
    """
    concepts = {c.concept_id: c for c in model.concepts}
    pieces: list[PlanPiece] = []
    asserted: set[str] = set()

    # Running record of the content already stated, so an optional concept that is
    # already conveyed is dropped: a quality/location modifier contained in the
    # primary sign ("black coating ... black", "... on the lamina ... on the
    # lamina"), or an agent reference that restates the disease name (viruses:
    # "tomato mosaic virus (tomato mosaic virus (ToMV))"). Deterministic.
    emitted = _Emitted()

    for seg in template.segments:
        if seg.kind == SEG_LIT:
            pieces.append(PlanPiece(kind=PIECE_LIT, text=seg.text))
        elif seg.kind == SEG_SLOT:
            phrase = _realize(concepts[seg.concept], model, template, variant)
            pieces.append(PlanPiece(kind=PIECE_CONCEPT, concept=seg.concept, phrase=phrase))
            asserted.add(seg.concept)
            emitted.add(phrase)
        elif seg.kind == SEG_OPT:
            if seg.concept in selected:
                phrase = _realize(concepts[seg.concept], model, template, variant)
                if not emitted.redundant(phrase):
                    pieces.append(
                        PlanPiece(
                            kind=PIECE_CONCEPT,
                            concept=seg.concept,
                            phrase=phrase,
                            glue=seg.glue,
                            suffix=seg.suffix,
                        )
                    )
                    asserted.add(seg.concept)
                    emitted.add(phrase)
        elif seg.kind == SEG_LIST:
            items: list[tuple[str, str]] = []
            for c in seg.concepts:
                if c not in selected:
                    continue
                phrase = _realize(concepts[c], model, template, variant)
                if not emitted.redundant(phrase):
                    items.append((c, phrase))
                    emitted.add(phrase)
            if items:
                pieces.append(
                    PlanPiece(kind=PIECE_LIST, items=tuple(items), glue=seg.glue, conj=seg.conj)
                )
                asserted.update(c for c, _ in items)

    return SentencePlan(
        disease_id=model.disease_id,
        template_id=template.id,
        family=template.family,
        register=template.register,
        hedged=template.hedged,
        asserted_concepts=tuple(sorted(asserted)),
        pieces=tuple(pieces),
    )


_OVERLAP_THRESHOLD = 0.6


class _Emitted:
    """Accumulates the content already stated in a caption to suppress redundancy."""

    def __init__(self) -> None:
        self.words: set[str] = set()
        self.text = ""

    def add(self, phrase: str) -> None:
        self.text += " " + phrase.lower()
        self.words |= set(phrase.lower().split())

    def redundant(self, phrase: str) -> bool:
        """Whether ``phrase`` is already conveyed by content emitted so far.

        A single-word modifier is redundant if that word already appears ("black"
        when "black sooty coating" was stated); a multi-word phrase is redundant if
        it is a substring of what was said, or if most of its content words were
        already stated (an agent reference that restates the disease name).
        """
        if not self.text:
            return False
        low = phrase.lower().strip()
        words = low.split()
        if len(words) == 1:
            return words[0] in self.words
        if low in self.text:
            return True
        overlap = sum(1 for w in words if w in self.words) / len(words)
        return overlap >= _OVERLAP_THRESHOLD


def _realize(concept: CaptionConcept, model: ConceptModel, template: Template, variant: str) -> str:
    """Deterministically pick one controlled realization for a concept."""
    options = concept.realizations
    if not options:
        raise ValueError(
            f"{model.disease_id}/{template.id}: concept {concept.concept_id} has no realization"
        )
    idx = choice_index(len(options), model.disease_id, template.id, concept.concept_id, variant)
    return options[idx]
