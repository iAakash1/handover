"""Caption Validator (component G). Fail closed; never trusts the generator.

Every check re-derives its expectation from the concept model and template and
compares it to the finished caption — a hallucination that slips one check is
caught by another (defense in depth, doc 03). The battery is per-caption
(``V-CAP-1..12``); corpus-wide duplicate detection lives in the Corpus Builder
(``V-COR-*``). See ``docs/CORPUS.md`` for what each rule guards against.
"""

from __future__ import annotations

import re

from plantdx.concepts import policies as concept_policies
from plantdx.concepts.models import STATUS_FORBIDDEN, CaptionConcept, ConceptModel
from plantdx.corpus.models import Caption
from plantdx.templates import compatible
from plantdx.templates.models import Template

CHECK_COUNT = 12

_CONFIDENCE_RANK = {"hedged": 1, "typical": 2, "asserted": 3}
_MODIFIER_CONCEPTS = concept_policies.MODIFIER_CONCEPTS


def validate_caption(caption: Caption, model: ConceptModel, template: Template) -> list[str]:
    """Run the 12-check per-caption battery; return the sorted violation list."""
    v: list[str] = []
    v += _v1_ontology_legality(caption, model)
    v += _v2_template_legality(caption, model, template)
    v += _v3_mandatory_present(caption, model)
    v += _v4_forbidden_absent(caption, model)
    v += _v5_never_appear(caption, model)
    v += _v6_duplicate_wording(caption)
    v += _v7_grammar(caption)
    v += _v8_modifier_legality(caption, model)
    v += _v9_observability(caption, model)
    v += _v10_confidence(caption, model)
    v += _v11_severity(caption)
    v += _v12_traceability(caption, model)
    return sorted(set(v))


def _available(model: ConceptModel) -> set[str]:
    return set(model.mandatory) | set(model.optional)


def _concept_map(model: ConceptModel) -> dict[str, CaptionConcept]:
    return {c.concept_id: c for c in model.concepts}


def _v1_ontology_legality(caption: Caption, model: ConceptModel) -> list[str]:
    """Asserted concepts are all available and include at least one defining concept.

    ``model.mandatory`` is the disease's defining/anchor set (disease_identity +
    primary_sign, or healthy_state). Per doc 02 the template library includes
    minimalist sign-only and name-only styles, so a caption need only assert *at
    least one* anchor — not every anchor — but never a non-available concept.
    """
    v: list[str] = []
    asserted = set(caption.concepts)
    illegal = asserted - _available(model)
    if illegal:
        v.append(f"V-CAP-1: {caption.caption_id} asserts non-available concepts {sorted(illegal)}")
    if not asserted & set(model.mandatory):
        v.append(f"V-CAP-1: {caption.caption_id} asserts no defining concept")
    return v


def _v2_template_legality(caption: Caption, model: ConceptModel, template: Template) -> list[str]:
    v: list[str] = []
    if caption.template_id != template.id:
        v.append(f"V-CAP-2: {caption.caption_id} template id mismatch")
    if not compatible(template, model):
        v.append(f"V-CAP-2: {caption.caption_id} uses template incompatible with disease")
    if caption.family != template.family:
        v.append(f"V-CAP-2: {caption.caption_id} family != template family")
    if not set(template.required) <= set(caption.concepts):
        v.append(f"V-CAP-2: {caption.caption_id} does not assert all required slots")
    return v


def _v3_mandatory_present(caption: Caption, model: ConceptModel) -> list[str]:
    """Each mandatory concept's surface realization must actually appear in the text."""
    v: list[str] = []
    concepts = _concept_map(model)
    low = caption.text.lower()
    asserted = set(caption.concepts)
    # Only the defining concepts the caption actually asserts must be textually
    # present (a sign-only caption asserts primary_sign but not disease_identity).
    for cid in set(model.mandatory) & asserted:
        concept = concepts.get(cid)
        if concept is None:
            continue
        if not any(r.lower() in low for r in concept.realizations):
            v.append(f"V-CAP-3: {caption.caption_id} asserted {cid} not present in text")
    return v


def _v4_forbidden_absent(caption: Caption, model: ConceptModel) -> list[str]:
    forbidden = set(model.forbidden)
    hit = set(caption.concepts) & forbidden
    if hit:
        return [f"V-CAP-4: {caption.caption_id} asserts forbidden concepts {sorted(hit)}"]
    return []


def _v5_never_appear(caption: Caption, model: ConceptModel) -> list[str]:
    v: list[str] = []
    low = caption.text.lower()
    for term in model.never_appear:
        if _word_present(term.lower(), low):
            v.append(f"V-CAP-5: {caption.caption_id} contains forbidden term {term!r}")
    return v


def _v6_duplicate_wording(caption: Caption) -> list[str]:
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", caption.text) if s.strip()]
    if len(sentences) != len(set(sentences)):
        return [f"V-CAP-6: {caption.caption_id} repeats an identical sentence"]
    return []


def _v7_grammar(caption: Caption) -> list[str]:
    v: list[str] = []
    text = caption.text
    if not text:
        v.append(f"V-CAP-7: {caption.caption_id} empty caption")
        return v
    if not text[0].isupper():
        v.append(f"V-CAP-7: {caption.caption_id} does not start with a capital letter")
    if text[-1] not in ".!?":
        v.append(f"V-CAP-7: {caption.caption_id} has no terminal punctuation")
    if "  " in text:
        v.append(f"V-CAP-7: {caption.caption_id} has a double space")
    if re.search(r"\s[,.;:]", text):
        v.append(f"V-CAP-7: {caption.caption_id} has a space before punctuation")
    if text.count("(") != text.count(")"):
        v.append(f"V-CAP-7: {caption.caption_id} has unbalanced parentheses")
    if re.search(r"[,;:]{2,}|\.{2,}", text):
        v.append(f"V-CAP-7: {caption.caption_id} has repeated punctuation")
    return v


def _v8_modifier_legality(caption: Caption, model: ConceptModel) -> list[str]:
    used = set(caption.concepts) & _MODIFIER_CONCEPTS
    if used and model.sign_type not in concept_policies.MODIFIABLE_SIGN_TYPES:
        return [
            f"V-CAP-8: {caption.caption_id} uses quality concepts {sorted(used)} "
            f"but sign type {model.sign_type!r} is not modifiable"
        ]
    return []


def _v9_observability(caption: Caption, model: ConceptModel) -> list[str]:
    """A visual-register caption may not assert a non-observable concept."""
    if caption.register != "visual":
        return []
    hidden = set(caption.concepts) & concept_policies.NON_OBSERVABLE_CONCEPTS
    if hidden:
        return [
            f"V-CAP-9: {caption.caption_id} visual caption asserts non-observable {sorted(hidden)}"
        ]
    return []


def _v10_confidence(caption: Caption, model: ConceptModel) -> list[str]:
    concepts = _concept_map(model)
    ranks = [_CONFIDENCE_RANK[concepts[c].confidence] for c in caption.concepts if c in concepts]
    expected = min(ranks) if ranks else _CONFIDENCE_RANK["asserted"]
    actual = _CONFIDENCE_RANK.get(caption.confidence)
    if actual != expected:
        return [f"V-CAP-10: {caption.caption_id} confidence {caption.confidence!r} != weakest link"]
    return []


def _v11_severity(caption: Caption) -> list[str]:
    low = caption.text.lower()
    for token in concept_policies.STAGE_TOKENS:
        if _word_present(token, low):
            return [f"V-CAP-11: {caption.caption_id} asserts severity stage {token!r}"]
    return []


def _v12_traceability(caption: Caption, model: ConceptModel) -> list[str]:
    """Recompute evidence + observability from the model and compare to the caption."""
    v: list[str] = []
    concepts = _concept_map(model)
    evidence: set[str] = set()
    observable = True
    for cid in caption.concepts:
        concept = concepts.get(cid)
        if concept is None or concept.status == STATUS_FORBIDDEN:
            continue
        evidence.update(concept.evidence)
        observable = observable and concept.observable
    if tuple(sorted(evidence)) != caption.evidence:
        v.append(f"V-CAP-12: {caption.caption_id} evidence does not match its concepts")
    if observable != caption.observable:
        v.append(f"V-CAP-12: {caption.caption_id} observable flag does not match its concepts")
    return v


def _word_present(term: str, text_lower: str) -> bool:
    """Word-boundary, case-insensitive containment (term already lowercased)."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text_lower) is not None
