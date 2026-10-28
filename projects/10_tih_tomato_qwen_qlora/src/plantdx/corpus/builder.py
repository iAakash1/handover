"""Corpus Builder (components D->I, disease-level).

Deterministically enumerates, per disease, a bounded and diverse set of captions
= (compatible template) x (a fixed family of concept subsets). Every candidate is
validated independently by the Caption Validator; a failing candidate is dropped
and recorded (fail closed — no invalid caption enters the corpus), and a disease
that yields zero accepted captions is a hard error (it indicates an ontology /
lexicon / template bug, not a data issue). Accepted captions are de-duplicated by
normalized text and sorted for byte-identical output.
"""

from __future__ import annotations

from plantdx.concepts.models import ConceptModel, ConceptModelSet
from plantdx.core.exceptions import PlantDxError
from plantdx.corpus import validator
from plantdx.corpus.generator import generate
from plantdx.corpus.models import Caption, Corpus
from plantdx.corpus.planner import plan_caption
from plantdx.templates import compatible_templates
from plantdx.templates.checksum import content_hash as template_checksum
from plantdx.templates.models import Template, TemplateLibrary
from plantdx.utils.hashing import sha256_hex

_CONFIDENCE_RANK = {"hedged": 1, "typical": 2, "asserted": 3}
_RANK_CONFIDENCE = {v: k for k, v in _CONFIDENCE_RANK.items()}
_MEDIUM_SUBSET_SIZE = 3
# Realization variants per (template, concept-subset): concepts with multiple
# controlled phrasings yield distinct valid captions; duplicates are dropped. Two
# is the deliberate balance between coverage of sparse diseases and avoiding a
# flood of one-word-swap near-duplicates that would depress lexical diversity.
_REALIZATION_VARIANTS = 2


class CorpusBuildError(PlantDxError):
    """Raised when a disease yields zero valid captions (a structural bug)."""


def build_corpus(
    concept_models: ConceptModelSet,
    library: TemplateLibrary,
    *,
    condition: str | None = None,
    crop: str | None = None,
) -> tuple[Corpus, dict[str, object]]:
    """Build the caption corpus. Returns ``(corpus, validation_report)``."""
    models = _select_models(concept_models, condition, crop)

    captions: list[Caption] = []
    seen: set[str] = set()
    rejected_by_check: dict[str, int] = {}
    accepted_by_disease: dict[str, int] = {}

    for model in models:
        accepted_here = 0
        for template in compatible_templates(library, model):
            for subset_index, subset in enumerate(_selections(model, template)):
                # Try a bounded number of realization variants per subset: concepts
                # with several controlled phrasings yield different (still valid,
                # still traceable) captions, raising diversity and the coverage of
                # sparse diseases. Identical results are dropped by de-duplication,
                # so richer diseases simply produce more unique captions.
                for realization in range(_REALIZATION_VARIANTS):
                    caption = _assemble(model, template, subset, f"{subset_index}:{realization}")
                    violations = validator.validate_caption(caption, model, template)
                    if violations:
                        for check in {v.split(":", 1)[0] for v in violations}:
                            rejected_by_check[check] = rejected_by_check.get(check, 0) + 1
                        continue
                    norm = " ".join(caption.text.lower().split())
                    if norm in seen:
                        continue
                    seen.add(norm)
                    captions.append(caption)
                    accepted_here += 1
        if accepted_here == 0:
            raise CorpusBuildError(f"disease {model.disease_id} produced zero valid captions")
        accepted_by_disease[model.disease_id] = accepted_here

    captions.sort(key=lambda c: (c.disease_id, c.template_id, c.caption_id))
    corpus = Corpus(
        captions=captions,
        provenance={
            "ontology_content_hash": concept_models.provenance.get("ontology_content_hash", ""),
            "vocabulary_content_hash": concept_models.provenance.get("vocabulary_content_hash", ""),
            "concepts_content_hash": concept_models.provenance.get("content_hash", ""),
            "template_content_hash": template_checksum(library),
            "template_set_version": library.template_set_version,
            "builder": "plantdx.corpus",
        },
    )
    report = {
        "kind": "plantdx.corpus.validation_report",
        "status": "valid",
        "checks_run": validator.CHECK_COUNT,
        "accepted": len(captions),
        "rejected_by_check": {k: rejected_by_check[k] for k in sorted(rejected_by_check)},
        "accepted_by_disease": {k: accepted_by_disease[k] for k in sorted(accepted_by_disease)},
    }
    return corpus, report


def _select_models(
    concept_models: ConceptModelSet, condition: str | None, crop: str | None
) -> list[ConceptModel]:
    models = sorted(concept_models.disease_models, key=lambda m: m.disease_id)
    if condition is not None:
        models = [m for m in models if m.disease_id == condition]
    if crop is not None:
        models = [m for m in models if m.crop == crop]
    return models


def _selections(model: ConceptModel, template: Template) -> list[frozenset[str]]:
    """A bounded, deterministic family of concept subsets to realize this template."""
    available = set(model.mandatory) | set(model.optional)
    opts = [c for c in template.optional if c in available]
    base = list(template.required)
    subsets: list[frozenset[str]] = [frozenset(base)]
    for opt in opts:
        subsets.append(frozenset([*base, opt]))
    if len(opts) >= 2:
        subsets.append(frozenset(base + opts[:_MEDIUM_SUBSET_SIZE]))
    if template.family in ("dense", "long") and opts:
        subsets.append(frozenset(base + opts))
    # Deduplicate while preserving deterministic order.
    seen: set[frozenset[str]] = set()
    unique: list[frozenset[str]] = []
    for s in subsets:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _assemble(
    model: ConceptModel, template: Template, selected: frozenset[str], variant: str
) -> Caption:
    """Plan + generate + attach traceable metadata for one caption."""
    plan = plan_caption(model, template, selected, variant)
    text = generate(plan)
    concepts = {c.concept_id: c for c in model.concepts}
    asserted = plan.asserted_concepts
    ranks = [_CONFIDENCE_RANK[concepts[c].confidence] for c in asserted if c in concepts]
    confidence = _RANK_CONFIDENCE[min(ranks)] if ranks else "asserted"
    observable = all(concepts[c].observable for c in asserted if c in concepts)
    evidence: set[str] = set()
    for c in asserted:
        if c in concepts:
            evidence.update(concepts[c].evidence)
    caption_id = "cap_" + sha256_hex(model.disease_id, template.id, text)[:16]
    return Caption(
        caption_id=caption_id,
        disease_id=model.disease_id,
        crop=model.crop,
        condition_type=model.condition_type,
        template_id=template.id,
        family=template.family,
        register=template.register,
        hedged=template.hedged,
        confidence=confidence,
        observable=observable,
        text=text,
        concepts=asserted,
        evidence=tuple(sorted(evidence)),
    )
