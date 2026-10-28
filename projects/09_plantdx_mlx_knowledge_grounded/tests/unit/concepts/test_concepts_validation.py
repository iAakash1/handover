"""Fail-closed tests: the concept-model validator detects every injected fault."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from plantdx.concepts import validator
from plantdx.concepts.models import (
    STATUS_FORBIDDEN,
    STATUS_OPTIONAL,
    CaptionConcept,
    ConceptModelSet,
)


def _violations(models: ConceptModelSet, ontology: Any, vocab: Any) -> list[str]:
    return validator.collect_violations(models, ontology, vocab)


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_valid_models_pass(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    assert _violations(models, onto.ontology, vocab) == []


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_missing_disease_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    broken.disease_models.pop()
    assert any(v.startswith("V-CON-1") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_unknown_concept_id_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    broken.disease_models[0].concepts[0].concept_id = "bogus_concept"
    assert any(v.startswith("V-CON-2") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_mandatory_without_realization_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    m = broken.disease_models[0]
    cid = m.mandatory[0]
    for c in m.concepts:
        if c.concept_id == cid:
            c.realizations = ()
    assert any(v.startswith("V-CON-3") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_forbidden_with_realizations_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    m = broken.disease_models[0]
    for c in m.concepts:
        if c.concept_id == "lesion_size":  # always forbidden
            c.realizations = ("small",)
    assert any(v.startswith("V-CON-4") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_bad_budget_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    broken.disease_models[0].min_information = 999
    assert any(v.startswith("V-CON-5") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_bad_observability_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    for c in broken.disease_models[0].concepts:
        if c.concept_id == "primary_sign":
            c.observable = False  # primary_sign is observable
    assert any(v.startswith("V-CON-6") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_never_appear_missing_stage_token_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    broken.disease_models[0].never_appear = ()
    assert any(v.startswith("V-CON-9") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_bad_ordering_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    m = broken.disease_models[0]
    m.ordering = tuple(reversed(m.ordering))
    assert any(v.startswith("V-CON-11") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_modifier_on_non_modifiable_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    # tomato_mosaic_virus is 'mottle' (not modifiable); inject an illegal lesion_color.
    m = next(x for x in broken.disease_models if x.disease_id == "tomato_mosaic_virus")
    m.concepts = (
        *m.concepts,
        CaptionConcept(
            concept_id="lesion_color",
            status=STATUS_OPTIONAL,
            observable=True,
            confidence="typical",
            sign_type="mottle",
            realizations=("green",),
            modifiers=(),
            evidence=("evidence:x",),
            dkb_fields=("color_vocabulary",),
        ),
    )
    m.optional = (*m.optional, "lesion_color")
    assert any(v.startswith("V-CON-7") for v in _violations(broken, onto.ontology, vocab))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_evidence_missing_detected(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    broken = copy.deepcopy(models)
    for c in broken.disease_models[0].concepts:
        if c.concept_id == "primary_sign" and c.status != STATUS_FORBIDDEN:
            c.evidence = ()
    assert any(v.startswith("V-CON-8") for v in _violations(broken, onto.ontology, vocab))
