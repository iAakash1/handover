"""Validation-failure tests: the vocabulary/lexicon validator battery fails closed."""

from __future__ import annotations

import dataclasses

import pytest
from _vocabulary_dkb import minimal_dkb

from plantdx.ontology.domain import builder as ontology_builder
from plantdx.ontology.domain.models import Node
from plantdx.vocabulary.domain import builder, lexicon, validator
from plantdx.vocabulary.domain.models import VocabularyResult
from plantdx.vocabulary.domain.validator import VocabularyValidationError


def _valid_result():
    dkb = minimal_dkb()
    ontology_builder.validate_dkb(dkb)
    ontology, _log = ontology_builder.build_ontology(dkb, "testsha")
    result = VocabularyResult(
        vocabulary_items=builder.build_vocabulary(ontology),
        lexicon_items=lexicon.build_lexicon(ontology),
    )
    return ontology, result


@pytest.mark.unit
def test_valid_vocabulary_passes() -> None:
    ontology, result = _valid_result()
    validator.validate(result, ontology)  # must not raise
    assert validator.collect_violations(result, ontology) == []


@pytest.mark.unit
def test_duplicate_id_detected() -> None:
    ontology, result = _valid_result()
    result.vocabulary_items[1] = dataclasses.replace(
        result.vocabulary_items[1], id=result.vocabulary_items[0].id
    )
    with pytest.raises(VocabularyValidationError, match="V-VOC-1"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_duplicate_realization_detected() -> None:
    ontology, result = _valid_result()
    same_symptom = [
        i for i in result.lexicon_items if i.ontology_node == "symptom:tomato_test_blight:prm:0"
    ]
    a, b = same_symptom[0], same_symptom[1]
    result.lexicon_items[result.lexicon_items.index(b)] = dataclasses.replace(
        b, canonical_form=a.canonical_form
    )
    with pytest.raises(VocabularyValidationError, match="V-VOC-2"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_orphan_concept_detected() -> None:
    ontology, result = _valid_result()
    result.vocabulary_items[0] = dataclasses.replace(result.vocabulary_items[0], concept_id="Bogus")
    with pytest.raises(VocabularyValidationError, match="V-VOC-3"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_missing_evidence_detected() -> None:
    ontology, result = _valid_result()
    color_item = next(i for i in result.vocabulary_items if i.concept == "color")
    idx = result.vocabulary_items.index(color_item)
    result.vocabulary_items[idx] = dataclasses.replace(color_item, evidence=())
    with pytest.raises(VocabularyValidationError, match="V-VOC-4"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_illegal_combination_detected() -> None:
    ontology, result = _valid_result()
    modifier_item = next(i for i in result.lexicon_items if ":mod:" in i.id)
    idx = result.lexicon_items.index(modifier_item)
    result.lexicon_items[idx] = dataclasses.replace(modifier_item, source="caused_by")
    with pytest.raises(VocabularyValidationError, match="V-VOC-5"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_illegal_modifier_on_non_primary_symptom_detected() -> None:
    ontology, result = _valid_result()
    modifier_item = next(i for i in result.lexicon_items if ":mod:" in i.id)
    secondary_symptom_id = "symptom:tomato_test_blight:sec:0"
    idx = result.lexicon_items.index(modifier_item)
    result.lexicon_items[idx] = dataclasses.replace(
        modifier_item, ontology_node=secondary_symptom_id
    )
    with pytest.raises(VocabularyValidationError, match="V-VOC-6"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_invalid_realization_detected() -> None:
    ontology, result = _valid_result()
    result.vocabulary_items[0] = dataclasses.replace(result.vocabulary_items[0], surface_form="")
    with pytest.raises(VocabularyValidationError, match="V-VOC-7"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_unused_concept_detected() -> None:
    ontology, result = _valid_result()
    ontology.nodes.append(Node("color:unused_test_color", "Color", {"canonical_label": "puce"}))
    with pytest.raises(VocabularyValidationError, match="V-VOC-8"):
        validator.validate(result, ontology)


@pytest.mark.unit
def test_conflicting_realization_detected() -> None:
    ontology, result = _valid_result()
    disease_item = next(i for i in result.vocabulary_items if i.concept == "disease_name")
    idx = result.vocabulary_items.index(disease_item)
    result.vocabulary_items[idx] = dataclasses.replace(
        disease_item, canonical_form="Not The Real Label"
    )
    with pytest.raises(VocabularyValidationError, match="V-VOC-9"):
        validator.validate(result, ontology)
