"""Symptom Lexicon Builder (C) tests: boundedness, dedup, primary-only modifiers."""

from __future__ import annotations

import pytest
from _vocabulary_dkb import minimal_dkb

from plantdx.ontology.domain import builder as ontology_builder
from plantdx.vocabulary.domain import lexicon


def _compile_ontology():
    dkb = minimal_dkb()
    ontology_builder.validate_dkb(dkb)
    ontology, _log = ontology_builder.build_ontology(dkb, "testsha")
    return ontology


@pytest.mark.unit
def test_every_symptom_gets_a_base_realization() -> None:
    ontology = _compile_ontology()
    items = lexicon.build_lexicon(ontology)
    symptom_ids = {n.id for n in ontology.nodes if n.type == "Symptom"}
    base_symptom_ids = {i.ontology_node for i in items if ":base" in i.id}
    assert base_symptom_ids == symptom_ids


@pytest.mark.unit
def test_secondary_symptom_gets_no_modifiers() -> None:
    """``leaflet yellowing`` is a secondary symptom on a modifiable (lesion) sign
    type — modifiers must still be withheld because it isn't primary."""
    ontology = _compile_ontology()
    items = lexicon.build_lexicon(ontology)
    secondary = next(
        n
        for n in ontology.nodes
        if n.type == "Symptom" and n.properties["canonical_label"] == "leaflet yellowing"
    )
    realizations = [i for i in items if i.ontology_node == secondary.id]
    assert len(realizations) == 1
    assert ":base" in realizations[0].id


@pytest.mark.unit
def test_healthy_symptom_gets_no_modifiers() -> None:
    ontology = _compile_ontology()
    items = lexicon.build_lexicon(ontology)
    healthy = next(n for n in ontology.nodes if n.id == "symptom:tomato_healthy:healthy:0")
    realizations = [i for i in items if i.ontology_node == healthy.id]
    assert len(realizations) == 1


@pytest.mark.unit
def test_primary_symptom_gets_one_modifier_per_attached_value() -> None:
    """Bounded, linear realization: one item per attached quality value, not a
    Cartesian product across axes."""
    ontology = _compile_ontology()
    items = lexicon.build_lexicon(ontology)
    primary = next(n for n in ontology.nodes if n.id == "symptom:tomato_test_blight:prm:0")
    realizations = [i for i in items if i.ontology_node == primary.id]
    # color(2) + shape(2) + extent(2) + texture(2, one deduped against shape) = 7 modifiers + 1 base
    assert len(realizations) == 7 + 1


@pytest.mark.unit
def test_cross_axis_word_collision_is_deduplicated() -> None:
    """tomato_test_blight has both shape:raised and texture:raised — only the
    higher-priority axis (shape, per MODIFIER_RELATIONS order) survives."""
    ontology = _compile_ontology()
    items = lexicon.build_lexicon(ontology)
    primary = next(n for n in ontology.nodes if n.id == "symptom:tomato_test_blight:prm:0")
    realizations = [i for i in items if i.ontology_node == primary.id]
    phrases = [i.canonical_form for i in realizations]
    assert phrases.count("raised lesion") == 1
    sources = {i.source for i in realizations if i.canonical_form == "raised lesion"}
    assert sources == {"has_shape"}


@pytest.mark.unit
def test_modifier_confidence_is_weakest_link() -> None:
    ontology = _compile_ontology()
    items = lexicon.build_lexicon(ontology)
    primary = next(n for n in ontology.nodes if n.id == "symptom:tomato_test_blight:prm:0")
    modifier = next(
        i for i in items if i.ontology_node == primary.id and i.canonical_form == "brown lesion"
    )
    # has_symptom (primary_symptoms) is "asserted", has_color is "typical" -> min is "typical".
    assert modifier.confidence == "typical"


@pytest.mark.unit
def test_no_combinatorial_explosion() -> None:
    """A modifier realization is always exactly ``<modifier> <head noun>`` — one
    quality value, never a pair/triple combination stacked across axes."""
    ontology = _compile_ontology()
    items = lexicon.build_lexicon(ontology)
    modifier_items = [i for i in items if ":mod:" in i.id]
    assert modifier_items  # the fixture does exercise the modifier path
    assert all(i.canonical_form.count(" ") == 1 for i in modifier_items)
