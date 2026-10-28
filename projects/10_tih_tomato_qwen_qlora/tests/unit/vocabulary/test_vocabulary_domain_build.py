"""Build-side tests for the vocabulary builder (B) and symptom lexicon (C)."""

from __future__ import annotations

import pytest
from _vocabulary_dkb import minimal_dkb

from plantdx.ontology.domain import builder as ontology_builder
from plantdx.vocabulary.domain import builder, graph_queries, lexicon, policies


def _compile_ontology():
    dkb = minimal_dkb()
    ontology_builder.validate_dkb(dkb)
    ontology, _log = ontology_builder.build_ontology(dkb, "testsha")
    return ontology


@pytest.mark.unit
def test_vocabulary_covers_every_category() -> None:
    ontology = _compile_ontology()
    items = builder.build_vocabulary(ontology)
    concepts = {item.concept for item in items}
    expected = {c.category for c in policies.CATEGORIES} | {policies.CONFIDENCE_CATEGORY.category}
    assert concepts == expected


@pytest.mark.unit
def test_agent_name_uses_scientific_name_not_node_id() -> None:
    ontology = _compile_ontology()
    items = builder.build_vocabulary(ontology)
    agent_items = {i.ontology_node: i for i in items if i.concept == "agent_name"}
    testus = agent_items["agent:testus_fungus"]
    assert testus.surface_form == "Testus fungus"
    assert testus.canonical_form == "Testus fungus"
    assert "tomato_test_blight" in testus.dkb_reference


@pytest.mark.unit
def test_disease_name_items_cover_all_condition_subtypes() -> None:
    ontology = _compile_ontology()
    items = builder.build_vocabulary(ontology)
    disease_labels = {i.surface_form for i in items if i.concept == "disease_name"}
    assert disease_labels == {"Test Blight", "Test Spot", "Healthy"}


@pytest.mark.unit
def test_shared_quality_value_traces_to_both_diseases() -> None:
    ontology = _compile_ontology()
    items = builder.build_vocabulary(ontology)
    brown = next(i for i in items if i.ontology_node == "color:brown")
    assert set(brown.dkb_reference) == {"tomato_test_blight", "tomato_test_spot"}


@pytest.mark.unit
def test_confidence_modifiers_are_fixed_and_ungrounded() -> None:
    ontology = _compile_ontology()
    items = builder.build_vocabulary(ontology)
    conf_items = [i for i in items if i.concept == policies.CONFIDENCE_CATEGORY.category]
    assert {i.surface_form for i in conf_items} == set(policies.CONFIDENCE_VALUES)
    assert all(i.ontology_node == "" and i.evidence == () for i in conf_items)


@pytest.mark.unit
def test_every_item_matches_the_required_schema() -> None:
    ontology = _compile_ontology()
    items = builder.build_vocabulary(ontology) + lexicon.build_lexicon(ontology)
    required_fields = {
        "id",
        "surface_form",
        "canonical_form",
        "concept",
        "concept_id",
        "confidence",
        "source",
        "ontology_node",
        "dkb_reference",
        "evidence",
        "language",
        "part_of_speech",
    }
    for item in items:
        assert required_fields <= vars(item).keys()
        assert item.language == "en"
        assert item.confidence in policies.CONFIDENCE_VALUES


@pytest.mark.unit
def test_node_label_prefers_scientific_name_for_agents() -> None:
    ontology = _compile_ontology()
    agent_node = next(n for n in ontology.nodes if n.id == "agent:testus_fungus")
    color_node = next(n for n in ontology.nodes if n.id == "color:brown")
    assert graph_queries.node_label(agent_node) == "Testus fungus"
    assert graph_queries.node_label(color_node) == "brown"
