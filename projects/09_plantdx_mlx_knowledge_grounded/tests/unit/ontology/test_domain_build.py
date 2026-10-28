"""Build-side tests for the domain ontology compiler: hierarchy, graph, ordering."""

from __future__ import annotations

import json

import pytest
from _dkb import minimal_dkb

from plantdx.ontology.domain import builder, policies, serialization
from plantdx.ontology.domain.builder import load_dkb, validate_dkb


def _compile():
    dkb = minimal_dkb()
    validate_dkb(dkb)
    ontology, _log = builder.build_ontology(dkb, "testsha")
    return dkb, ontology


@pytest.mark.unit
def test_load_and_validate_dkb(tmp_path) -> None:
    path = tmp_path / "dkb.json"
    path.write_text(json.dumps(minimal_dkb()), encoding="utf-8")
    dkb = load_dkb(path)
    assert len(dkb["diseases"]) == 3
    validate_dkb(dkb)  # must not raise


@pytest.mark.unit
def test_concept_hierarchy_inheritance() -> None:
    assert policies.is_subtype("Disease", "Condition")
    assert policies.is_subtype("Bacterium", "Pathogen")
    assert policies.is_subtype("Bacterium", "CausalAgent")
    assert not policies.is_subtype("Disease", "Pathogen")
    assert policies.ancestors("Bacterium")[-1] == "Entity"  # every chain roots at Entity


@pytest.mark.unit
def test_graph_generation() -> None:
    _dkb, ontology = _compile()
    types = {n.id: n.type for n in ontology.nodes}
    assert types["condition:tomato_test_blight"] == "Disease"
    assert types["condition:tomato_healthy"] == "HealthyState"
    # symptoms and shared value nodes exist
    assert any(n.type == "Symptom" for n in ontology.nodes)
    assert "color:brown" in types and types["color:brown"] == "Color"
    # a disease has caused_by; the healthy class has none (rule F4)
    caused = {(e.source) for e in ontology.edges if e.type == "caused_by"}
    assert "condition:tomato_test_blight" in caused
    assert "condition:tomato_healthy" not in caused


@pytest.mark.unit
def test_shared_value_node_is_reused() -> None:
    _dkb, ontology = _compile()
    # color:brown is a single node referenced by both fungal diseases (a real graph)
    brown_sources = {
        e.source for e in ontology.edges if e.type == "has_color" and e.target == "color:brown"
    }
    assert "condition:tomato_test_blight" in brown_sources
    assert "condition:tomato_test_spot" in brown_sources


@pytest.mark.unit
def test_differentials_matched_bidirectionally() -> None:
    _dkb, ontology = _compile()
    diff = {(e.source, e.target) for e in ontology.edges if e.type == "differentiated_from"}
    assert ("condition:tomato_test_blight", "condition:tomato_test_spot") in diff
    assert ("condition:tomato_test_spot", "condition:tomato_test_blight") in diff


@pytest.mark.unit
def test_severity_split_licensing() -> None:
    _dkb, ontology = _compile()
    for e in ontology.edges:
        if e.type == "typical_at_severity":
            assert e.attributes["flags"]["image_licensed"] is False
        if e.type == "has_extent":
            assert e.attributes["flags"]["image_licensed"] is True


@pytest.mark.unit
def test_node_and_property_ordering_is_sorted() -> None:
    _dkb, ontology = _compile()
    doc = serialization.ontology_document(ontology)
    node_ids = [n["id"] for n in doc["nodes"]]
    assert node_ids == sorted(node_ids)  # nodes sorted by id
    text = serialization.canonical_json(doc)
    # canonical_json sorts keys: a re-dump with sort_keys must be identical
    assert text == json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False, indent=2) + "\n"


@pytest.mark.unit
def test_serialization_roundtrips() -> None:
    _dkb, ontology = _compile()
    for doc_fn in (
        serialization.ontology_document,
        serialization.concept_graph_document,
        serialization.concept_index_document,
    ):
        parsed = json.loads(serialization.canonical_json(doc_fn(ontology)))
        assert isinstance(parsed, dict) and parsed
