"""Validation-failure tests: fail closed on bad DKBs and bad graphs."""

from __future__ import annotations

import pytest
from _dkb import clone, minimal_dkb

from plantdx.core.exceptions import KnowledgeBaseError
from plantdx.ontology.domain import builder, policies, validator
from plantdx.ontology.domain.models import ConceptType, Edge, Node
from plantdx.ontology.domain.validator import OntologyValidationError


def _valid_ontology():
    dkb = minimal_dkb()
    ontology, _ = builder.build_ontology(dkb, "testsha")
    return dkb, ontology


# --- DKB-level failures -----------------------------------------------------


@pytest.mark.unit
def test_empty_dkb_rejected() -> None:
    with pytest.raises(KnowledgeBaseError):
        builder.validate_dkb({"diseases": [], "metadata": {"reference_registry": {}}})


@pytest.mark.unit
def test_malformed_dkb_missing_field() -> None:
    dkb = clone(minimal_dkb())
    del dkb["diseases"][0]["agent_category"]
    with pytest.raises(KnowledgeBaseError):
        builder.validate_dkb(dkb)


@pytest.mark.unit
def test_duplicate_disease_id_rejected() -> None:
    dkb = clone(minimal_dkb())
    dkb["diseases"][1]["id"] = dkb["diseases"][0]["id"]
    with pytest.raises(KnowledgeBaseError):
        builder.validate_dkb(dkb)


# --- graph-level failures (fail closed) -------------------------------------


@pytest.mark.unit
def test_valid_ontology_passes() -> None:
    dkb, ontology = _valid_ontology()
    validator.validate(ontology, dkb)  # must not raise


@pytest.mark.unit
def test_unknown_concept_type_detected() -> None:
    dkb, ontology = _valid_ontology()
    ontology.nodes.append(Node("bogus:1", "Bogus", {}))
    with pytest.raises(OntologyValidationError, match="V-ONT-1"):
        validator.validate(ontology, dkb)


@pytest.mark.unit
def test_forbidden_relationship_detected() -> None:
    dkb, ontology = _valid_ontology()
    # HealthyState must not have a caused_by edge (rule F4).
    ontology.edges.append(
        Edge(
            "e:bad",
            "caused_by",
            "condition:tomato_healthy",
            "agent:testus_fungus",
            {"confidence": "asserted", "evidence": ["evidence:REF1"]},
        )
    )
    with pytest.raises(OntologyValidationError, match="F4"):
        validator.validate(ontology, dkb)


@pytest.mark.unit
def test_dangling_edge_detected() -> None:
    dkb, ontology = _valid_ontology()
    ontology.edges.append(
        Edge("e:dangle", "affects", "condition:does_not_exist", "crop:tomato", {})
    )
    with pytest.raises(OntologyValidationError, match="V-ONT-6"):
        validator.validate(ontology, dkb)


@pytest.mark.unit
def test_uncovered_dkb_field_detected() -> None:
    dkb, ontology = _valid_ontology()
    dkb["diseases"][0]["a_new_unmapped_field"] = ["x"]
    with pytest.raises(OntologyValidationError, match="V-ONT-11"):
        validator.validate(ontology, dkb)


# --- inheritance acyclicity -------------------------------------------------


@pytest.mark.unit
def test_shipped_taxonomy_is_acyclic() -> None:
    assert validator._v10_acyclic() == []  # the real T-Box has no is_a cycle


@pytest.mark.unit
def test_circular_inheritance_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    cyclic = {"A": ConceptType("A", "B"), "B": ConceptType("B", "A")}
    monkeypatch.setattr(policies, "CONCEPT_TYPES", tuple(cyclic.values()))
    monkeypatch.setattr(policies, "CONCEPT_TYPE_BY_ID", cyclic)
    assert validator._v10_acyclic()  # cycle detected -> non-empty
