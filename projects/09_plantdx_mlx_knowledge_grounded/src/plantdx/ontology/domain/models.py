"""Value objects for the domain-ontology compiler.

A node is a typed individual; an edge is a typed, directed relation with
attributes (confidence, evidence, flags). Plain dataclasses, no behavior.
Determinism is the caller's responsibility (everything is sorted before output).

This is the *domain ontology* (knowledge graph) of ontology_design/. It is
distinct from the caption-concept model in ``plantdx.ontology.models`` (which is
a downstream view and a separate milestone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    """Assertion-licensing level on a domain edge (ontology_design/04, /05)."""

    ASSERTED = "asserted"  # licensed by the dataset label (hallmark / diagnostic)
    TYPICAL = "typical"  # characteristic of the class, not guaranteed per image
    HEDGED = "hedged"  # secondary / rare


@dataclass
class Node:
    """A typed individual in the graph."""

    id: str
    type: str  # a concept-type id declared in the T-Box (policies.CONCEPT_TYPES)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A typed, directed relation between two nodes, with attributes."""

    id: str
    type: str  # a relation-type name declared in the T-Box (policies.RELATION_TYPES)
    source: str
    target: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConceptType:
    """A T-Box concept type with its single-inheritance parent."""

    id: str
    is_a: str | None
    abstract: bool = False
    closed: bool = False


@dataclass(frozen=True)
class RelationType:
    """A T-Box relation type: domain/range plus the edge-attribute contract."""

    name: str
    domain: tuple[str, ...]
    range: tuple[str, ...]
    cardinality_out: str  # "1", "0..1", "0..n", "1..n"
    cardinality_in: str
    carries_confidence: bool = False
    carries_evidence: bool = False
    carries_flags: tuple[str, ...] = ()
    symmetric: bool = False
    transitive: bool = False


@dataclass
class Ontology:
    """The complete built ontology: T-Box + A-Box + provenance (no timestamps)."""

    schema_version: str
    ontology_version: str
    concept_types: list[ConceptType]
    relation_types: list[RelationType]
    nodes: list[Node]
    edges: list[Edge]
    provenance: dict[str, Any] = field(default_factory=dict)
