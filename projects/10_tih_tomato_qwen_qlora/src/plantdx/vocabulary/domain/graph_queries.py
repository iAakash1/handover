"""Small, pure read-only queries over a compiled :class:`Ontology`.

Shared by the Vocabulary Builder (B) and the Symptom Lexicon Builder (C); no
graph database, just dict indices over the ontology's flat node/edge lists,
mirroring the read side of ``plantdx.ontology.domain.graph.ConceptGraph``.
"""

from __future__ import annotations

from plantdx.ontology.domain.models import Edge, Node, Ontology

_CONFIDENCE_RANK = {"hedged": 1, "typical": 2, "asserted": 3}

# CausalAgent subtypes carry their name under `scientific_name`, not
# `canonical_label` (see ontology_design/ concept schema); every other node
# type in the ontology uses `canonical_label`.
_AGENT_CONCEPT_TYPES = frozenset(
    {"Bacterium", "Fungus", "Oomycete", "Virus", "ArthropodPest", "InsectPest", "Saprophyte"}
)


def node_label(node: Node) -> str:
    """The best human-readable label for a node (falls back to the node id)."""
    key = "scientific_name" if node.type in _AGENT_CONCEPT_TYPES else "canonical_label"
    return str(node.properties.get(key, node.id))


def max_confidence(edges: list[Edge]) -> str:
    """Highest-priority confidence across a set of edges (default "asserted")."""
    best, best_rank = "asserted", 0
    for edge in edges:
        value = edge.attributes.get("confidence")
        if value and _CONFIDENCE_RANK[value] > best_rank:
            best, best_rank = value, _CONFIDENCE_RANK[value]
    return best


def min_confidence(a: str, b: str) -> str:
    """The weaker of two confidence levels (a compound claim is as strong as its weakest part)."""
    return a if _CONFIDENCE_RANK[a] <= _CONFIDENCE_RANK[b] else b


def union_evidence(edges: list[Edge]) -> tuple[str, ...]:
    """Sorted union of the `evidence` attribute across a set of edges."""
    ids: set[str] = set()
    for edge in edges:
        ids.update(edge.attributes.get("evidence", []))
    return tuple(sorted(ids))


def index_in_edges(ontology: Ontology, relation: str) -> dict[str, list[Edge]]:
    """Target node id -> edges of `relation` type pointing at it."""
    index: dict[str, list[Edge]] = {}
    for edge in ontology.edges:
        if edge.type == relation:
            index.setdefault(edge.target, []).append(edge)
    return index


def index_out_edges(ontology: Ontology, relation: str | None = None) -> dict[str, list[Edge]]:
    """Source node id -> outgoing edges, optionally filtered to one relation type."""
    index: dict[str, list[Edge]] = {}
    for edge in ontology.edges:
        if relation is None or edge.type == relation:
            index.setdefault(edge.source, []).append(edge)
    return index


def index_symptom_owner(ontology: Ontology) -> dict[str, str]:
    """Symptom node id -> owning condition node id, via has_symptom edges."""
    return {edge.target: edge.source for edge in ontology.edges if edge.type == "has_symptom"}
