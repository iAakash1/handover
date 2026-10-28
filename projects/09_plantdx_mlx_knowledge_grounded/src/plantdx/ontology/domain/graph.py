"""The in-memory concept graph: a plain container with idempotent upserts.

No graph database, no RDF engine — just dicts of nodes and edges keyed by
deterministic ids. Re-adding a node merges properties; re-adding an edge merges
confidence (max), evidence (union), and flags (OR). Iteration is always sorted.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from plantdx.ontology.domain.models import Edge, Node

_CONFIDENCE_RANK = {"hedged": 1, "typical": 2, "asserted": 3}


class ConceptGraph:
    """A deterministic node/edge store keyed by id."""

    def __init__(self) -> None:
        """Initialize an empty node/edge store."""
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}

    # --- nodes ---
    def upsert_node(
        self, node_id: str, type_id: str, properties: dict[str, Any] | None = None
    ) -> Node:
        """Create the node, or merge new properties into an existing one."""
        existing = self._nodes.get(node_id)
        if existing is not None:
            for key, value in (properties or {}).items():
                existing.properties.setdefault(key, value)
            return existing
        node = Node(node_id, type_id, dict(properties or {}))
        self._nodes[node_id] = node
        return node

    def has_node(self, node_id: str) -> bool:
        """Return True if a node with this id exists."""
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Node | None:
        """Return the node with this id, or None if it does not exist."""
        return self._nodes.get(node_id)

    def nodes(self) -> list[Node]:
        """All nodes, sorted by id."""
        return [self._nodes[k] for k in sorted(self._nodes)]

    def nodes_of_type(self, type_id: str) -> list[Node]:
        """Return all nodes of the given concept type, sorted by id."""
        return [n for n in self.nodes() if n.type == type_id]

    # --- edges ---
    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge, or merge into the existing edge with the same id."""
        existing = self._edges.get(edge.id)
        if existing is None:
            self._edges[edge.id] = edge
            return edge
        _merge_edge_attributes(existing.attributes, edge.attributes)
        return existing

    def edges(self) -> list[Edge]:
        """All edges, sorted by (type, source, target, id) for stable output."""
        return sorted(self._edges.values(), key=lambda e: (e.type, e.source, e.target, e.id))

    def out_edges(self, source: str, relation: str | None = None) -> list[Edge]:
        """Return edges leaving `source`, optionally filtered to one relation type."""
        return [
            e
            for e in self.edges()
            if e.source == source and (relation is None or e.type == relation)
        ]

    def in_edges(self, target: str, relation: str | None = None) -> list[Edge]:
        """Return edges entering `target`, optionally filtered to one relation type."""
        return [
            e
            for e in self.edges()
            if e.target == target and (relation is None or e.type == relation)
        ]


def _merge_edge_attributes(into: dict[str, Any], other: dict[str, Any]) -> None:
    """Merge confidence (max), evidence (sorted union), flags (OR); keep the rest."""
    if "confidence" in other:
        current = into.get("confidence")
        if current is None or _CONFIDENCE_RANK[other["confidence"]] > _CONFIDENCE_RANK[current]:
            into["confidence"] = other["confidence"]
    if "evidence" in other:
        merged: Iterable[str] = set(into.get("evidence", ())) | set(other["evidence"])
        into["evidence"] = sorted(merged)
    if "flags" in other:
        flags = dict(into.get("flags", {}))
        for key, value in other["flags"].items():
            flags[key] = bool(flags.get(key, False)) or bool(value)
        into["flags"] = flags
    for key, value in other.items():
        if key not in ("confidence", "evidence", "flags"):
            into.setdefault(key, value)
