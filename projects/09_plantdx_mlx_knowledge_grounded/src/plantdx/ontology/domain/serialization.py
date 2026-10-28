"""Canonical, deterministic serialization of the ontology to JSON documents.

Every document is stable, pretty-printed, UTF-8, sorted (keys via ``sort_keys``,
lists sorted explicitly). No timestamps, no machine-dependent ordering.
"""

from __future__ import annotations

import json
from typing import Any

from plantdx.ontology.domain.models import ConceptType, Edge, Node, Ontology, RelationType


def canonical_json(obj: Any) -> str:
    """Serialize deterministically: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _concept_type_dict(c: ConceptType) -> dict[str, Any]:
    return {"id": c.id, "is_a": c.is_a, "abstract": c.abstract, "closed": c.closed}


def _relation_type_dict(r: RelationType) -> dict[str, Any]:
    return {
        "name": r.name,
        "domain": list(r.domain),
        "range": list(r.range),
        "cardinality_out": r.cardinality_out,
        "cardinality_in": r.cardinality_in,
        "carries_confidence": r.carries_confidence,
        "carries_evidence": r.carries_evidence,
        "carries_flags": list(r.carries_flags),
        "symmetric": r.symmetric,
        "transitive": r.transitive,
    }


def _node_dict(n: Node) -> dict[str, Any]:
    return {"id": n.id, "type": n.type, "properties": n.properties}


def _edge_dict(e: Edge) -> dict[str, Any]:
    return {
        "id": e.id,
        "type": e.type,
        "source": e.source,
        "target": e.target,
        "attributes": e.attributes,
    }


def _sorted_concept_types(o: Ontology) -> list[dict[str, Any]]:
    return [_concept_type_dict(c) for c in sorted(o.concept_types, key=lambda c: c.id)]


def _sorted_relation_types(o: Ontology) -> list[dict[str, Any]]:
    return [_relation_type_dict(r) for r in sorted(o.relation_types, key=lambda r: r.name)]


def _sorted_nodes(o: Ontology) -> list[dict[str, Any]]:
    return [_node_dict(n) for n in sorted(o.nodes, key=lambda n: n.id)]


def _sorted_edges(o: Ontology) -> list[dict[str, Any]]:
    return [
        _edge_dict(e) for e in sorted(o.edges, key=lambda e: (e.type, e.source, e.target, e.id))
    ]


def schema_content(o: Ontology) -> dict[str, Any]:
    """The T-Box only (basis of ``schema_hash``)."""
    return {
        "schema_version": o.schema_version,
        "concept_types": _sorted_concept_types(o),
        "relation_types": _sorted_relation_types(o),
    }


def semantic_content(o: Ontology) -> dict[str, Any]:
    """The full semantic content excluding provenance (basis of ``content_hash``)."""
    return {
        "schema_version": o.schema_version,
        "ontology_version": o.ontology_version,
        "concept_types": _sorted_concept_types(o),
        "relation_types": _sorted_relation_types(o),
        "nodes": _sorted_nodes(o),
        "edges": _sorted_edges(o),
    }


def ontology_document(o: Ontology) -> dict[str, Any]:
    """The complete ``ontology.json`` document (T-Box + A-Box + provenance)."""
    return {
        "kind": "plantdx.ontology.domain",
        "schema_version": o.schema_version,
        "ontology_version": o.ontology_version,
        "provenance": o.provenance,
        "concept_types": _sorted_concept_types(o),
        "relation_types": _sorted_relation_types(o),
        "nodes": _sorted_nodes(o),
        "edges": _sorted_edges(o),
    }


def concept_graph_document(o: Ontology) -> dict[str, Any]:
    """A graph-centric view (``concept_graph.json``): light nodes + directed edges."""
    return {
        "kind": "plantdx.ontology.concept_graph",
        "schema_version": o.schema_version,
        "ontology_version": o.ontology_version,
        "content_hash": o.provenance.get("content_hash", ""),
        "nodes": [{"id": n.id, "type": n.type} for n in sorted(o.nodes, key=lambda n: n.id)],
        "edges": [
            {"source": e.source, "relation": e.type, "target": e.target, "attributes": e.attributes}
            for e in sorted(o.edges, key=lambda e: (e.type, e.source, e.target, e.id))
        ],
    }


def concept_index_document(o: Ontology) -> dict[str, Any]:
    """Lookup indices (``concept_index.json``): by type, by crop, condition→symptoms."""
    by_type: dict[str, list[str]] = {}
    for node in o.nodes:
        by_type.setdefault(node.type, []).append(node.id)
    for ids in by_type.values():
        ids.sort()

    conditions = sorted(n.id for n in o.nodes if _is_condition(n.type))
    by_crop: dict[str, list[str]] = {}
    condition_symptoms: dict[str, list[str]] = {}
    for edge in o.edges:
        if edge.type == "affects":
            by_crop.setdefault(edge.target, []).append(edge.source)
        elif edge.type == "has_symptom":
            condition_symptoms.setdefault(edge.source, []).append(edge.target)
    for ids in by_crop.values():
        ids.sort()
    for ids in condition_symptoms.values():
        ids.sort()

    return {
        "kind": "plantdx.ontology.concept_index",
        "schema_version": o.schema_version,
        "ontology_version": o.ontology_version,
        "content_hash": o.provenance.get("content_hash", ""),
        "conditions": conditions,
        "by_type": {k: by_type[k] for k in sorted(by_type)},
        "by_crop": {k: by_crop[k] for k in sorted(by_crop)},
        "condition_symptoms": {k: condition_symptoms[k] for k in sorted(condition_symptoms)},
    }


_CONDITION_TYPES = frozenset({"Disease", "PestDamage", "SurfaceColonization", "HealthyState"})


def _is_condition(type_id: str) -> bool:
    return type_id in _CONDITION_TYPES
