"""Deterministic ontology statistics (``ontology_statistics.json``)."""

from __future__ import annotations

from typing import Any

from plantdx.ontology.domain import policies
from plantdx.ontology.domain.models import Ontology

_CONDITION_TYPES = ("Disease", "PestDamage", "SurfaceColonization", "HealthyState")


def _inheritance_depth() -> int:
    return max(len(policies.ancestors(c.id)) for c in policies.CONCEPT_TYPES)


def compute(ontology: Ontology, dkb: dict[str, Any], validation_status: str) -> dict[str, Any]:
    """Return the statistics document (all values deterministic)."""
    nodes_by_type: dict[str, int] = {}
    for node in ontology.nodes:
        nodes_by_type[node.type] = nodes_by_type.get(node.type, 0) + 1
    edges_by_relation: dict[str, int] = {}
    for edge in ontology.edges:
        edges_by_relation[edge.type] = edges_by_relation.get(edge.type, 0) + 1

    property_count = sum(len(n.properties) for n in ontology.nodes)
    consumed = len(policies.CONSUMED_FIELDS)
    total_fields = len(policies.CONSUMED_FIELDS | policies.ALLOWLIST_FIELDS)

    return {
        "schema_version": ontology.schema_version,
        "ontology_version": ontology.ontology_version,
        "build_checksum": ontology.provenance.get("content_hash", ""),
        "validation_status": validation_status,
        "concept_count": len(ontology.nodes),
        "edge_count": len(ontology.edges),
        "relation_count": len(edges_by_relation),
        "concept_type_count": len(ontology.concept_types),
        "relation_type_count": len(ontology.relation_types),
        "disease_concepts": nodes_by_type.get("Disease", 0),
        "condition_concepts": sum(nodes_by_type.get(t, 0) for t in _CONDITION_TYPES),
        "symptom_concepts": nodes_by_type.get("Symptom", 0),
        "color_concepts": nodes_by_type.get("Color", 0),
        "shape_concepts": nodes_by_type.get("Shape", 0),
        "texture_concepts": nodes_by_type.get("Texture", 0),
        "morphology_concepts": nodes_by_type.get("Shape", 0) + nodes_by_type.get("Texture", 0),
        "evidence_concepts": sum(
            nodes_by_type.get(t, 0) for t in ("PeerReviewed", "ExtensionService", "Textbook")
        ),
        "property_count": property_count,
        "inheritance_depth": _inheritance_depth(),
        "nodes_by_type": {k: nodes_by_type[k] for k in sorted(nodes_by_type)},
        "edges_by_relation": {k: edges_by_relation[k] for k in sorted(edges_by_relation)},
        "coverage": {
            "dkb_field_count": total_fields,
            "consumed_fields": consumed,
            "allowlisted_fields": total_fields - consumed,
            "consumed_fraction": round(consumed / total_fields, 4),
            "all_fields_accounted": validation_status == "valid",
        },
    }
