"""The ontology validator battery. Fail closed: any error aborts the build.

Implements the checks of ontology_design/09 that apply to the built graph
(V-ONT-1..11). Each check appends human-readable violations; ``validate`` raises
:class:`OntologyValidationError` with the full, sorted list if any exist.
"""

from __future__ import annotations

from typing import Any

from plantdx.core.exceptions import PlantDxError
from plantdx.ontology.domain import policies
from plantdx.ontology.domain.models import Edge, Ontology

_CONDITION_TYPES = frozenset({"Disease", "PestDamage", "SurfaceColonization", "HealthyState"})


class OntologyValidationError(PlantDxError):
    """Raised when the built ontology violates one or more rules (fail closed)."""

    def __init__(self, violations: list[str]) -> None:
        """Initialize the error with the sorted list of rule violations."""
        self.violations = violations
        super().__init__(
            f"ontology validation failed ({len(violations)} error(s)):\n  "
            + "\n  ".join(violations)
        )


def validate(ontology: Ontology, dkb: dict[str, Any]) -> None:
    """Run the full battery; raise on any violation."""
    node_type = {n.id: n.type for n in ontology.nodes}
    node_props = {n.id: n.properties for n in ontology.nodes}
    edges = ontology.edges
    out: dict[tuple[str, str], list[Edge]] = {}
    in_count: dict[tuple[str, str], int] = {}
    for e in edges:
        out.setdefault((e.source, e.type), []).append(e)
        in_count[(e.target, e.type)] = in_count.get((e.target, e.type), 0) + 1

    violations: list[str] = []
    violations += _v1_schema(ontology, node_type)
    violations += _v6_referential(edges, node_type)
    violations += _v2_domain_range(edges, node_type)
    violations += _v3_cardinality(ontology, node_type, out, in_count)
    violations += _v4_forbidden(ontology, node_type, node_props, out, edges)
    violations += _v5_consistency(node_type, node_props, out, edges)
    violations += _v8_observability(ontology, node_props, edges)
    violations += _v10_acyclic()
    violations += _v11_coverage(dkb)

    if violations:
        raise OntologyValidationError(sorted(set(violations)))


def _v1_schema(o: Ontology, node_type: dict[str, str]) -> list[str]:
    out: list[str] = []
    for node in o.nodes:
        ct = policies.CONCEPT_TYPE_BY_ID.get(node.type)
        if ct is None:
            out.append(f"V-ONT-1: node {node.id} has unknown type {node.type!r}")
        elif ct.abstract:
            out.append(f"V-ONT-1: node {node.id} has abstract type {node.type!r}")
    for edge in o.edges:
        if edge.type not in policies.RELATION_TYPE_BY_NAME:
            out.append(f"V-ONT-1: edge {edge.id} has unknown relation {edge.type!r}")
    return out


def _v6_referential(edges: list[Edge], node_type: dict[str, str]) -> list[str]:
    out: list[str] = []
    for edge in edges:
        if edge.source not in node_type:
            out.append(f"V-ONT-6: edge {edge.id} dangling source {edge.source}")
        if edge.target not in node_type:
            out.append(f"V-ONT-6: edge {edge.id} dangling target {edge.target}")
        for ev in edge.attributes.get("evidence", []):
            if ev not in node_type:
                out.append(f"V-ONT-7: edge {edge.id} references missing evidence {ev}")
            elif not policies.is_subtype(node_type[ev], "Evidence"):
                out.append(f"V-ONT-7: edge {edge.id} evidence {ev} is not an Evidence node")
    return out


def _v2_domain_range(edges: list[Edge], node_type: dict[str, str]) -> list[str]:
    out: list[str] = []
    for edge in edges:
        rt = policies.RELATION_TYPE_BY_NAME.get(edge.type)
        st, tt = node_type.get(edge.source), node_type.get(edge.target)
        if rt is None or st is None or tt is None:
            continue  # handled by V1/V6
        if not any(policies.is_subtype(st, d) for d in rt.domain):
            out.append(f"V-ONT-2: edge {edge.id} source type {st} not in domain {rt.domain}")
        if not any(policies.is_subtype(tt, r) for r in rt.range):
            out.append(f"V-ONT-2: edge {edge.id} target type {tt} not in range {rt.range}")
    return out


def _v3_cardinality(
    o: Ontology,
    node_type: dict[str, str],
    out: dict[tuple[str, str], list[Edge]],
    in_count: dict[tuple[str, str], int],
) -> list[str]:
    v: list[str] = []
    for node in o.nodes:
        if node.type in _CONDITION_TYPES:
            n_symptoms = len(out.get((node.id, "has_symptom"), []))
            if n_symptoms == 0:
                v.append(f"V-ONT-3: condition {node.id} has no has_symptom edge")
            if node.type == "HealthyState":
                if n_symptoms != 1:
                    v.append(f"V-ONT-3: HealthyState {node.id} must have exactly 1 symptom")
                if out.get((node.id, "caused_by")):
                    v.append(f"V-ONT-3: HealthyState {node.id} must have no caused_by")
        elif node.type == "Symptom":
            if len(out.get((node.id, "has_sign_type"), [])) != 1:
                v.append(f"V-ONT-3: symptom {node.id} must have exactly 1 has_sign_type")
            if in_count.get((node.id, "has_symptom"), 0) != 1:
                v.append(f"V-ONT-3: symptom {node.id} must belong to exactly 1 condition")
        elif policies.is_subtype(node.type, "CausalAgent") and node.type != "NoAgent":
            if len(out.get((node.id, "agent_in_category"), [])) != 1:
                v.append(f"V-ONT-3: agent {node.id} must have exactly 1 agent_in_category")
    return v


def _v4_forbidden(
    o: Ontology,
    node_type: dict[str, str],
    node_props: dict[str, dict[str, Any]],
    out: dict[tuple[str, str], list[Edge]],
    edges: list[Edge],
) -> list[str]:
    v: list[str] = []
    sign_of = {e.source: e.target for e in edges if e.type == "has_sign_type"}
    for edge in edges:
        st, tt = node_type.get(edge.source), node_type.get(edge.target)
        if edge.type == "caused_by":
            if st == "PestDamage" and tt is not None and policies.is_subtype(tt, "Pathogen"):
                v.append(f"V-ONT-4 (F1): {edge.id} PestDamage caused_by a Pathogen")
            if st == "HealthyState":
                v.append(f"V-ONT-4 (F4): {edge.id} HealthyState has a caused_by edge")
            if not edge.attributes.get("evidence"):
                v.append(f"V-ONT-4 (F10): {edge.id} caused_by has no evidence")
        elif edge.type == "has_symptom":
            if st == "HealthyState" and sign_of.get(edge.target) != "sign:healthy_surface":
                v.append(f"V-ONT-4 (F3): {edge.id} HealthyState has a non-healthy symptom")
            if (
                edge.attributes.get("confidence") == "asserted"
                and node_props.get(edge.target, {}).get("observable") is False
            ):
                v.append(f"V-ONT-4 (F7): {edge.id} asserted symptom that is non-observable")
        elif edge.type == "appears_on":
            if tt not in ("LeafRegion", "PlantPart"):
                v.append(f"V-ONT-4 (F5): {edge.id} appears_on a non-anatomy target ({tt})")
        elif edge.type == "typical_at_severity":
            if edge.attributes.get("flags", {}).get("image_licensed") is not False:
                v.append(
                    f"V-ONT-4 (F8): {edge.id} typical_at_severity must be image_licensed=false"
                )
    return v


def _v5_consistency(
    node_type: dict[str, str],
    node_props: dict[str, dict[str, Any]],
    out: dict[tuple[str, str], list[Edge]],
    edges: list[Edge],
) -> list[str]:
    v: list[str] = []
    agent_category = {}  # agent id -> category (via agent_in_category)
    for e in edges:
        if e.type == "agent_in_category":
            agent_category[e.source] = e.target.split(":", 1)[1]
    for e in edges:
        if e.type == "caused_by":
            expected = node_props.get(e.source, {}).get("agent_category")
            actual = agent_category.get(e.target)
            if actual is not None and expected is not None and actual != expected:
                v.append(f"V-ONT-5 (C1): {e.id} agent category {actual} != condition {expected}")
        elif e.type == "has_extent":
            if e.attributes.get("flags", {}).get("image_licensed") is not True:
                v.append(f"V-ONT-5 (C5): {e.id} has_extent must be image_licensed=true")
    return v


def _v8_observability(
    o: Ontology, node_props: dict[str, dict[str, Any]], edges: list[Edge]
) -> list[str]:
    v: list[str] = []
    for node in o.nodes:
        if node.type != "Symptom":
            continue
        expected = node.properties.get("source_field") != policies.FORBIDDEN_FIELD
        if node.properties.get("observable") != expected:
            v.append(f"V-ONT-8: symptom {node.id} observable != field-derived value")
    for e in edges:
        if (
            e.type == "appears_on"
            and e.target.startswith("part:")
            and node_props.get(e.source, {}).get("observable") is not False
        ):
            v.append(f"V-ONT-8: symptom {e.source} appears_on non-leaf part but is observable")
    return v


def _v10_acyclic() -> list[str]:
    v: list[str] = []
    for concept in policies.CONCEPT_TYPES:
        seen: set[str] = set()
        cur: str | None = concept.id
        while cur is not None:
            if cur in seen:
                v.append(f"V-ONT-10: is_a cycle at {concept.id}")
                break
            seen.add(cur)
            parent = policies.CONCEPT_TYPE_BY_ID[cur].is_a
            cur = parent
    return v


def _v11_coverage(dkb: dict[str, Any]) -> list[str]:
    v: list[str] = []
    known = policies.CONSUMED_FIELDS | policies.ALLOWLIST_FIELDS
    for disease in dkb.get("diseases", []):
        uncovered = set(disease.keys()) - known
        if uncovered:
            v.append(
                f"V-ONT-11: disease {disease.get('id')} has uncovered fields {sorted(uncovered)}"
            )
    return v
