"""Vocabulary Builder — component (B): ``Vocabulary = f(Ontology, Policies)``.

Deterministic projection of the ontology into flat, per-category concept
vocabulary. Consumes only :class:`plantdx.ontology.domain.models.Ontology` —
never the DKB, never an LLM/VLM, never randomness. Every item is traceable to
a specific ontology node and, through its grounding relation, to the DKB
disease(s) and evidence that licensed it.
"""

from __future__ import annotations

from plantdx.ontology.domain.models import Node, Ontology
from plantdx.vocabulary.domain import graph_queries, policies
from plantdx.vocabulary.domain.models import LexicalItem


def _disease_id(nodes_by_id: dict[str, Node], condition_id: str) -> str:
    return str(nodes_by_id[condition_id].properties.get("disease_id", condition_id))


def _direct_category_items(
    category: policies.CategorySpec,
    ontology: Ontology,
    nodes_by_id: dict[str, Node],
) -> list[LexicalItem]:
    """Items reached by a single in-edge of `category.relation` (color/shape/... )."""
    in_edges = graph_queries.index_in_edges(ontology, category.relation)
    items: list[LexicalItem] = []
    for node in ontology.nodes:
        if node.type not in category.concept_types:
            continue
        edges = in_edges.get(node.id, [])
        conditions = sorted({_disease_id(nodes_by_id, e.source) for e in edges})
        label = graph_queries.node_label(node)
        items.append(
            LexicalItem(
                id=f"vocab:{node.id}",
                surface_form=label,
                canonical_form=label,
                concept=category.category,
                concept_id=node.type,
                confidence=graph_queries.max_confidence(edges)
                if category.carries_evidence
                else "asserted",
                source=category.relation,
                ontology_node=node.id,
                dkb_reference=tuple(conditions),
                evidence=graph_queries.union_evidence(edges) if category.carries_evidence else (),
                part_of_speech=category.part_of_speech,
            )
        )
    return items


def _two_hop_category_items(
    category: policies.CategorySpec,
    ontology: Ontology,
    nodes_by_id: dict[str, Node],
    symptom_owner: dict[str, str],
) -> list[LexicalItem]:
    """Items reached via a Symptom (leaf_region/sign_type/observability_modifier).

    The relation targets a closed-vocabulary node from a Symptom; the DKB
    disease association is the symptom's *owning condition*, one hop further.
    These relations never carry evidence (T-Box-level structural facts).
    """
    in_edges = graph_queries.index_in_edges(ontology, category.relation)
    items: list[LexicalItem] = []
    for node in ontology.nodes:
        if node.type not in category.concept_types:
            continue
        edges = in_edges.get(node.id, [])
        conditions = sorted(
            {
                _disease_id(nodes_by_id, symptom_owner[e.source])
                for e in edges
                if e.source in symptom_owner
            }
        )
        label = graph_queries.node_label(node)
        items.append(
            LexicalItem(
                id=f"vocab:{node.id}",
                surface_form=label,
                canonical_form=label,
                concept=category.category,
                concept_id=node.type,
                confidence="asserted",
                source=category.relation,
                ontology_node=node.id,
                dkb_reference=tuple(conditions),
                evidence=(),
                part_of_speech=category.part_of_speech,
            )
        )
    return items


def _disease_name_items(
    category: policies.CategorySpec, ontology: Ontology, nodes_by_id: dict[str, Node]
) -> list[LexicalItem]:
    """Disease/condition identity items; evidence aggregated from all outgoing edges."""
    out_edges = graph_queries.index_out_edges(ontology)
    items: list[LexicalItem] = []
    for node in ontology.nodes:
        if node.type not in category.concept_types:
            continue
        edges = out_edges.get(node.id, [])
        label = graph_queries.node_label(node)
        items.append(
            LexicalItem(
                id=f"vocab:{node.id}",
                surface_form=label,
                canonical_form=label,
                concept=category.category,
                concept_id=node.type,
                confidence=graph_queries.max_confidence(edges),
                source=category.relation,
                ontology_node=node.id,
                dkb_reference=(_disease_id(nodes_by_id, node.id),),
                evidence=graph_queries.union_evidence(edges),
                part_of_speech=category.part_of_speech,
            )
        )
    return items


def _confidence_modifier_items() -> list[LexicalItem]:
    """The 3 fixed confidence values (edge-attribute enum, not ontology nodes)."""
    return [
        LexicalItem(
            id=f"vocab:confidence:{value}",
            surface_form=value,
            canonical_form=value,
            concept=policies.CONFIDENCE_CATEGORY.category,
            concept_id="Confidence",
            confidence=value,
            source=policies.CONFIDENCE_CATEGORY.relation,
            ontology_node="",
            dkb_reference=(),
            evidence=(),
            part_of_speech=policies.CONFIDENCE_CATEGORY.part_of_speech,
        )
        for value in policies.CONFIDENCE_VALUES
    ]


def build_vocabulary(ontology: Ontology) -> list[LexicalItem]:
    """Derive the flat concept vocabulary from a compiled ontology.

    One item per ontology node in a covered category (color, shape, texture,
    extent, leaf_region, sign_type, agent_name, disease_name, environment,
    observability_modifier), plus the 3 fixed confidence-modifier items.
    Sorted by id for deterministic output.
    """
    nodes_by_id = {n.id: n for n in ontology.nodes}
    symptom_owner = graph_queries.index_symptom_owner(ontology)

    items: list[LexicalItem] = []
    for category in policies.CATEGORIES:
        if category.relation == "identity":
            items.extend(_disease_name_items(category, ontology, nodes_by_id))
        elif category.category in ("leaf_region", "sign_type", "observability_modifier"):
            items.extend(_two_hop_category_items(category, ontology, nodes_by_id, symptom_owner))
        else:
            items.extend(_direct_category_items(category, ontology, nodes_by_id))
    items.extend(_confidence_modifier_items())

    items.sort(key=lambda item: item.id)
    return items
