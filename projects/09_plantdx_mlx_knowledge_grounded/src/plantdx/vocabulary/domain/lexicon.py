"""Symptom Lexicon Builder — component (C): bounded symptom realizations.

Every realization is a *legal combination permitted by the ontology* — never a
Cartesian product. Two kinds of item, both grounded in a single Symptom node:

* **base** — the symptom's own DKB phrase, verbatim (always emitted; zero risk).
* **single-modifier** — the symptom's sign-type head noun ("lesion", "coating",
  "gall", "stippling" — Caption Framework 01 §2.4's co-selection set) combined
  with exactly one quality value already attached to the *owning condition*
  (``has_color``/``has_shape``/``has_texture``/``has_extent``). One item per
  attached value: linear in modifier count, never multiplied across axes.
  Only emitted for *primary* symptoms (``has_symptom`` flag ``primary=True``).

Multi-modifier stacking ("circular brown lesion") is the future Vocabulary
Expander's (component F) job, not this builder's — out of scope here.

Some DKB conditions reuse the same descriptive word across two quality axes
for the same condition (e.g. mango_gall_midge has both ``shape:raised`` and
``texture:raised``) — legitimate, distinct ontology facts that would
otherwise realize as the same surface phrase twice for one symptom. When
that happens only the highest-priority axis is kept, in ``policies.
MODIFIER_RELATIONS`` order (color, then shape, then texture, then extent),
so every symptom's realization set stays duplicate-free without discarding
any ontology fact silently — the dropped axis is still present verbatim in
``vocabulary.json`` under its own concept node.
"""

from __future__ import annotations

from plantdx.ontology.domain.models import Edge, Node, Ontology
from plantdx.vocabulary.domain import graph_queries, policies
from plantdx.vocabulary.domain.models import LexicalItem


def build_lexicon(ontology: Ontology) -> list[LexicalItem]:
    """Derive the bounded symptom lexicon from a compiled ontology."""
    nodes_by_id: dict[str, Node] = {n.id: n for n in ontology.nodes}
    has_sign_type = {e.source: e.target for e in ontology.edges if e.type == "has_sign_type"}
    condition_out_edges = graph_queries.index_out_edges(ontology)

    items: list[LexicalItem] = []
    for edge in sorted(
        (e for e in ontology.edges if e.type == "has_symptom"),
        key=lambda e: (e.source, e.target),
    ):
        condition_id, symptom_id = edge.source, edge.target
        symptom = nodes_by_id[symptom_id]
        disease_id = str(nodes_by_id[condition_id].properties.get("disease_id", condition_id))

        items.append(_base_item(symptom, edge, disease_id))

        is_primary = bool(edge.attributes.get("flags", {}).get("primary", False))
        sign_type_id = has_sign_type.get(symptom_id)
        sign_type = sign_type_id.split(":", 1)[1] if sign_type_id else None
        if is_primary and sign_type in policies.MODIFIABLE_SIGN_TYPES and sign_type_id:
            head_noun = graph_queries.node_label(nodes_by_id[sign_type_id])
            candidate_edges = condition_out_edges.get(condition_id, [])
            modifier_edges = sorted(
                (e for e in candidate_edges if e.type in policies.MODIFIER_RELATIONS),
                key=lambda e: (policies.MODIFIER_RELATIONS.index(e.type), e.target),
            )
            seen_phrases: set[str] = set()
            for modifier_edge in modifier_edges:
                modifier_label = graph_queries.node_label(nodes_by_id[modifier_edge.target])
                phrase = f"{modifier_label} {head_noun}"
                if phrase in seen_phrases:
                    continue
                seen_phrases.add(phrase)
                items.append(
                    _modifier_item(
                        symptom,
                        edge,
                        modifier_edge,
                        nodes_by_id[modifier_edge.target],
                        head_noun,
                        disease_id,
                    )
                )

    items.sort(key=lambda item: item.id)
    return items


def _base_item(symptom: Node, has_symptom_edge: Edge, disease_id: str) -> LexicalItem:
    canonical = graph_queries.node_label(symptom)
    surface = str(symptom.properties.get("source_text", canonical))
    return LexicalItem(
        id=f"lex:{symptom.id}:base",
        surface_form=surface,
        canonical_form=canonical,
        concept="symptom_realization",
        concept_id="Symptom",
        confidence=str(has_symptom_edge.attributes.get("confidence", "asserted")),
        source="has_symptom",
        ontology_node=symptom.id,
        dkb_reference=(disease_id,),
        evidence=tuple(sorted(has_symptom_edge.attributes.get("evidence", []))),
        part_of_speech="phrase",
    )


def _modifier_item(
    symptom: Node,
    has_symptom_edge: Edge,
    modifier_edge: Edge,
    modifier_node: Node,
    head_noun: str,
    disease_id: str,
) -> LexicalItem:
    modifier_label = graph_queries.node_label(modifier_node)
    phrase = f"{modifier_label} {head_noun}"
    confidence = graph_queries.min_confidence(
        graph_queries.max_confidence([has_symptom_edge]),
        graph_queries.max_confidence([modifier_edge]),
    )
    return LexicalItem(
        id=f"lex:{symptom.id}:mod:{modifier_node.id}",
        surface_form=phrase,
        canonical_form=phrase,
        concept="symptom_realization",
        concept_id="Symptom",
        confidence=confidence,
        source=modifier_edge.type,
        ontology_node=symptom.id,
        dkb_reference=(disease_id,),
        evidence=graph_queries.union_evidence([has_symptom_edge, modifier_edge]),
        part_of_speech="phrase",
    )
