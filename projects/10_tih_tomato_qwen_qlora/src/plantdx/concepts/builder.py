"""Caption Concept Model builder (component A).

Pure function: ``build_concept_models(dkb, ontology, vocabulary) -> ConceptModelSet``.
Deterministic and order-independent — iterate diseases in sorted order, derive
every concept from the DKB (its designed input, doc 01) cross-linked to the
compiled ontology and vocabulary for evidence, confidence, sign types, and
controlled realizations. No randomness, no wall-clock, no image, no LLM/VLM.
"""

from __future__ import annotations

import re
from typing import Any

from plantdx.concepts import policies
from plantdx.concepts.models import (
    STATUS_FORBIDDEN,
    STATUS_MANDATORY,
    STATUS_OPTIONAL,
    CaptionConcept,
    ConceptModel,
    ConceptModelSet,
)
from plantdx.ontology.domain.models import Edge, Node, Ontology
from plantdx.vocabulary.domain import graph_queries
from plantdx.vocabulary.domain.models import VocabularyResult

_CONDITION_TYPES = frozenset({"Disease", "PestDamage", "SurfaceColonization", "HealthyState"})
_PRIMARY_FIELDS = frozenset({"diagnostic_visual_features", "primary_symptoms"})


def build_concept_models(
    dkb: dict[str, Any], ontology: Ontology, vocabulary: VocabularyResult
) -> ConceptModelSet:
    """Derive one :class:`ConceptModel` per disease from the frozen inputs."""
    ctx = _Context(dkb, ontology, vocabulary)
    models = [ctx.build_one(disease) for disease in sorted(dkb["diseases"], key=lambda d: d["id"])]
    return ConceptModelSet(
        disease_models=models,
        provenance={
            "concepts_version": policies.CONCEPTS_VERSION,
            "schema_version": policies.SCHEMA_VERSION,
            "ontology_content_hash": ontology.provenance.get("content_hash", ""),
            "vocabulary_content_hash": vocabulary.provenance.get("content_hash", ""),
            "dkb_sha256": ontology.provenance.get("dkb_sha256", ""),
            "builder": "plantdx.concepts",
        },
    )


class _Context:
    """Precomputed indices shared across all diseases (built once)."""

    def __init__(
        self, dkb: dict[str, Any], ontology: Ontology, vocabulary: VocabularyResult
    ) -> None:
        self.dkb_by_id = {d["id"]: d for d in dkb["diseases"]}
        self.nodes_by_id: dict[str, Node] = {n.id: n for n in ontology.nodes}
        self.out_edges = graph_queries.index_out_edges(ontology)
        self.condition_by_disease: dict[str, Node] = {
            str(n.properties.get("disease_id")): n
            for n in ontology.nodes
            if n.type in _CONDITION_TYPES
        }
        # disease_id -> {concept category -> [vocabulary items]}
        self.vocab_by_disease: dict[str, dict[str, list[Any]]] = {}
        for item in vocabulary.vocabulary_items:
            for disease_id in item.dkb_reference:
                self.vocab_by_disease.setdefault(disease_id, {}).setdefault(
                    item.concept, []
                ).append(item)

    # -- ontology helpers --------------------------------------------------- #

    def _condition(self, disease_id: str) -> Node:
        return self.condition_by_disease[disease_id]

    def _out(self, node_id: str, relation: str) -> list[Edge]:
        return [e for e in self.out_edges.get(node_id, []) if e.type == relation]

    def _symptom_sign_type(self, symptom_id: str) -> str | None:
        for edge in self._out(symptom_id, "has_sign_type"):
            return edge.target.split(":", 1)[1]
        return None

    def _vocab(self, disease_id: str, category: str) -> list[Any]:
        return self.vocab_by_disease.get(disease_id, {}).get(category, [])

    # -- per-disease build -------------------------------------------------- #

    def build_one(self, disease: dict[str, Any]) -> ConceptModel:
        disease_id = disease["id"]
        condition = self._condition(disease_id)
        concepts: dict[str, CaptionConcept] = {}

        primary_sign_type = self._primary_sign_type(condition)
        modifiable = primary_sign_type in policies.MODIFIABLE_SIGN_TYPES

        self._add_identity(concepts, disease, condition)
        self._add_host(concepts, disease, condition)
        self._add_agent(concepts, disease, condition)
        self._add_signs(concepts, disease, condition, primary_sign_type)
        if modifiable:
            self._add_qualities(concepts, disease_id, primary_sign_type)
        self._add_extent(concepts, disease_id)
        self._add_location(concepts, disease_id)
        self._add_dkb_field_concepts(concepts, disease)
        self._add_differential(concepts, disease, condition)

        is_healthy = condition.type == "HealthyState"
        mandatory = self._mandatory(concepts, is_healthy)
        # Finalize statuses now that mandatory is known.
        for cid, concept in concepts.items():
            concepts[cid] = _with_status(
                concept, STATUS_MANDATORY if cid in mandatory else STATUS_OPTIONAL
            )
        forbidden = self._forbidden(concepts, is_healthy)
        for cid in forbidden:
            concepts.setdefault(cid, _forbidden_concept(cid))

        ordered = tuple(c for c in policies.CONCEPT_ORDER if c in concepts)
        optional = tuple(
            c
            for c in policies.CONCEPT_ORDER
            if c in concepts and concepts[c].status == STATUS_OPTIONAL
        )
        return ConceptModel(
            disease_id=disease_id,
            crop=disease["crop"],
            condition_type=condition.type,
            sign_type="healthy" if is_healthy else (primary_sign_type or "none"),
            is_pathogen_disease=bool(disease["is_pathogen_disease"]),
            agent_category=disease["agent_category"],
            register_policy={
                "visual": True,
                "clinical": True,
                "educational": True,
                "severity_conditioned": False,
            },
            mandatory=tuple(c for c in policies.CONCEPT_ORDER if c in mandatory),
            optional=optional,
            forbidden=tuple(sorted(forbidden)),
            ordering=ordered,
            min_information=len(mandatory),
            max_information=len(mandatory) + len(optional),
            concepts=tuple(concepts[c] for c in sorted(concepts)),
            never_appear=self._never_appear(disease, condition),
        )

    # -- concept derivations ------------------------------------------------ #

    def _primary_sign_type(self, condition: Node) -> str | None:
        for edge in self._out(condition.id, "has_symptom"):
            if edge.attributes.get("flags", {}).get("primary"):
                sign = self._symptom_sign_type(edge.target)
                if sign is not None:
                    return sign
        return None

    def _add_identity(
        self, out: dict[str, CaptionConcept], disease: dict[str, Any], condition: Node
    ) -> None:
        # Disease common names are lowercased for running text ("early blight",
        # not "Early Blight") and stripped of spelling/abbreviation parentheticals
        # ("sooty mould (sooty mold)" -> "sooty mould"); the generator capitalizes
        # sentence-initial words.
        surfaces = _dedup_sorted(
            [
                _strip_paren(str(condition.properties.get("canonical_label", "")).lower()),
                _strip_paren(str(disease.get("common_name", "")).lower()),
            ]
        )
        out["disease_identity"] = CaptionConcept(
            concept_id="disease_identity",
            status=STATUS_OPTIONAL,
            observable=True,
            confidence="asserted",
            sign_type=None,
            realizations=surfaces,
            modifiers=(),
            evidence=_all_out_evidence(self, condition.id),
            dkb_fields=("class_label", "common_name", "disease"),
        )

    def _add_host(
        self, out: dict[str, CaptionConcept], disease: dict[str, Any], condition: Node
    ) -> None:
        out["host"] = CaptionConcept(
            concept_id="host",
            status=STATUS_OPTIONAL,
            observable=True,
            confidence="asserted",
            sign_type=None,
            realizations=(f"{disease['crop']} leaf",),
            modifiers=(),
            evidence=_edges_evidence(self._out(condition.id, "affects")),
            dkb_fields=("crop", "host_plant"),
        )

    def _add_agent(
        self, out: dict[str, CaptionConcept], disease: dict[str, Any], condition: Node
    ) -> None:
        caused_by = self._out(condition.id, "caused_by")
        category = disease["agent_category"]
        descriptor = policies.AGENT_CATEGORY_DESCRIPTORS.get(category)
        if descriptor is not None:
            out["agent_category_descriptor"] = CaptionConcept(
                concept_id="agent_category_descriptor",
                status=STATUS_OPTIONAL,
                observable=False,
                confidence="asserted",
                sign_type=None,
                realizations=(descriptor,),
                modifiers=(),
                evidence=_edges_evidence(caused_by) or _disease_evidence(disease),
                dkb_fields=("agent_category",),
            )
        if caused_by:
            agent = self.nodes_by_id.get(caused_by[0].target)
            if agent is not None:
                out["agent_reference"] = CaptionConcept(
                    concept_id="agent_reference",
                    status=STATUS_OPTIONAL,
                    observable=False,
                    confidence="asserted",
                    sign_type=None,
                    realizations=(graph_queries.node_label(agent),),
                    modifiers=(),
                    evidence=_edges_evidence(caused_by),
                    dkb_fields=("scientific_name",),
                )

    def _add_signs(
        self,
        out: dict[str, CaptionConcept],
        disease: dict[str, Any],
        condition: Node,
        primary_sign_type: str | None,
    ) -> None:
        primary: list[str] = []
        primary_ev: list[Edge] = []
        secondary: list[str] = []
        secondary_ev: list[Edge] = []
        healthy: list[str] = []
        healthy_ev: list[Edge] = []
        for edge in self._out(condition.id, "has_symptom"):
            symptom = self.nodes_by_id[edge.target]
            text = _symptom_text(symptom)
            field = symptom.properties.get("source_field")
            sign = self._symptom_sign_type(edge.target)
            if sign == "healthy_surface":
                healthy.append(text)
                healthy_ev.append(edge)
            elif field in _PRIMARY_FIELDS:
                primary.append(text)
                primary_ev.append(edge)
            elif field == "secondary_symptoms":
                secondary.append(text)
                secondary_ev.append(edge)
        # Primary-sign realizations must read as noun phrases (a template says
        # "showing {primary}" / "{primary} can be seen"); DKB clauses like "begin
        # on oldest leaves" or "forms a film" are dropped. Fall back to the full
        # set only if filtering would leave the mandatory concept empty.
        primary_np = [p for p in primary if _is_noun_phrase(p)]
        if primary:
            out["primary_sign"] = CaptionConcept(
                concept_id="primary_sign",
                status=STATUS_MANDATORY,
                observable=True,
                confidence="asserted",
                sign_type=primary_sign_type,
                realizations=_dedup_sorted(primary_np or primary),
                modifiers=self._modifier_values(disease["id"], primary_sign_type),
                evidence=_edges_evidence(primary_ev),
                dkb_fields=("diagnostic_visual_features", "primary_symptoms"),
            )
        # Secondary signs are hedged ("may later develop {secondary}"); require a
        # noun phrase and drop any phrase carrying a severity-stage token so the
        # caption is never generated-then-rejected (V-CAP-11 stays as defense in depth).
        secondary_np = [s for s in secondary if _is_noun_phrase(s) and not _has_stage_token(s)]
        if secondary_np:
            out["secondary_sign"] = CaptionConcept(
                concept_id="secondary_sign",
                status=STATUS_OPTIONAL,
                observable=True,
                confidence="hedged",
                sign_type=None,
                realizations=_dedup_sorted(secondary_np),
                modifiers=(),
                evidence=_edges_evidence(secondary_ev),
                dkb_fields=("secondary_symptoms",),
            )
        if healthy:
            out["healthy_state"] = CaptionConcept(
                concept_id="healthy_state",
                status=STATUS_MANDATORY,
                observable=True,
                confidence="asserted",
                sign_type="healthy",
                realizations=self._healthy_observations(disease, healthy),
                modifiers=(),
                evidence=_edges_evidence(healthy_ev) or _disease_evidence(disease),
                dkb_fields=(
                    "diagnostic_visual_features",
                    "primary_symptoms",
                    "texture_changes",
                    "leaf_margin_changes",
                ),
            )

    def _healthy_observations(
        self, disease: dict[str, Any], fallback: list[str]
    ) -> tuple[str, ...]:
        """Multiple atomic, evidence-supported healthy observations from DKB fields.

        Pulls the clean noun-phrase descriptors of a healthy leaf (uniform green
        lamina, intact margins, glossy surface, ...) from the DKB's healthy fields
        so a healthy caption has real variety instead of one repeated mouthful.
        Every phrase is a verbatim DKB observation — nothing is invented.
        """
        phrases: list[str] = [_strip_paren(p) for p in fallback]
        for field_name in (
            "diagnostic_visual_features",
            "primary_symptoms",
            "texture_changes",
            "leaf_margin_changes",
        ):
            for phrase in _clean_phrases(disease.get(field_name)):
                stripped = _strip_paren(phrase)
                if _is_noun_phrase(stripped):
                    phrases.append(stripped)
        return _dedup_sorted(phrases)

    def _modifier_values(self, disease_id: str, primary_sign_type: str | None) -> tuple[str, ...]:
        if primary_sign_type not in policies.MODIFIABLE_SIGN_TYPES:
            return ()
        values: list[str] = []
        for category in ("color", "shape", "texture", "extent"):
            values.extend(_strip_paren(i.surface_form) for i in self._vocab(disease_id, category))
        return _dedup_sorted(values)

    def _add_qualities(
        self, out: dict[str, CaptionConcept], disease_id: str, primary_sign_type: str | None
    ) -> None:
        specs = (
            ("lesion_color", "color", ("color_vocabulary", "leaf_color")),
            ("lesion_shape", "shape", ("shape_vocabulary", "lesion_shape")),
            ("texture", "texture", ("texture_vocabulary", "texture_changes")),
        )
        for concept_id, category, fields in specs:
            items = self._vocab(disease_id, category)
            if not items:
                continue
            out[concept_id] = CaptionConcept(
                concept_id=concept_id,
                status=STATUS_OPTIONAL,
                observable=True,
                confidence="typical",
                sign_type=primary_sign_type,
                realizations=_dedup_sorted([_strip_paren(i.surface_form) for i in items]),
                modifiers=(),
                evidence=_items_evidence(items),
                dkb_fields=fields,
            )

    def _add_extent(self, out: dict[str, CaptionConcept], disease_id: str) -> None:
        items = self._vocab(disease_id, "extent")
        realizations = _dedup_sorted(
            [_strip_paren(i.surface_form) for i in items if not _has_stage_token(i.surface_form)]
        )
        if realizations:
            out["extent"] = CaptionConcept(
                concept_id="extent",
                status=STATUS_OPTIONAL,
                observable=True,
                confidence="typical",
                sign_type=None,
                realizations=realizations,
                modifiers=(),
                evidence=_items_evidence(items),
                dkb_fields=("severity_vocabulary",),
            )

    def _add_location(self, out: dict[str, CaptionConcept], disease_id: str) -> None:
        items = self._vocab(disease_id, "leaf_region")
        if items:
            out["leaf_location"] = CaptionConcept(
                concept_id="leaf_location",
                status=STATUS_OPTIONAL,
                observable=True,
                confidence="typical",
                sign_type=None,
                realizations=_dedup_sorted([i.surface_form.replace("_", " ") for i in items]),
                modifiers=(),
                evidence=_items_evidence(items),
                dkb_fields=("lesion_distribution", "leaf_margin_changes"),
            )

    def _add_dkb_field_concepts(
        self, out: dict[str, CaptionConcept], disease: dict[str, Any]
    ) -> None:
        evidence = _disease_evidence(disease)
        for field_name, concept_id in policies.DKB_FIELD_CONCEPTS:
            phrases = _clean_phrases(disease.get(field_name))
            if not phrases:
                continue
            out[concept_id] = CaptionConcept(
                concept_id=concept_id,
                status=STATUS_OPTIONAL,
                observable=True,
                confidence="typical",
                sign_type=None,
                realizations=_dedup_sorted(phrases),
                modifiers=(),
                evidence=evidence,
                dkb_fields=(field_name,),
            )

    def _add_differential(
        self, out: dict[str, CaptionConcept], disease: dict[str, Any], condition: Node
    ) -> None:
        edges = self._out(condition.id, "differentiated_from")
        rivals = _dedup_sorted(
            [
                _strip_paren(_rival_label(self.nodes_by_id[e.target]).lower())
                for e in edges
                if e.target in self.nodes_by_id
            ]
        )
        if rivals:
            out["differential"] = CaptionConcept(
                concept_id="differential",
                status=STATUS_OPTIONAL,
                observable=True,
                confidence="typical",
                sign_type=None,
                realizations=rivals,
                modifiers=(),
                evidence=_edges_evidence(edges) or _disease_evidence(disease),
                dkb_fields=("confused_with", "key_differentiating_features"),
            )

    # -- status resolution -------------------------------------------------- #

    def _mandatory(self, concepts: dict[str, CaptionConcept], is_healthy: bool) -> set[str]:
        if is_healthy:
            return {c for c in ("disease_identity", "healthy_state") if c in concepts}
        return {c for c in ("disease_identity", "primary_sign") if c in concepts}

    def _forbidden(self, concepts: dict[str, CaptionConcept], is_healthy: bool) -> set[str]:
        forbidden = set(policies.ALWAYS_FORBIDDEN)
        if is_healthy:
            # A healthy leaf may not assert any disease sign.
            forbidden |= {
                c
                for c in policies.CONCEPT_ORDER
                if c not in ("host", "disease_identity", "healthy_state")
            }
        else:
            forbidden.add("healthy_state")
        return forbidden - set(concepts)

    def _never_appear(self, disease: dict[str, Any], condition: Node) -> tuple[str, ...]:
        terms: list[str] = []
        # 1. Non-observable symptom surfaces (fruit/twig/etc.) owned by this disease.
        for edge in self._out(condition.id, "has_symptom"):
            symptom = self.nodes_by_id[edge.target]
            if symptom.properties.get("observable") is False:
                terms.append(_symptom_text(symptom))
        # 2. Rival diseases' primary-sign hallmark surfaces (cross-disease leakage).
        for edge in self._out(condition.id, "differentiated_from"):
            rival = self.nodes_by_id.get(edge.target)
            if rival is None:
                continue
            for rival_symptom_edge in self._out(rival.id, "has_symptom"):
                if rival_symptom_edge.attributes.get("flags", {}).get("primary"):
                    sym = self.nodes_by_id[rival_symptom_edge.target]
                    terms.append(_symptom_text(sym))
        # 3. DKB-authored forbidden terms/adjectives.
        terms.extend(_clean_phrases(disease.get("forbidden_terms")))
        terms.extend(_clean_phrases(disease.get("forbidden_adjectives")))
        # 4. Severity-stage tokens (severity_conditioned is False).
        terms.extend(policies.STAGE_TOKENS)
        return _dedup_sorted(terms)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _with_status(concept: CaptionConcept, status: str) -> CaptionConcept:
    if concept.status == status:
        return concept
    return CaptionConcept(
        concept_id=concept.concept_id,
        status=status,
        observable=concept.observable,
        confidence=concept.confidence,
        sign_type=concept.sign_type,
        realizations=concept.realizations,
        modifiers=concept.modifiers,
        evidence=concept.evidence,
        dkb_fields=concept.dkb_fields,
    )


def _forbidden_concept(concept_id: str) -> CaptionConcept:
    return CaptionConcept(
        concept_id=concept_id,
        status=STATUS_FORBIDDEN,
        observable=concept_id not in policies.NON_OBSERVABLE_CONCEPTS,
        confidence="asserted",
        sign_type=None,
        realizations=(),
        modifiers=(),
        evidence=(),
        dkb_fields=(),
    )


def _symptom_text(symptom: Node) -> str:
    """The verbatim DKB phrase for a symptom node (falls back to its canonical label)."""
    return str(symptom.properties.get("source_text", symptom.properties["canonical_label"]))


def _rival_label(node: Node) -> str:
    """The display label of a rival condition node (for differential realizations)."""
    return str(node.properties.get("canonical_label", node.id))


# First words / markers that make a symptom phrase a clause rather than a noun
# phrase; such phrases cannot fill a "{primary} can be seen" slot grammatically.
_CLAUSE_LEADERS = frozenset(
    {
        "begin",
        "begins",
        "form",
        "forms",
        "forming",
        "often",
        "may",
        "might",
        "can",
        "appear",
        "appears",
        "develop",
        "develops",
        "start",
        "starts",
        "cause",
        "causes",
        "produce",
        "produces",
        "occur",
        "occurs",
        "turn",
        "turns",
        "become",
        "becomes",
        "tend",
        "tends",
        "affect",
        "affects",
        "spread",
        "spreads",
        "senesce",
        "senesces",
        "enlarge",
        "enlarges",
        "coalesce",
        "coalesces",
        "progress",
        "progresses",
        "expand",
        "expands",
        "first",
        "later",
        "when",
        "as",
        "if",
        "while",
    }
)
_CLAUSE_MARKERS = ("may be visible", "are visible", "is visible", "may be present", "may develop")
# Bare finite verbs that, when they END a phrase, mark it as a clause ("young
# leaves distort", "affected leaflets distort as they senesce").
_TRAILING_VERBS = frozenset(
    {
        "distort",
        "distorts",
        "curl",
        "curls",
        "drop",
        "drops",
        "wilt",
        "wilts",
        "senesce",
        "senesces",
        "coalesce",
        "coalesces",
        "die",
        "dies",
        "shrivel",
        "shrivels",
        "yellow",
        "yellows",
        "necrose",
        "necroses",
        "collapse",
        "collapses",
    }
)


def _is_noun_phrase(phrase: str) -> bool:
    """Deterministic check that a symptom phrase reads as a noun phrase, not a clause.

    Rejects phrases beginning with a verb or adverb (a leading ``-ly`` adverb or a
    known clause verb), ending in a bare finite verb ("young leaves distort"), or
    containing a finite-verb marker — none of which can fill a "showing {primary}"
    / "{primary} can be seen" slot grammatically.
    """
    words = phrase.strip().lower().split()
    if not words:
        return False
    first = words[0].strip(",;:")
    if first in _CLAUSE_LEADERS or (first.endswith("ly") and len(first) > 3):
        return False
    if words[-1].strip(".,;:") in _TRAILING_VERBS:
        return False
    low = phrase.lower()
    return not any(marker in low for marker in _CLAUSE_MARKERS)


def _strip_paren(text: str) -> str:
    """Drop parenthetical disambiguation notes from a controlled quality value.

    DKB quality vocabulary annotates some values with a bracketed note ("yellow
    (halo)", "reddish (early)", "black (necrotic galls)") that disambiguates the
    DKB entry but reads badly in a caption. Applied only to short quality-axis
    values (color/shape/texture/extent), never to primary signs or agent names.
    """
    return re.sub(r"\s+", " ", re.sub(r"\s*\([^)]*\)", "", text)).strip()


def _has_stage_token(text: str) -> bool:
    """Whether a phrase contains a severity-stage token (word-boundary match)."""
    low = text.lower()
    return any(re.search(rf"(?<!\w){re.escape(t)}(?!\w)", low) for t in policies.STAGE_TOKENS)


def _dedup_sorted(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted({v.strip() for v in values if v and v.strip()}))


def _clean_phrases(value: Any) -> list[str]:
    """DKB field value -> caption-usable phrases, dropping absence/negation values.

    A phrase whose first word denotes absence ("none characteristic", "minimal;
    ...", "no visible ...") is treated as "this feature is not present for this
    disease" and dropped, so a caption never asserts a negated feature. This is a
    deterministic first-token check, not fuzzy NLP.
    """
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if (
            text
            and text.lower() not in policies.NA_VALUES
            and not _is_negation(text)
            and not _has_stage_token(text)
        ):
            out.append(text)
    return out


def _is_negation(text: str) -> bool:
    """Whether ``text`` begins with an absence/negation token (feature not present)."""
    first = ""
    for ch in text.lower():
        if ch.isalnum() or ch == "/":
            first += ch
        else:
            break
    return first in policies.NEGATION_TOKENS


def _edges_evidence(edges: list[Edge]) -> tuple[str, ...]:
    return graph_queries.union_evidence(edges)


def _items_evidence(items: list[Any]) -> tuple[str, ...]:
    ids: set[str] = set()
    for item in items:
        ids.update(item.evidence)
    return tuple(sorted(ids))


def _all_out_evidence(ctx: _Context, node_id: str) -> tuple[str, ...]:
    return graph_queries.union_evidence(ctx.out_edges.get(node_id, []))


def _disease_evidence(disease: dict[str, Any]) -> tuple[str, ...]:
    refs = disease.get("references", {})
    keys: set[str] = set()
    for group in ("recent_research", "extension_service", "textbook"):
        keys.update(refs.get(group, []))
    return tuple(sorted(f"evidence:{k}" for k in keys))
