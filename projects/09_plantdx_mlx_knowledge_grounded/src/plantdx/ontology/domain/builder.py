"""The deterministic ontology compiler: ``Ontology = f(DKB, Policies)``.

Pure and order-independent: iterate everything in sorted order, derive ids from
stable keys, and never use randomness or timestamps. See ontology_design/10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plantdx.core.exceptions import KnowledgeBaseError
from plantdx.ontology.domain import checksum, policies
from plantdx.ontology.domain.graph import ConceptGraph
from plantdx.ontology.domain.models import Edge, Ontology
from plantdx.utils.hashing import sha256_bytes
from plantdx.utils.io import read_json

_REQUIRED_DISEASE_KEYS = frozenset(
    {"id", "crop", "class_label", "is_pathogen_disease", "agent_category", "references"}
)
_NA_SLUGS = frozenset({"", "not_applicable", "none", "n_a", "na"})
_EVIDENCE_TIER_PRIORITY = {"peer_reviewed": 3, "extension_service": 2, "textbook": 1}
_EVIDENCE_TYPE = {
    "peer_reviewed": "PeerReviewed",
    "extension_service": "ExtensionService",
    "textbook": "Textbook",
}


# --------------------------------------------------------------------------- #
# DKB loading and validation
# --------------------------------------------------------------------------- #


def load_dkb(path: str | Path) -> dict[str, Any]:
    """Read the DKB JSON into a dict (raises KnowledgeBaseError if unreadable)."""
    path = Path(path)
    if not path.is_file():
        raise KnowledgeBaseError(f"DKB not found: {path}")
    try:
        data = read_json(path)
    except Exception as exc:
        raise KnowledgeBaseError(f"DKB is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise KnowledgeBaseError(f"DKB root must be an object: {path}")
    return data


def dkb_file_sha256(path: str | Path) -> str:
    """SHA-256 of the DKB file bytes (pins the source of truth in provenance)."""
    return sha256_bytes(Path(path).read_bytes())


def validate_dkb(dkb: dict[str, Any]) -> None:
    """Fail closed on a malformed or empty DKB."""
    diseases = dkb.get("diseases")
    if not isinstance(diseases, list) or not diseases:
        raise KnowledgeBaseError("DKB has no diseases")
    registry = dkb.get("metadata", {}).get("reference_registry")
    if not isinstance(registry, dict):
        raise KnowledgeBaseError("DKB metadata.reference_registry missing")
    seen_ids: set[str] = set()
    for i, disease in enumerate(diseases):
        if not isinstance(disease, dict):
            raise KnowledgeBaseError(f"DKB disease #{i} is not an object")
        missing = _REQUIRED_DISEASE_KEYS - disease.keys()
        if missing:
            raise KnowledgeBaseError(
                f"DKB disease #{i} ({disease.get('id', '?')}) missing fields: {sorted(missing)}"
            )
        if disease["id"] in seen_ids:
            raise KnowledgeBaseError(f"DKB duplicate disease id: {disease['id']}")
        seen_ids.add(disease["id"])


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_ontology(dkb: dict[str, Any], dkb_sha256: str) -> tuple[Ontology, list[str]]:
    """Compile the DKB into an :class:`Ontology`; return it plus a build log."""
    graph = ConceptGraph()
    log: list[str] = ["stage: load closed vocabularies"]

    _instantiate_closed_vocabularies(graph)
    log.append("stage: instantiate evidence registry")
    tiers = _evidence_tiers(dkb)
    _instantiate_evidence(graph, dkb["metadata"]["reference_registry"], tiers)

    crop_index = _crop_condition_index(dkb)
    log.append(f"stage: build conditions (n={len(dkb['diseases'])})")
    for disease in sorted(dkb["diseases"], key=lambda d: d["id"]):
        _build_condition(graph, disease, crop_index)
        log.append(
            f"  condition {disease['id']}: "
            f"{len(graph.out_edges('condition:' + disease['id'], 'has_symptom'))} symptoms"
        )

    nodes = graph.nodes()
    edges = graph.edges()
    ontology = Ontology(
        schema_version=policies.SCHEMA_VERSION,
        ontology_version=policies.ONTOLOGY_VERSION,
        concept_types=list(policies.CONCEPT_TYPES),
        relation_types=list(policies.RELATION_TYPES),
        nodes=nodes,
        edges=edges,
        provenance={
            "dkb_sha256": dkb_sha256,
            "builder": "plantdx.ontology.domain",
        },
    )
    ontology.provenance["schema_hash"] = checksum.schema_hash(ontology)
    ontology.provenance["content_hash"] = checksum.content_hash(ontology)
    log.append(f"stage: complete (nodes={len(nodes)}, edges={len(edges)})")
    return ontology, log


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def _instantiate_closed_vocabularies(graph: ConceptGraph) -> None:
    for sign in policies.SIGN_TYPES:
        graph.upsert_node(f"sign:{sign}", "SignType", {"canonical_label": sign})
    for region in policies.LEAF_REGIONS:
        graph.upsert_node(
            f"region:{region}", "LeafRegion", {"canonical_label": region, "observable": True}
        )
    for part in policies.NON_LEAF_PARTS:
        graph.upsert_node(
            f"part:{part}", "PlantPart", {"canonical_label": part, "observable": False}
        )
    for stage in policies.SEVERITY_STAGES:
        graph.upsert_node(f"severity:{stage}", "Severity", {"canonical_label": stage})
    for value in policies.OBSERVABILITY:
        graph.upsert_node(f"observability:{value}", "Observability", {"canonical_label": value})
    for category in policies.AGENT_CATEGORIES:
        graph.upsert_node(f"agentcat:{category}", "AgentCategory", {"canonical_label": category})


def _evidence_tiers(dkb: dict[str, Any]) -> dict[str, str]:
    """Key -> tier, taking the highest-priority tier across all references (deterministic)."""
    best: dict[str, str] = {}
    for disease in sorted(dkb["diseases"], key=lambda d: d["id"]):
        refs = disease.get("references", {})
        for group, tier in (
            ("recent_research", "peer_reviewed"),
            ("extension_service", "extension_service"),
            ("textbook", "textbook"),
        ):
            for key in refs.get(group, []):
                if _EVIDENCE_TIER_PRIORITY[tier] > _EVIDENCE_TIER_PRIORITY.get(
                    best.get(key, ""), 0
                ):
                    best[key] = tier
    return best


def _instantiate_evidence(
    graph: ConceptGraph, registry: dict[str, Any], tiers: dict[str, str]
) -> None:
    for key in sorted(registry):
        tier = tiers.get(key, "textbook")
        graph.upsert_node(
            f"evidence:{key}",
            _EVIDENCE_TYPE[tier],
            {
                "citation": registry[key].get("citation", ""),
                "url": registry[key].get("url", ""),
                "tier": tier,
            },
        )


def _crop_condition_index(dkb: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """Crop -> [(condition_id, class_label_lower)] for differential matching."""
    index: dict[str, list[tuple[str, str]]] = {}
    for disease in dkb["diseases"]:
        index.setdefault(disease["crop"], []).append(
            (f"condition:{disease['id']}", disease["class_label"].lower())
        )
    for crop in index:
        index[crop].sort()
    return index


def _build_condition(
    graph: ConceptGraph, d: dict[str, Any], crop_index: dict[str, list[tuple[str, str]]]
) -> None:
    subtype = policies.condition_subtype(d["is_pathogen_disease"], d["agent_category"])
    cond_id = f"condition:{d['id']}"
    graph.upsert_node(
        cond_id,
        subtype,
        {
            "canonical_label": d["class_label"],
            "disease_id": d["id"],
            "crop": d["crop"],
            "disease_name": d.get("disease", ""),
            "common_name": d.get("common_name", ""),
            "is_pathogen_disease": d["is_pathogen_disease"],
            "agent_category": d["agent_category"],
            "scientific_name": d.get("scientific_name", ""),
            "pathogen_type": d.get("pathogen_type", ""),
            "taxonomy_note": d.get("taxonomy_note", ""),
            "host_plant": d.get("host_plant", ""),
            "dataset": d.get("dataset", ""),
            "disease_progression": d.get("disease_progression", ""),
        },
    )
    evidence = _evidence_ids(d)

    crop_id = f"crop:{policies.slug(d['crop'])}"
    graph.upsert_node(crop_id, "Crop", {"canonical_label": d["crop"]})
    _edge(graph, "affects", cond_id, crop_id, confidence="asserted", evidence=evidence)

    if subtype != "HealthyState":
        _build_agent(graph, d, cond_id, subtype, evidence)
        _build_symptoms(graph, d, cond_id, subtype, evidence)
    else:
        _build_healthy_symptom(graph, d, cond_id, evidence)

    _build_qualities(graph, d, cond_id, evidence)
    _build_severity(graph, d, cond_id, evidence)
    _build_differentials(graph, d, cond_id, evidence, crop_index)
    _build_environment(graph, d, cond_id, evidence)


def _build_agent(
    graph: ConceptGraph, d: dict[str, Any], cond_id: str, subtype: str, evidence: list[str]
) -> None:
    agent_type = policies.AGENT_TYPE_BY_CATEGORY[d["agent_category"]]
    agent_id = f"agent:{policies.slug(d['scientific_name'])}"
    graph.upsert_node(
        agent_id,
        agent_type,
        {
            "scientific_name": d["scientific_name"],
            "synonyms": sorted(d.get("scientific_name_synonyms", [])),
            "pathogen_type": d.get("pathogen_type", ""),
        },
    )
    _edge(graph, "agent_in_category", agent_id, f"agentcat:{d['agent_category']}")
    family = d.get("pathogen_family", "").strip()
    if family and family not in ("N/A", "None") and policies.is_subtype(agent_type, "Pathogen"):
        family_id = f"family:{policies.slug(family)}"
        graph.upsert_node(family_id, "PathogenFamily", {"canonical_label": family})
        _edge(graph, "member_of_family", agent_id, family_id, evidence=evidence)
    _edge(
        graph,
        "caused_by",
        cond_id,
        agent_id,
        confidence="asserted",
        evidence=evidence,
        flags={"disputed": False},
    )


def _build_symptoms(
    graph: ConceptGraph, d: dict[str, Any], cond_id: str, subtype: str, evidence: list[str]
) -> None:
    seen: dict[str, str] = {}  # canonical label -> symptom id (dedup within condition, rule R1)
    for field_name, abbr, confidence in policies.SYMPTOM_FIELDS:
        is_forbidden = field_name == policies.FORBIDDEN_FIELD
        observable = not is_forbidden
        for i, phrase in enumerate(d.get(field_name, [])):
            if not phrase.strip():
                continue
            label = _canon(phrase)
            sid = seen.get(label)
            if sid is None:
                sid = f"symptom:{d['id']}:{abbr}:{i}"
                seen[label] = sid
                graph.upsert_node(
                    sid,
                    "Symptom",
                    {
                        "canonical_label": label,
                        "source_text": phrase.strip(),
                        "source_field": field_name,
                        "observable": observable,
                    },
                )
                sign = policies.classify_sign_type(phrase, subtype, d["agent_category"])
                _edge(graph, "has_sign_type", sid, f"sign:{sign}")
                obs_value = "observable" if observable else "non_observable"
                _edge(graph, "has_observability", sid, f"observability:{obs_value}")
                for region in _match_leaf_regions(phrase):
                    _edge(graph, "appears_on", sid, f"region:{region}")
                if is_forbidden:
                    for part in _match_non_leaf(phrase):
                        _edge(graph, "appears_on", sid, f"part:{part}")
            _edge(
                graph,
                "has_symptom",
                cond_id,
                sid,
                confidence=confidence,
                evidence=evidence,
                flags={"primary": field_name in policies.PRIMARY_FIELDS},
            )


def _build_healthy_symptom(
    graph: ConceptGraph, d: dict[str, Any], cond_id: str, evidence: list[str]
) -> None:
    """A HealthyState has exactly one healthy_surface symptom (rules C2/F3)."""
    source = d.get("primary_symptoms") or [d.get("class_label", "healthy leaf")]
    sid = f"symptom:{d['id']}:healthy:0"
    graph.upsert_node(
        sid,
        "Symptom",
        {
            "canonical_label": "healthy leaf surface",
            "source_text": source[0].strip(),
            "source_field": "primary_symptoms",
            "observable": True,
        },
    )
    _edge(graph, "has_sign_type", sid, "sign:healthy_surface")
    _edge(graph, "has_observability", sid, "observability:observable")
    _edge(
        graph,
        "has_symptom",
        cond_id,
        sid,
        confidence="asserted",
        evidence=evidence,
        flags={"primary": True},
    )


def _build_qualities(
    graph: ConceptGraph, d: dict[str, Any], cond_id: str, evidence: list[str]
) -> None:
    for axis, field_name in policies.QUALITY_AXES:
        concept_type = axis.capitalize()  # Color / Shape / Texture
        relation = f"has_{axis}"
        for term in d.get(field_name, []):
            if not term.strip():
                continue
            value_id = f"{axis}:{policies.slug(term)}"
            graph.upsert_node(value_id, concept_type, {"canonical_label": _canon(term)})
            _edge(graph, relation, cond_id, value_id, confidence="typical", evidence=evidence)


def _build_severity(
    graph: ConceptGraph, d: dict[str, Any], cond_id: str, evidence: list[str]
) -> None:
    for term in d.get("severity_vocabulary", []):
        if policies.slug(term) in _NA_SLUGS:
            continue
        extent_id = f"extent:{policies.slug(term)}"
        graph.upsert_node(extent_id, "Extent", {"canonical_label": _canon(term)})
        _edge(
            graph,
            "has_extent",
            cond_id,
            extent_id,
            confidence="typical",
            evidence=evidence,
            flags={"image_licensed": True},
        )
    severity = d.get("severity", {})
    for stage in policies.SEVERITY_STAGES:
        values = severity.get(stage, [])
        if values and not all(policies.slug(v) in _NA_SLUGS for v in values):
            _edge(
                graph,
                "typical_at_severity",
                cond_id,
                f"severity:{stage}",
                confidence="typical",
                evidence=evidence,
                flags={"image_licensed": False},
            )


def _build_differentials(
    graph: ConceptGraph,
    d: dict[str, Any],
    cond_id: str,
    evidence: list[str],
    crop_index: dict[str, list[tuple[str, str]]],
) -> None:
    others = [(cid, label) for cid, label in crop_index.get(d["crop"], []) if cid != cond_id]
    for phrase in d.get("confused_with", []):
        text = phrase.lower()
        for other_id, other_label in others:
            if other_label and other_label in text:
                _edge(
                    graph,
                    "differentiated_from",
                    cond_id,
                    other_id,
                    confidence="asserted",
                    evidence=evidence,
                    note=phrase.strip(),
                )


def _build_environment(
    graph: ConceptGraph, d: dict[str, Any], cond_id: str, evidence: list[str]
) -> None:
    for phrase in d.get("environmental_conditions", []):
        if not phrase.strip():
            continue
        env_id = f"env:{policies.slug(phrase)}"
        graph.upsert_node(env_id, "EnvironmentalCondition", {"canonical_label": _canon(phrase)})
        _edge(graph, "favored_by", cond_id, env_id, confidence="typical", evidence=evidence)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _edge(
    graph: ConceptGraph,
    relation: str,
    source: str,
    target: str,
    *,
    confidence: str | None = None,
    evidence: list[str] | None = None,
    flags: dict[str, bool] | None = None,
    note: str | None = None,
) -> None:
    attributes: dict[str, Any] = {}
    if confidence is not None:
        attributes["confidence"] = confidence
    if evidence is not None:
        attributes["evidence"] = sorted(evidence)
    if flags is not None:
        attributes["flags"] = dict(flags)
    if note is not None:
        attributes["note"] = note
    graph.add_edge(Edge(f"e:{source}:{relation}:{target}", relation, source, target, attributes))


def _evidence_ids(d: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    refs = d.get("references", {})
    for group in ("recent_research", "extension_service", "textbook"):
        keys.update(refs.get(group, []))
    return sorted(f"evidence:{key}" for key in keys)


def _canon(text: str) -> str:
    """Human-readable canonical label: lowercase, whitespace-normalized."""
    return " ".join(text.strip().lower().split())


def _match_leaf_regions(phrase: str) -> list[str]:
    text = phrase.lower()
    found: list[str] = []
    for keyword, region in policies.LEAF_REGION_KEYWORDS:
        if keyword in text and region not in found:
            found.append(region)
    return sorted(found)


def _match_non_leaf(phrase: str) -> list[str]:
    text = phrase.lower()
    found: list[str] = []
    for keyword, part in policies.NON_LEAF_KEYWORDS:
        if keyword in text and part not in found:
            found.append(part)
    return sorted(found)
