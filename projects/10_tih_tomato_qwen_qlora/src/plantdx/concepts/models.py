"""Value objects for the Caption Concept Model.

Plain dataclasses, no behavior. Determinism is the builder's responsibility
(everything is sorted before output), exactly as in the ontology and vocabulary
compilers. A :class:`ConceptModel` is the deterministic, per-disease semantic
contract every caption of that disease must obey.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Concept legality status within a disease's model.
STATUS_MANDATORY = "mandatory"
STATUS_OPTIONAL = "optional"
STATUS_FORBIDDEN = "forbidden"


@dataclass
class CaptionConcept:
    """One concept available to (or forbidden from) captions of a disease.

    Traceability chain: ``dkb_fields`` (the DKB fields that licensed it) ->
    ``evidence`` (the citation keys resolving in the DKB reference registry).
    ``realizations`` are the controlled surface phrases a caption may use for
    this concept; ``modifiers`` are the legal single-word quality values that may
    decorate it (modifier legality); ``sign_type`` is the anatomy/sign it attaches
    to (relationship legality).
    """

    concept_id: str  # a plantdx.core.enums.ConceptId value, e.g. "primary_sign"
    status: str  # mandatory | optional | forbidden
    observable: bool  # visible on a single leaf (observability legality)
    confidence: str  # asserted | typical | hedged
    sign_type: str | None  # the sign type this concept attaches to, if any
    realizations: tuple[str, ...]  # controlled surface phrases (sorted)
    modifiers: tuple[str, ...]  # legal modifier values (sorted)
    evidence: tuple[str, ...]  # evidence node/citation ids (sorted)
    dkb_fields: tuple[str, ...]  # source DKB fields (sorted; traceability)


@dataclass
class ConceptModel:
    """The deterministic caption concept model for one disease/condition."""

    disease_id: str
    crop: str
    condition_type: str  # Disease | PestDamage | SurfaceColonization | HealthyState
    sign_type: str  # the disease's effective primary sign type, or "healthy"/"none"
    is_pathogen_disease: bool
    agent_category: str
    register_policy: dict[str, bool]  # visual/clinical/educational/severity_conditioned
    mandatory: tuple[str, ...]  # concept_ids (canonical order)
    optional: tuple[str, ...]  # concept_ids (canonical order)
    forbidden: tuple[str, ...]  # concept_ids (canonical order)
    ordering: tuple[str, ...]  # canonical concept ordering for sentence planning
    min_information: int
    max_information: int
    concepts: tuple[CaptionConcept, ...]  # per-concept detail (sorted by concept_id)
    never_appear: tuple[str, ...]  # forbidden surface terms (sorted)


@dataclass
class ConceptModelSet:
    """The complete build output: one model per disease plus provenance."""

    disease_models: list[ConceptModel]
    provenance: dict[str, str] = field(default_factory=dict)
