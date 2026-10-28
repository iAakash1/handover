"""The concept-model validator battery. Fail closed.

Each check re-derives its expectation independently from the frozen inputs
(DKB, ontology, vocabulary) rather than trusting the builder's own bookkeeping
— the same defense-in-depth pattern as the ontology and vocabulary validators.
Any violation aborts the build. See ``docs/CONCEPTS.md`` for what each
``V-CON-*`` rule guards against.
"""

from __future__ import annotations

from plantdx.concepts import policies
from plantdx.concepts.models import (
    STATUS_FORBIDDEN,
    STATUS_MANDATORY,
    STATUS_OPTIONAL,
    ConceptModel,
    ConceptModelSet,
)
from plantdx.core.enums import ConceptId
from plantdx.core.exceptions import PlantDxError
from plantdx.ontology.domain.models import Ontology
from plantdx.vocabulary.domain.models import VocabularyResult

CHECK_COUNT = 11

_VALID_CONCEPT_IDS = frozenset(c.value for c in ConceptId)
# Concepts whose realizations trace through an evidence-carrying relation; each
# must carry >=1 evidence id when it has realizations (traceability). Concepts
# excluded here are backed by structural (evidence-free) facts: leaf_location
# (appears_on), host (crop), and the agent concepts.
_EVIDENCE_REQUIRED = frozenset(
    {
        "disease_identity",
        "primary_sign",
        "secondary_sign",
        "healthy_state",
        "lesion_color",
        "lesion_shape",
        "texture",
        "extent",
        "chlorosis",
        "necrosis",
        "leaf_deformation",
        "lesion_distribution",
        "differential",
    }
)


class ConceptValidationError(PlantDxError):
    """Raised when the built concept models violate one or more rules."""

    def __init__(self, violations: list[str]) -> None:
        """Initialize the error with the sorted list of rule violations."""
        self.violations = violations
        super().__init__(
            f"concept-model validation failed ({len(violations)} error(s)):\n  "
            + "\n  ".join(violations)
        )


def collect_violations(
    result: ConceptModelSet, ontology: Ontology, vocabulary: VocabularyResult
) -> list[str]:
    """Run the full battery and return the sorted, deduplicated violation list."""
    violations: list[str] = []
    violations += _v1_disease_coverage(result, ontology)
    for model in result.disease_models:
        violations += _v2_valid_concept_ids(model)
        violations += _v3_mandatory_realized(model)
        violations += _v4_forbidden_empty(model)
        violations += _v5_budget(model)
        violations += _v6_observability(model)
        violations += _v7_modifier_legality(model)
        violations += _v8_evidence(model)
        violations += _v9_never_appear(model)
        violations += _v10_healthy_legality(model)
        violations += _v11_ordering(model)
    return sorted(set(violations))


def validate(result: ConceptModelSet, ontology: Ontology, vocabulary: VocabularyResult) -> None:
    """Run the full battery; raise :class:`ConceptValidationError` on any violation."""
    violations = collect_violations(result, ontology, vocabulary)
    if violations:
        raise ConceptValidationError(violations)


def _v1_disease_coverage(result: ConceptModelSet, ontology: Ontology) -> list[str]:
    """Exactly one model per ontology condition; no extras, no duplicates."""
    v: list[str] = []
    model_ids = [m.disease_id for m in result.disease_models]
    seen: set[str] = set()
    for did in model_ids:
        if did in seen:
            v.append(f"V-CON-1: duplicate concept model for {did}")
        seen.add(did)
    condition_ids = {
        str(n.properties.get("disease_id"))
        for n in ontology.nodes
        if n.type in ("Disease", "PestDamage", "SurfaceColonization", "HealthyState")
    }
    for missing in sorted(condition_ids - seen):
        v.append(f"V-CON-1: ontology condition {missing} has no concept model")
    for extra in sorted(seen - condition_ids):
        v.append(f"V-CON-1: concept model {extra} has no ontology condition")
    return v


def _v2_valid_concept_ids(model: ConceptModel) -> list[str]:
    v: list[str] = []
    for concept in model.concepts:
        if concept.concept_id not in _VALID_CONCEPT_IDS:
            v.append(f"V-CON-2: {model.disease_id} has unknown concept_id {concept.concept_id!r}")
        if concept.status not in (STATUS_MANDATORY, STATUS_OPTIONAL, STATUS_FORBIDDEN):
            v.append(
                f"V-CON-2: {model.disease_id}/{concept.concept_id} bad status {concept.status!r}"
            )
    return v


def _v3_mandatory_realized(model: ConceptModel) -> list[str]:
    """Every mandatory concept exists, is marked mandatory, and has >=1 realization."""
    v: list[str] = []
    by_id = {c.concept_id: c for c in model.concepts}
    if not model.mandatory:
        v.append(f"V-CON-3: {model.disease_id} has no mandatory concepts")
    for cid in model.mandatory:
        concept = by_id.get(cid)
        if concept is None:
            v.append(f"V-CON-3: {model.disease_id} mandatory {cid} missing from concepts")
        elif concept.status != STATUS_MANDATORY:
            v.append(f"V-CON-3: {model.disease_id} mandatory {cid} not marked mandatory")
        elif not concept.realizations:
            v.append(f"V-CON-3: {model.disease_id} mandatory {cid} has no realization")
    return v


def _v4_forbidden_empty(model: ConceptModel) -> list[str]:
    """Forbidden concepts carry no realizations; ALWAYS_FORBIDDEN is present."""
    v: list[str] = []
    by_id = {c.concept_id: c for c in model.concepts}
    for cid in model.forbidden:
        concept = by_id.get(cid)
        if concept is not None and concept.realizations:
            v.append(f"V-CON-4: {model.disease_id} forbidden {cid} still has realizations")
    for cid in policies.ALWAYS_FORBIDDEN:
        if cid not in model.forbidden:
            v.append(f"V-CON-4: {model.disease_id} must forbid {cid}")
    return v


def _v5_budget(model: ConceptModel) -> list[str]:
    v: list[str] = []
    if model.min_information != len(model.mandatory):
        v.append(f"V-CON-5: {model.disease_id} min_information != |mandatory|")
    if model.max_information != len(model.mandatory) + len(model.optional):
        v.append(f"V-CON-5: {model.disease_id} max_information != |mandatory|+|optional|")
    if model.min_information > model.max_information:
        v.append(f"V-CON-5: {model.disease_id} min_information > max_information")
    return v


def _v6_observability(model: ConceptModel) -> list[str]:
    """Observability flag matches the fixed non-observable-concept set."""
    v: list[str] = []
    for concept in model.concepts:
        expected = concept.concept_id not in policies.NON_OBSERVABLE_CONCEPTS
        if concept.observable != expected:
            v.append(
                f"V-CON-6: {model.disease_id}/{concept.concept_id} observable={concept.observable} "
                f"!= expected {expected}"
            )
    return v


def _v7_modifier_legality(model: ConceptModel) -> list[str]:
    """Quality-modifier concepts and primary-sign modifiers require a modifiable sign type."""
    v: list[str] = []
    by_id = {c.concept_id: c for c in model.concepts}
    primary = by_id.get("primary_sign")
    modifiable = primary is not None and primary.sign_type in policies.MODIFIABLE_SIGN_TYPES
    for cid in policies.MODIFIER_CONCEPTS:
        concept = by_id.get(cid)
        if concept is not None and concept.status != STATUS_FORBIDDEN and not modifiable:
            v.append(f"V-CON-7: {model.disease_id} offers {cid} but primary sign is not modifiable")
    if primary is not None and primary.modifiers and not modifiable:
        v.append(
            f"V-CON-7: {model.disease_id} primary_sign carries modifiers but sign is not modifiable"
        )
    return v


def _v8_evidence(model: ConceptModel) -> list[str]:
    """Evidence-required concepts with realizations must carry >=1 evidence id."""
    v: list[str] = []
    for concept in model.concepts:
        if (
            concept.concept_id in _EVIDENCE_REQUIRED
            and concept.status != STATUS_FORBIDDEN
            and concept.realizations
            and not concept.evidence
        ):
            v.append(f"V-CON-8: {model.disease_id}/{concept.concept_id} has no evidence")
    return v


def _v9_never_appear(model: ConceptModel) -> list[str]:
    """never_appear must include stage tokens and must not forbid a mandatory realization."""
    v: list[str] = []
    never = set(model.never_appear)
    for token in policies.STAGE_TOKENS:
        if token not in never:
            v.append(f"V-CON-9: {model.disease_id} never_appear missing stage token {token!r}")
    by_id = {c.concept_id: c for c in model.concepts}
    for cid in model.mandatory:
        concept = by_id.get(cid)
        if concept is None:
            continue
        for phrase in concept.realizations:
            if phrase in never:
                v.append(f"V-CON-9: {model.disease_id} forbids its own mandatory phrase {phrase!r}")
    return v


def _v10_healthy_legality(model: ConceptModel) -> list[str]:
    v: list[str] = []
    is_healthy = model.condition_type == "HealthyState"
    if is_healthy:
        if "healthy_state" not in model.mandatory:
            v.append(f"V-CON-10: {model.disease_id} healthy must make healthy_state mandatory")
        if "primary_sign" in model.mandatory + model.optional:
            v.append(f"V-CON-10: {model.disease_id} healthy model must not offer disease signs")
    else:
        if "healthy_state" not in model.forbidden:
            v.append(f"V-CON-10: {model.disease_id} non-healthy model must forbid healthy_state")
        if "primary_sign" not in model.mandatory:
            v.append(f"V-CON-10: {model.disease_id} non-healthy model must require primary_sign")
    return v


def _v11_ordering(model: ConceptModel) -> list[str]:
    """Ordering is a subsequence of the canonical order covering exactly present concepts."""
    v: list[str] = []
    present = {c.concept_id for c in model.concepts}
    expected = tuple(c for c in policies.CONCEPT_ORDER if c in present)
    if model.ordering != expected:
        v.append(f"V-CON-11: {model.disease_id} ordering is not the canonical concept order")
    return v
