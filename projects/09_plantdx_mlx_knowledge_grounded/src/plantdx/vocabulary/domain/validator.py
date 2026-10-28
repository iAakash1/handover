"""The vocabulary + symptom lexicon validator battery. Fail closed.

Each check re-derives its expectation independently from the ontology rather
than trusting the builder's own bookkeeping — the same defense-in-depth
pattern as ``plantdx.ontology.domain.validator``. Any violation aborts the
build. See ``docs/VOCABULARY.md`` for what each V-VOC-* rule guards against.
"""

from __future__ import annotations

from plantdx.core.exceptions import PlantDxError
from plantdx.ontology.domain import policies as ontology_policies
from plantdx.ontology.domain.models import Ontology
from plantdx.vocabulary.domain import graph_queries, policies
from plantdx.vocabulary.domain.models import LexicalItem, VocabularyResult

_CONFIDENCE_VALUES = frozenset(policies.CONFIDENCE_VALUES)
_CATEGORY_BY_NAME = {c.category: c for c in policies.CATEGORIES}
_COVERED_CONCEPT_TYPES = frozenset(t for c in policies.CATEGORIES for t in c.concept_types)


def _both_artifacts(result: VocabularyResult) -> tuple[tuple[str, list[LexicalItem]], ...]:
    """``(artifact_name, items)`` pairs, for checks that scan vocabulary + lexicon alike."""
    return (("vocabulary", result.vocabulary_items), ("lexicon", result.lexicon_items))


class VocabularyValidationError(PlantDxError):
    """Raised when the built vocabulary/lexicon violates one or more rules."""

    def __init__(self, violations: list[str]) -> None:
        """Initialize the error with the sorted list of rule violations."""
        self.violations = violations
        super().__init__(
            f"vocabulary validation failed ({len(violations)} error(s)):\n  "
            + "\n  ".join(violations)
        )


def collect_violations(result: VocabularyResult, ontology: Ontology) -> list[str]:
    """Run the full battery and return the sorted, deduplicated violation list."""
    violations: list[str] = []
    violations += _v1_duplicate_ids(result)
    violations += _v2_duplicate_realizations(result)
    violations += _v3_orphan_concepts(result, ontology)
    violations += _v4_missing_evidence(result)
    violations += _v5_illegal_combinations(result, ontology)
    violations += _v6_illegal_modifiers(result, ontology)
    violations += _v7_invalid_realizations(result)
    violations += _v8_unused_concepts(result, ontology)
    violations += _v9_conflicting_realizations(result, ontology)
    return sorted(set(violations))


def validate(result: VocabularyResult, ontology: Ontology) -> None:
    """Run the full battery; raise :class:`VocabularyValidationError` on any violation."""
    violations = collect_violations(result, ontology)
    if violations:
        raise VocabularyValidationError(violations)


def _v1_duplicate_ids(result: VocabularyResult) -> list[str]:
    v: list[str] = []
    seen: dict[str, str] = {}
    for artifact, items in _both_artifacts(result):
        for item in items:
            if item.id in seen:
                v.append(f"V-VOC-1: duplicate id {item.id!r} (first seen in {seen[item.id]})")
            else:
                seen[item.id] = artifact
    return v


def _v2_duplicate_realizations(result: VocabularyResult) -> list[str]:
    """No symptom emits two lexicon items with the same canonical realization."""
    v: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in result.lexicon_items:
        key = (item.ontology_node, item.canonical_form)
        if key in seen:
            v.append(
                f"V-VOC-2: duplicate realization {item.canonical_form!r} for {item.ontology_node}"
            )
        seen.add(key)
    return v


def _v3_orphan_concepts(result: VocabularyResult, ontology: Ontology) -> list[str]:
    """Every concept_id/ontology_node must resolve to something real."""
    v: list[str] = []
    node_ids = {n.id for n in ontology.nodes}
    for artifact, items in _both_artifacts(result):
        for item in items:
            known_type = item.concept_id in ontology_policies.CONCEPT_TYPE_BY_ID
            if item.concept_id != "Confidence" and not known_type:
                v.append(
                    f"V-VOC-3: {artifact} item {item.id} has unknown concept_id {item.concept_id!r}"
                )
            if item.ontology_node and item.ontology_node not in node_ids:
                v.append(
                    f"V-VOC-3: {artifact} item {item.id} references missing "
                    f"ontology_node {item.ontology_node!r}"
                )
    return v


def _v4_missing_evidence(result: VocabularyResult) -> list[str]:
    """Evidence presence must match each category's evidence-carrying contract."""
    v: list[str] = []
    for item in result.vocabulary_items:
        if item.concept == policies.CONFIDENCE_CATEGORY.category:
            if item.evidence:
                v.append(f"V-VOC-4: {item.id} (confidence_modifier) unexpectedly carries evidence")
            continue
        category = _CATEGORY_BY_NAME[item.concept]
        if category.carries_evidence and not item.evidence:
            v.append(f"V-VOC-4: vocabulary item {item.id} ({item.concept}) has no evidence")
        if not category.carries_evidence and item.evidence:
            v.append(
                f"V-VOC-4: vocabulary item {item.id} ({item.concept}) unexpectedly carries evidence"
            )
    for item in result.lexicon_items:
        if not item.evidence:
            v.append(f"V-VOC-4: lexicon item {item.id} has no evidence")
    return v


def _v5_illegal_combinations(result: VocabularyResult, ontology: Ontology) -> list[str]:
    """Every modifier realization must use a legal modifier relation on a legal sign type."""
    v: list[str] = []
    has_sign_type = {e.source: e.target for e in ontology.edges if e.type == "has_sign_type"}
    for item in result.lexicon_items:
        if ":mod:" not in item.id:
            continue
        if item.source not in policies.MODIFIER_RELATIONS:
            v.append(f"V-VOC-5: {item.id} uses non-modifier relation {item.source!r}")
        sign_type_id = has_sign_type.get(item.ontology_node)
        sign_type = sign_type_id.split(":", 1)[1] if sign_type_id else None
        if sign_type not in policies.MODIFIABLE_SIGN_TYPES:
            v.append(f"V-VOC-5: {item.id} modifies non-modifiable sign type {sign_type!r}")
    return v


def _v6_illegal_modifiers(result: VocabularyResult, ontology: Ontology) -> list[str]:
    """Modifier realizations are only legal on primary symptoms."""
    v: list[str] = []
    primary_symptoms = {
        e.target
        for e in ontology.edges
        if e.type == "has_symptom" and e.attributes.get("flags", {}).get("primary", False)
    }
    for item in result.lexicon_items:
        if ":mod:" in item.id and item.ontology_node not in primary_symptoms:
            v.append(f"V-VOC-6: {item.id} modifier attached to a non-primary symptom")
    return v


def _v7_invalid_realizations(result: VocabularyResult) -> list[str]:
    """Every item must satisfy the shared :class:`LexicalItem` schema contract."""
    v: list[str] = []
    for artifact, items in _both_artifacts(result):
        for item in items:
            if not item.id:
                v.append(f"V-VOC-7: {artifact} item has empty id")
            if not item.surface_form or item.surface_form != item.surface_form.strip():
                v.append(f"V-VOC-7: {item.id} has invalid surface_form {item.surface_form!r}")
            if "  " in item.surface_form:
                v.append(f"V-VOC-7: {item.id} surface_form has repeated whitespace")
            if not item.canonical_form or item.canonical_form != item.canonical_form.strip():
                v.append(f"V-VOC-7: {item.id} has invalid canonical_form {item.canonical_form!r}")
            if item.confidence not in _CONFIDENCE_VALUES:
                v.append(f"V-VOC-7: {item.id} has invalid confidence {item.confidence!r}")
            if not item.part_of_speech:
                v.append(f"V-VOC-7: {item.id} has empty part_of_speech")
            if item.language != "en":
                v.append(f"V-VOC-7: {item.id} has unexpected language {item.language!r}")
            if not item.concept:
                v.append(f"V-VOC-7: {item.id} has empty concept")
    return v


def _v8_unused_concepts(result: VocabularyResult, ontology: Ontology) -> list[str]:
    """Every ontology node in a covered category must produce a vocabulary item."""
    v: list[str] = []
    produced = {item.ontology_node for item in result.vocabulary_items}
    for node in ontology.nodes:
        if node.type in _COVERED_CONCEPT_TYPES and node.id not in produced:
            v.append(f"V-VOC-8: ontology node {node.id} ({node.type}) has no vocabulary item")
    return v


def _v9_conflicting_realizations(result: VocabularyResult, ontology: Ontology) -> list[str]:
    """Item labels must agree with an independent re-derivation from the ontology."""
    v: list[str] = []
    nodes_by_id = {n.id: n for n in ontology.nodes}
    for item in result.vocabulary_items:
        if item.concept == policies.CONFIDENCE_CATEGORY.category:
            continue
        node = nodes_by_id.get(item.ontology_node)
        if node is None:
            continue  # already flagged by V-VOC-3
        expected = graph_queries.node_label(node)
        if item.canonical_form != expected:
            v.append(
                f"V-VOC-9: {item.id} canonical_form {item.canonical_form!r} "
                f"conflicts with ontology node label {expected!r}"
            )
    for item in result.lexicon_items:
        if ":base" not in item.id:
            continue
        node = nodes_by_id.get(item.ontology_node)
        if node is None:
            continue
        expected = graph_queries.node_label(node)
        if item.canonical_form != expected:
            v.append(
                f"V-VOC-9: {item.id} canonical_form {item.canonical_form!r} "
                f"conflicts with symptom label {expected!r}"
            )
    return v
