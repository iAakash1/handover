"""The twelve caption validators V1-V12 (doc 03 §2).

Each validator is a small, single-responsibility subclass of :class:`BaseValidator`.
All are image-blind: they enforce consistency with the label's licensed
description, never pixel content.

Milestone 3 implements the bodies; Milestone 1 fixes the classes, ids, and the
one-line responsibility of each.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from plantdx.core.types import CaptionRecord, ValidatorResult


class BaseValidator(ABC):
    """Abstract base for the 12 blocking validators."""

    validator_id: str = "V0"

    @abstractmethod
    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:
        """Return a pass/fail result for ``record`` given the validation context."""
        raise NotImplementedError


class V1OntologyConformance(BaseValidator):
    """Asserted concept set ⊆ allowed, ⊇ required, within the info budget."""

    validator_id = "V1"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V1 ontology conformance")


class V2ForbiddenSymptom(BaseValidator):
    """No non-leaf-observable / foreign symptom is stated (observability)."""

    validator_id = "V2"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V2 forbidden-symptom detection")


class V3ForbiddenVocabulary(BaseValidator):
    """No forbidden term/adjective appears (per-disease ``never_appear``)."""

    validator_id = "V3"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V3 forbidden-vocabulary detection")


class V4ClosedVocabulary(BaseValidator):
    """Every domain content word is licensed by the DKB whitelist."""

    validator_id = "V4"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V4 closed-vocabulary enforcement")


class V5RequiredContent(BaseValidator):
    """Disease identity and ≥1 primary sign (or healthy_state) are present."""

    validator_id = "V5"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V5 required-content presence")


class V6NoDrift(BaseValidator):
    """Expansion introduced/dropped no concept; every modifier traces to the DKB."""

    validator_id = "V6"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V6 no-drift / realization integrity")


class V7RegisterConsistency(BaseValidator):
    """No non-observable concept in a visual caption; correct pest/pathogen language."""

    validator_id = "V7"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V7 register & pest/pathogen consistency")


class V8CrossDiseaseLeakage(BaseValidator):
    """No rival disease's hallmark term appears (except in a differential clause)."""

    validator_id = "V8"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V8 cross-disease leakage")


class V9SeverityGuard(BaseValidator):
    """No per-image severity stage claim unless a severity label is supplied."""

    validator_id = "V9"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V9 severity-claim guard")


class V10Consistency(BaseValidator):
    """No internal contradiction / mutual-exclusion violation."""

    validator_id = "V10"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V10 consistency / contradiction")


class V11Grammar(BaseValidator):
    """Grammatical, natural English; sentence count matches the template style."""

    validator_id = "V11"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V11 grammar & fluency")


class V12Duplication(BaseValidator):
    """No intra-caption, exact, or within-image near-duplicate."""

    validator_id = "V12"

    def validate(self, record: CaptionRecord, context: Mapping[str, Any]) -> ValidatorResult:  # noqa: D102
        raise NotImplementedError("Milestone 3: V12 duplication")


#: The validators in their fixed execution order (V1..V12).
ORDERED_VALIDATORS: tuple[type[BaseValidator], ...] = (
    V1OntologyConformance,
    V2ForbiddenSymptom,
    V3ForbiddenVocabulary,
    V4ClosedVocabulary,
    V5RequiredContent,
    V6NoDrift,
    V7RegisterConsistency,
    V8CrossDiseaseLeakage,
    V9SeverityGuard,
    V10Consistency,
    V11Grammar,
    V12Duplication,
)
