"""Validator Battery — component (G) (doc 03 §2, §4).

Runs V1..V12 in order for a candidate caption and returns a
:class:`plantdx.core.types.ValidationReport`. Blocking failures drive the
reseeded-regeneration loop owned by the engine; the battery also records soft
checks and aggregates a run-level report.
"""

from __future__ import annotations

from plantdx.core.types import CaptionRecord, ValidationReport
from plantdx.validation.grammar import GrammarChecker
from plantdx.validation.report import RunReport, ValidationContext
from plantdx.validation.validators import ORDERED_VALIDATORS, BaseValidator


class ValidatorBattery:
    """Orchestrates the 12 blocking validators plus soft checks.

    Args:
        grammar: The grammar checker used by V11.
        validators: The ordered validator instances (defaults to V1..V12).
    """

    def __init__(
        self,
        grammar: GrammarChecker,
        validators: tuple[BaseValidator, ...] | None = None,
    ) -> None:
        """Initialize the battery with the grammar checker and validator instances."""
        self.grammar = grammar
        self.validators: tuple[BaseValidator, ...] = validators or tuple(
            v() for v in ORDERED_VALIDATORS
        )

    def run(self, record: CaptionRecord, context: ValidationContext) -> ValidationReport:
        """Validate one caption; return its report (accept / needs-regenerate)."""
        raise NotImplementedError("Milestone 3: validator battery execution")

    def aggregate(self, library_version: str) -> RunReport:
        """Return the run-level aggregate report (rejections, fallback rate)."""
        raise NotImplementedError("Milestone 3: run-level aggregation")
