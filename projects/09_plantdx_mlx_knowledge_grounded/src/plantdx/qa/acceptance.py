"""QA acceptance evaluation (doc 05 §5, §6).

Computes inter-annotator agreement (Cohen's kappa), triages defects into
critical/major/minor, applies the batch acceptance rule (critical = 0,
major ≤ 1%, minor ≤ 5% per disease), and writes the acceptance sign-off.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from plantdx.qa.review import ChecklistVerdict


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """Outcome of the acceptance audit for a library version."""

    library_version: str
    accepted: bool
    kappa: float
    per_disease_critical: dict[str, int]
    per_disease_major_rate: dict[str, float]
    per_disease_minor_rate: dict[str, float]


class AcceptanceEvaluator:
    """Evaluates the batch acceptance rule and writes the sign-off (doc 05 §6)."""

    def __init__(
        self,
        max_major_rate: float = 0.01,
        max_minor_rate: float = 0.05,
        min_kappa: float = 0.80,
    ) -> None:
        """Initialize the evaluator with the acceptance-rule thresholds."""
        self.max_major_rate = max_major_rate
        self.max_minor_rate = max_minor_rate
        self.min_kappa = min_kappa

    def cohens_kappa(self, verdicts: Sequence[ChecklistVerdict]) -> float:
        """Compute inter-annotator agreement on the accept/reject decision."""
        raise NotImplementedError("Milestone 4: Cohen's kappa")

    def evaluate(
        self,
        verdicts: Sequence[ChecklistVerdict],
        library_version: str,
        out_path: str | Path,
    ) -> AcceptanceResult:
        """Apply the acceptance rule and write ``acceptance_<version>.md``."""
        raise NotImplementedError("Milestone 4: acceptance evaluation")
