"""Corpus-level diversity metrics and gate evaluation (doc 00 §7.7).

Computes distinct-n, self-BLEU, template entropy, and concept / concept-pair
coverage over a generated corpus, then checks them against the hard acceptance
gates. A corpus failing any hard gate is rejected (:class:`DiversityGateError`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from plantdx.core.types import CaptionRecord


@dataclass(frozen=True, slots=True)
class DiversityMetrics:
    """Measured diversity metrics for a corpus (per disease and global)."""

    distinct_1: float
    distinct_2: float
    distinct_3: float
    self_bleu: float
    template_entropy_ratio: float
    concept_coverage: float
    concept_pair_coverage: float
    max_template_share: float


class DiversityEvaluator:
    """Computes diversity metrics and evaluates them against configured gates."""

    def __init__(self, gates: dict[str, float]) -> None:
        """Initialize the evaluator with the configured diversity gate thresholds."""
        self.gates = gates

    def compute(self, records: Sequence[CaptionRecord]) -> DiversityMetrics:
        """Compute diversity metrics over a set of caption records."""
        raise NotImplementedError("Milestone 3: diversity metric computation")

    def check_gates(self, metrics: DiversityMetrics) -> None:
        """Validate metrics against the hard gates.

        Raises:
            plantdx.core.exceptions.DiversityGateError: If any hard gate fails.
        """
        raise NotImplementedError("Milestone 3: diversity gate evaluation")
