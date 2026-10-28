"""Zero-shot vs fine-tuned comparison (Milestone 6).

Aggregates per-model zero-shot and fine-tuned metrics into the comparison tables
reported in the paper, including the diagnostic confusable-pair breakdown.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from plantdx.core.enums import TargetModel


class ComparisonReporter:
    """Builds the cross-model, zero-shot-vs-fine-tuned comparison report."""

    def __init__(self, report_dir: str | Path) -> None:
        """Initialize the reporter with the output directory for the comparison report."""
        self.report_dir = Path(report_dir)

    def build(self, models: Sequence[TargetModel]) -> Path:
        """Assemble the comparison report across models; return its path."""
        raise NotImplementedError("Milestone 6: zero-shot vs fine-tuned comparison")
