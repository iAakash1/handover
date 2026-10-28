"""Zero-shot evaluation harness (Milestone 6).

Runs the base (un-tuned) target VLMs on the test and diagnostic splits to
establish the zero-shot baseline that motivates the knowledge-grounded approach.
Uses MLX inference; produces per-model prediction files for the comparison step.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.core.enums import Split, TargetModel


class ZeroShotEvaluator:
    """Evaluates a base VLM zero-shot on a split.

    Args:
        model: The target model to evaluate (base weights).
        splits: Which splits to run (default: test + diagnostic).
        report_dir: Where prediction/metric files are written.
    """

    def __init__(
        self,
        model: TargetModel,
        splits: tuple[Split, ...],
        report_dir: str | Path,
    ) -> None:
        """Initialize the evaluator with the target model, splits, and report dir."""
        self.model = model
        self.splits = splits
        self.report_dir = Path(report_dir)

    def run(self) -> Path:
        """Run zero-shot inference and write predictions; return the report path."""
        raise NotImplementedError("Milestone 6: zero-shot evaluation")
