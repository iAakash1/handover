"""Evaluation metrics (Milestone 6) (doc 04 training.evaluation).

Classification metrics (accuracy, macro-F1, confusion matrix) and knowledge-
grounded caption metrics (grounding precision, forbidden-term rate, concept
recall) computed against the DKB-derived ground truth for a class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Standard classification metrics for the identify task."""

    accuracy: float
    macro_f1: float
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class CaptionMetrics:
    """Knowledge-grounded caption-quality metrics."""

    grounding_precision: float  # fraction of stated signs licensed by the DKB
    forbidden_term_rate: float  # fraction of captions containing a forbidden term
    concept_recall: float  # fraction of required concepts mentioned


def classification_metrics(
    predictions: Sequence[str], targets: Sequence[str], labels: Sequence[str]
) -> ClassificationMetrics:
    """Compute classification metrics for predicted vs true disease ids."""
    raise NotImplementedError("Milestone 6: classification metrics")


def caption_metrics(predictions: Sequence[str], disease_ids: Sequence[str]) -> CaptionMetrics:
    """Compute grounding/forbidden/recall metrics against DKB-derived truth."""
    raise NotImplementedError("Milestone 6: caption metrics")
