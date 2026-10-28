"""Per-disease breakdown: accuracy, quality, and hallucination stats per class."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PerDiseaseRow:
    """One disease's aggregate stats for one model (base or fine-tuned)."""

    disease_id: str
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    avg_confidence: float | None
    avg_response_length: float
    hallucination_count: int
    hallucination_rate: float


def compute_per_disease_table(
    disease_ids: Sequence[str],
    predictions: Sequence[str],
    targets: Sequence[str],
    response_texts: Sequence[str],
    confidences: Sequence[float | None],
    hallucinated: Sequence[bool],
) -> list[PerDiseaseRow]:
    """Compute one :class:`PerDiseaseRow` per disease in `disease_ids`.

    All sequence arguments must be aligned by index and the same length as
    `targets`. A disease with zero ground-truth samples still appears with
    `sample_count=0` and zeroed metrics (never silently dropped from the table).
    """
    from sklearn.metrics import precision_recall_fscore_support

    n = len(targets)
    if not (len(predictions) == len(response_texts) == len(confidences) == len(hallucinated) == n):
        raise ValueError("all per-sample sequences must be the same length as targets")

    label_list = list(disease_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=label_list, average=None, zero_division=0
    )

    rows: list[PerDiseaseRow] = []
    for idx, disease_id in enumerate(label_list):
        indices = [i for i in range(n) if targets[i] == disease_id]
        rows.append(
            _row_for_disease(
                disease_id,
                indices,
                predictions,
                response_texts,
                confidences,
                hallucinated,
                precision[idx],
                recall[idx],
                f1[idx],
            )
        )
    return rows


def _row_for_disease(
    disease_id: str,
    indices: list[int],
    predictions: Sequence[str],
    response_texts: Sequence[str],
    confidences: Sequence[float | None],
    hallucinated: Sequence[bool],
    precision: float,
    recall: float,
    f1: float,
) -> PerDiseaseRow:
    count = len(indices)
    if count == 0:
        return PerDiseaseRow(
            disease_id=disease_id,
            sample_count=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            avg_confidence=None,
            avg_response_length=0.0,
            hallucination_count=0,
            hallucination_rate=0.0,
        )

    correct = sum(1 for i in indices if predictions[i] == disease_id)
    lengths = [len(response_texts[i].split()) for i in indices]
    known_confidences: list[float] = [c for i in indices if (c := confidences[i]) is not None]
    hallucination_count = sum(1 for i in indices if hallucinated[i])

    return PerDiseaseRow(
        disease_id=disease_id,
        sample_count=count,
        accuracy=correct / count,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        avg_confidence=(
            sum(known_confidences) / len(known_confidences) if known_confidences else None
        ),
        avg_response_length=sum(lengths) / count,
        hallucination_count=hallucination_count,
        hallucination_rate=hallucination_count / count,
    )
