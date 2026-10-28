"""Per-disease breakdown table."""

from __future__ import annotations

import pytest

from plantdx.evaluation.per_disease import compute_per_disease_table


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_per_disease_table_shape_and_zero_sample_handling() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed (make install-eval)")
    disease_ids = ["a", "b", "c"]  # "c" has zero ground-truth samples
    targets = ["a", "a", "b"]
    predictions = ["a", "b", "b"]
    texts = ["one two", "three four five", "six"]
    confidences = [0.9, None, 0.5]
    hallucinated = [False, True, False]

    rows = compute_per_disease_table(
        disease_ids, predictions, targets, texts, confidences, hallucinated
    )
    by_id = {r.disease_id: r for r in rows}

    assert by_id["a"].sample_count == 2
    assert by_id["a"].accuracy == pytest.approx(0.5)
    assert by_id["a"].avg_confidence == pytest.approx(0.9)  # None excluded from mean
    assert by_id["b"].hallucination_count == 0
    assert by_id["c"].sample_count == 0
    assert by_id["c"].avg_confidence is None


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_length_mismatch_raises() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed (make install-eval)")
    with pytest.raises(ValueError, match="same length"):
        compute_per_disease_table(["a"], ["a"], ["a", "a"], ["x"], [None], [False])
