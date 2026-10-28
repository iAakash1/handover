"""Disease-label extraction + classification metrics."""

from __future__ import annotations

import pytest

from plantdx.evaluation.classification import compute_classification_metrics, extract_disease_id


@pytest.mark.unit
@pytest.mark.requires_dkb
class TestExtraction:
    def test_exact_disease_phrase(self, lexicon) -> None:
        text = "This tomato leaf is affected by bacterial spot."
        assert extract_disease_id(text, lexicon) == "tomato_bacterial_spot"

    def test_healthy(self, lexicon) -> None:
        assert extract_disease_id("a healthy tomato leaf", lexicon) == "tomato_healthy"

    def test_longest_match_disambiguates_spot_diseases(self, lexicon) -> None:
        # "spot" alone is ambiguous between Bacterial Spot and Target Spot; the
        # full phrase must win, not a bare substring.
        assert extract_disease_id("this shows target spot", lexicon) == "tomato_target_spot"
        assert extract_disease_id("this shows bacterial spot", lexicon) == "tomato_bacterial_spot"

    def test_no_match_is_unclassified(self, lexicon) -> None:
        assert extract_disease_id("completely unrelated text", lexicon) == "unclassified"

    def test_alias_matches(self, lexicon) -> None:
        assert extract_disease_id("damage from spider mite", lexicon) == "tomato_spider_mites"


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_classification_metrics_accuracy() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed (make install-eval)")
    labels = ["a", "b", "c"]
    targets = ["a", "a", "b", "b", "c"]
    predictions = ["a", "b", "b", "b", "c"]  # 4/5 correct
    metrics = compute_classification_metrics(predictions, targets, labels)
    assert metrics.accuracy == pytest.approx(0.8)
    assert metrics.top1_accuracy == metrics.accuracy
    assert metrics.sample_count == 5


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_classification_metrics_perfect_score() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed (make install-eval)")
    labels = ["a", "b"]
    metrics = compute_classification_metrics(["a", "b", "a"], ["a", "b", "a"], labels)
    assert metrics.accuracy == 1.0
    assert metrics.f1_macro == 1.0
    assert metrics.balanced_accuracy == 1.0


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_classification_metrics_length_mismatch_raises() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed (make install-eval)")
    with pytest.raises(ValueError, match="same length"):
        compute_classification_metrics(["a"], ["a", "b"], ["a", "b"])


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_unclassified_count() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed (make install-eval)")
    labels = ["a", "unclassified"]
    metrics = compute_classification_metrics(
        ["a", "unclassified", "unclassified"], ["a", "a", "a"], labels
    )
    assert metrics.unclassified_count == 2
