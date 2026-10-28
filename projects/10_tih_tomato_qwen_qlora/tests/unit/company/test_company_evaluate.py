"""Answer parsing, metrics, and the confusion matrix."""

from __future__ import annotations

import pytest

from plantdx.company.evaluate import (
    UNPARSEABLE,
    compare,
    confusion_matrix,
    evaluate_predictions,
    normalize_answer,
    parse_prediction,
    top_confusions,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("Late Blight", "late_blight"),
        ("late blight", "late_blight"),
        ("LATE BLIGHT", "late_blight"),
        ("Late_Blight", "late_blight"),
        ("Diagnosis: Late Blight", "late_blight"),
        ("diagnosis: late blight", "late_blight"),
        ("Answer: Late Blight.", "late_blight"),
        ("The condition is: Late Blight", "late_blight"),
        ("Late Blight\nVisible indication: dark lesions", "late_blight"),
        ("  Late Blight  ", "late_blight"),
        ("Healthy Leaf", "healthy_leaf"),
        ("healthy_leaf", "healthy_leaf"),
        ("Aphid", "aphid"),
    ],
)
def test_parsing_is_robust_to_wrapping_and_punctuation(
    company_dataset,
    generated: str,
    expected: str,
) -> None:
    assert parse_prediction(generated, company_dataset) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "generated",
    ["", "   ", "I am not sure what this is", "Septoria Leaf Spot", "42", "banana"],
)
def test_an_unknown_answer_is_never_guessed_into_a_class(
    company_dataset,
    generated: str,
) -> None:
    assert parse_prediction(generated, company_dataset) == UNPARSEABLE


@pytest.mark.unit
@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ("lateblight", "late_blight"),
        ("LateBlight", "late_blight"),
        ("healthyleaf", "healthy_leaf"),
        ("The image shows healthyleaf tissue", "healthy_leaf"),
    ],
)
def test_a_run_together_label_still_resolves(
    company_dataset, generated: str, expected: str
) -> None:
    """A model that writes `mealybugs` means `Mealy Bugs`; spacing is not a class."""
    assert parse_prediction(generated, company_dataset) == expected


@pytest.mark.unit
def test_run_together_matching_never_merges_two_classes(company_dataset) -> None:
    """The despaced fallback only applies while every label stays distinct."""
    despaced = {c.label.lower().replace(" ", "") for c in company_dataset.classes}
    assert len(despaced) == len(company_dataset.classes)


@pytest.mark.unit
def test_shipped_labels_stay_distinct_without_spaces() -> None:
    """Guards the real 11-class vocabulary against a future colliding label."""
    from pathlib import Path as _Path

    from plantdx.company.classes import load_company_dataset

    root = _Path(__file__).resolve().parents[3]
    dataset = load_company_dataset(root / "configs" / "company_tomato_classes.yaml")
    despaced = {c.label.lower().replace(" ", "") for c in dataset.classes}
    assert len(despaced) == len(dataset.classes)


@pytest.mark.unit
def test_longest_label_wins_when_one_contains_another(company_dataset) -> None:
    """`Healthy Leaf` must not be captured by a shorter overlapping phrase."""
    assert parse_prediction("this looks like Healthy Leaf to me", company_dataset) == "healthy_leaf"


@pytest.mark.unit
def test_normalize_strips_prefixes_and_punctuation() -> None:
    assert normalize_answer("Diagnosis: Early Blight.") == "early blight"
    assert normalize_answer("**Late Blight**") == "late blight"
    assert normalize_answer("label - Aphid") == "aphid"


@pytest.mark.unit
def test_metrics_on_a_perfect_prediction(company_dataset) -> None:
    truths = ["aphid", "late_blight", "healthy_leaf"]
    overall, rows, matrix = evaluate_predictions(truths, list(truths), company_dataset)
    assert overall["accuracy"] == pytest.approx(1.0)
    assert overall["f1_macro"] == pytest.approx(1.0)
    assert all(row.f1 == pytest.approx(1.0) for row in rows)
    assert [row[index] for index, row in enumerate(matrix)] == [1, 1, 1]


@pytest.mark.unit
def test_metrics_handle_an_unparseable_prediction(company_dataset) -> None:
    truths = ["aphid", "late_blight"]
    predictions = ["aphid", UNPARSEABLE]
    overall, rows, matrix = evaluate_predictions(truths, predictions, company_dataset)
    assert overall["accuracy"] == pytest.approx(0.5)
    late = next(r for r in rows if r.class_id == "late_blight")
    assert late.recall == pytest.approx(0.0)
    assert late.support == 1
    # unparseable lands in the trailing column, never on a real class
    assert matrix[1][-1] == 1


@pytest.mark.unit
def test_every_class_gets_a_row_even_with_zero_support(company_dataset) -> None:
    _overall, rows, _matrix = evaluate_predictions(["aphid"], ["aphid"], company_dataset)
    assert [r.class_id for r in rows] == list(company_dataset.class_ids)
    assert next(r for r in rows if r.class_id == "late_blight").support == 0


@pytest.mark.unit
def test_confusion_matrix_shape_and_placement() -> None:
    classes = ["a", "b"]
    matrix = confusion_matrix(["a", "a", "b"], ["a", "b", UNPARSEABLE], classes)
    assert len(matrix) == 2
    assert all(len(row) == 3 for row in matrix)
    assert matrix[0] == [1, 1, 0]
    assert matrix[1] == [0, 0, 1]


@pytest.mark.unit
def test_top_confusions_are_ranked_and_exclude_the_diagonal() -> None:
    classes = ["a", "b"]
    matrix = confusion_matrix(["a"] * 5 + ["b"] * 3, ["b"] * 5 + ["a"] * 3, classes)
    pairs = top_confusions(matrix, classes)
    assert pairs[0] == {"true": "a", "predicted": "b", "count": 5}
    assert all(p["true"] != p["predicted"] for p in pairs)


@pytest.mark.unit
def test_compare_reports_deltas_and_tolerates_a_missing_side() -> None:
    class Stub:
        accuracy = 0.5
        f1_macro = 0.4
        f1_weighted = 0.45
        precision_macro = 0.4
        recall_macro = 0.4

    class Better(Stub):
        accuracy = 0.9
        f1_macro = 0.85

    rows = compare(Stub(), Better())
    accuracy = next(r for r in rows if r["metric"] == "accuracy")
    assert accuracy["delta"] == pytest.approx(0.4)

    one_sided = compare(None, Better())
    assert all(r["base"] is None and r["delta"] is None for r in one_sided)
