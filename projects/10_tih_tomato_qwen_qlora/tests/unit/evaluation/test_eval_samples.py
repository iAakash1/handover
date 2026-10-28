"""Deterministic sample selection + comparison table rendering."""

from __future__ import annotations

import pytest

from plantdx.evaluation.samples import (
    build_sample_comparison,
    render_markdown_table,
    select_sample_indices,
)


@pytest.mark.unit
def test_selection_is_deterministic() -> None:
    a = select_sample_indices(1000, seed=42, count=50)
    b = select_sample_indices(1000, seed=42, count=50)
    assert a == b
    assert len(a) == 50
    assert a == sorted(a)


@pytest.mark.unit
def test_selection_differs_by_seed() -> None:
    a = select_sample_indices(1000, seed=1, count=50)
    b = select_sample_indices(1000, seed=2, count=50)
    assert a != b


@pytest.mark.unit
def test_selection_caps_at_total() -> None:
    assert len(select_sample_indices(5, seed=1, count=50)) == 5


@pytest.mark.unit
def test_selection_empty_total() -> None:
    assert select_sample_indices(0, seed=1, count=50) == []


@pytest.mark.unit
def test_winner_judgement() -> None:
    row = build_sample_comparison(
        image_path="/x.JPG",
        ground_truth_disease="tomato_bacterial_spot",
        instruction="Describe.",
        ground_truth_answer="shows bacterial spot",
        base_answer="shows a plant",
        finetuned_answer="shows bacterial spot",
        base_bleu1=0.2,
        finetuned_bleu1=0.9,
    )
    assert row.winner == "finetuned"


@pytest.mark.unit
def test_tie_within_margin() -> None:
    row = build_sample_comparison(
        image_path="/x.JPG",
        ground_truth_disease="tomato_healthy",
        instruction="Describe.",
        ground_truth_answer="healthy",
        base_answer="healthy leaf",
        finetuned_answer="healthy leaf",
        base_bleu1=0.80,
        finetuned_bleu1=0.81,
    )
    assert row.winner == "tie"


@pytest.mark.unit
def test_markdown_table_escapes_pipes() -> None:
    row = build_sample_comparison(
        image_path="/x.JPG",
        ground_truth_disease="tomato_healthy",
        instruction="Describe.",
        ground_truth_answer="a | b",
        base_answer="c",
        finetuned_answer="d",
        base_bleu1=0.1,
        finetuned_bleu1=0.9,
    )
    table = render_markdown_table([row])
    assert "\\|" in table
    assert table.count("\n") >= 3  # header + separator + at least one data row
