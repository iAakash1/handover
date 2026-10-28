"""Response-quality heuristics: length, fluency, redundancy, repetition, diversity."""

from __future__ import annotations

import pytest

from plantdx.evaluation.quality import score_response_quality, summarize_response_quality


@pytest.mark.unit
def test_well_formed_response_scores_high_fluency() -> None:
    q = score_response_quality("This tomato leaf shows bacterial spot with small dark spots.")
    assert q.fluency_score == 1.0
    assert q.word_count == 10
    assert q.sentence_count == 1


@pytest.mark.unit
def test_repeated_words_flag_redundancy() -> None:
    q = score_response_quality("leaf leaf leaf shows shows bacterial spot")
    assert q.redundancy_rate > 0


@pytest.mark.unit
def test_duplicate_sentences_flag_repetition() -> None:
    q = score_response_quality("This leaf has spots. This leaf has spots.")
    assert q.repetition_rate > 0


@pytest.mark.unit
def test_empty_response() -> None:
    q = score_response_quality("")
    assert q.word_count == 0
    assert q.fluency_score == 0.0


@pytest.mark.unit
def test_lexical_diversity_bounds() -> None:
    q_diverse = score_response_quality("this leaf shows unique distinct varied words")
    q_repeated = score_response_quality("leaf leaf leaf leaf leaf leaf")
    assert q_diverse.lexical_diversity > q_repeated.lexical_diversity


@pytest.mark.unit
def test_summarize_empty_list() -> None:
    summary = summarize_response_quality([])
    assert summary["avg_word_count"] == 0.0


@pytest.mark.unit
def test_summarize_averages_correctly() -> None:
    rows = [score_response_quality("a b c"), score_response_quality("d e f g")]
    summary = summarize_response_quality(rows)
    assert summary["avg_word_count"] == pytest.approx(3.5)
