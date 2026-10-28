"""Text metrics: the dependency-free helpers always run; the official-library
integrations run only when their package (and, for BERTScore, a working
backbone load) is actually available in this environment.
"""

from __future__ import annotations

import pytest

from plantdx.evaluation.text_metrics import (
    _distinct_n,
    _sentence_similarity,
    _token_overlap,
    tokenize,
)

# --------------------------------------------------------------------------- #
# Dependency-free helpers (always run)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_tokenize() -> None:
    assert tokenize("This Leaf, shows Bacterial-Spot!") == [
        "this",
        "leaf",
        "shows",
        "bacterial",
        "spot",
    ]


@pytest.mark.unit
def test_token_overlap_identical() -> None:
    assert _token_overlap("a leaf shows spots", "a leaf shows spots") == 1.0


@pytest.mark.unit
def test_token_overlap_unrelated() -> None:
    assert _token_overlap("completely different text", "a leaf shows spots") == 0.0


@pytest.mark.unit
def test_token_overlap_empty_reference() -> None:
    assert _token_overlap("some text", "") == 0.0


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_sentence_similarity_identical_is_high() -> None:
    pytest.importorskip("sklearn", reason="scikit-learn not installed (make install-eval)")
    sims = _sentence_similarity(["a leaf shows spots"], ["a leaf shows spots"])
    assert sims[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_distinct_n_all_unique() -> None:
    assert _distinct_n(["a b c", "d e f"], 1) == 1.0


@pytest.mark.unit
def test_distinct_n_all_repeated() -> None:
    assert _distinct_n(["a a a", "a a a"], 1) == pytest.approx(1.0 / 6.0)


# --------------------------------------------------------------------------- #
# Official-library integrations (skip if the [eval] extra isn't installed)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.requires_eval_stack
class TestOfficialLibraryMetrics:
    def test_bleu_and_cider(self, has_eval_stack: bool) -> None:
        if not has_eval_stack:
            pytest.skip("nltk/pycocoevalcap/rouge-score not installed (make install-eval)")
        from plantdx.evaluation.text_metrics import _compute_bleu, _compute_cider

        ids = ["0", "1"]
        hyps = ["this leaf shows bacterial spot", "a completely unrelated healthy leaf"]
        refs = ["this leaf shows bacterial spot", "this leaf shows early blight lesions"]
        _, bleu_per_sample = _compute_bleu(ids, hyps, refs)
        assert bleu_per_sample[0][0] > bleu_per_sample[0][1]  # exact match scores higher

        _, cider_per_sample = _compute_cider(ids, hyps, refs)
        assert cider_per_sample[0] > cider_per_sample[1]

    def test_rouge_l(self, has_eval_stack: bool) -> None:
        if not has_eval_stack:
            pytest.skip("rouge-score not installed (make install-eval)")
        from plantdx.evaluation.text_metrics import _compute_rouge_l

        assert _compute_rouge_l("a leaf shows spots", "a leaf shows spots") == 1.0
        assert _compute_rouge_l("unrelated", "a leaf shows spots") < 1.0

    def test_meteor(self, has_eval_stack: bool) -> None:
        if not has_eval_stack:
            pytest.skip("nltk not installed or WordNet not downloaded (make install-eval)")
        from plantdx.evaluation.text_metrics import _compute_meteor

        longer = "this tomato leaf shows bacterial spot with small dark lesions"
        assert _compute_meteor(longer, longer) > 0.9

    def test_bertscore(self, has_bertscore: bool) -> None:
        if not has_bertscore:
            pytest.skip(
                "bert-score backbone unavailable in this environment "
                "(see docs/EVALUATION.md#troubleshooting)"
            )
        from plantdx.evaluation.text_metrics import _compute_bertscore

        scores = _compute_bertscore(["a leaf shows spots"], ["a leaf shows spots"])
        assert scores[0] > 0.99

    def test_compute_text_metrics_end_to_end(
        self, has_eval_stack: bool, has_bertscore: bool
    ) -> None:
        if not (has_eval_stack and has_bertscore):
            pytest.skip("full [eval] stack (incl. working BERTScore) not available")
        from plantdx.evaluation.text_metrics import compute_text_metrics

        pairs = [("0", "a leaf shows bacterial spot", "a leaf shows bacterial spot lesions")]
        per_sample, corpus = compute_text_metrics(pairs)
        assert corpus.sample_count == 1
        assert 0.0 <= per_sample[0].bleu1 <= 1.0
        assert 0.0 <= corpus.bertscore_f1 <= 1.0
