"""Statistical comparison: paired t-test, Wilcoxon, bootstrap CI, NaN-free edges."""

from __future__ import annotations

import math

import pytest

from plantdx.evaluation.stats import compare_paired, describe


@pytest.mark.unit
def test_describe_basic() -> None:
    stats = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.mean == pytest.approx(3.0)
    assert stats.n == 5


@pytest.mark.unit
def test_describe_empty() -> None:
    stats = describe([])
    assert stats.n == 0
    assert stats.mean == 0.0


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_compare_paired_detects_clear_improvement() -> None:
    pytest.importorskip("scipy", reason="scipy not installed (make install-eval)")
    import random

    random.seed(1)
    base = [random.gauss(0.5, 0.05) for _ in range(30)]
    finetuned = [b + random.gauss(0.3, 0.02) for b in base]
    cmp = compare_paired("metric", base, finetuned, seed=1)
    assert cmp.mean_difference == pytest.approx(0.3, abs=0.05)
    assert cmp.significant_at_0_05 is True


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_compare_paired_identical_samples_never_nan() -> None:
    pytest.importorskip("scipy", reason="scipy not installed (make install-eval)")
    cmp = compare_paired("metric", [0.5] * 10, [0.5] * 10, seed=1)
    assert not math.isnan(cmp.t_p_value)
    assert not math.isnan(cmp.wilcoxon_p_value)
    assert cmp.mean_difference == 0.0
    assert cmp.significant_at_0_05 is False


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_compare_paired_constant_difference_never_nan() -> None:
    pytest.importorskip("scipy", reason="scipy not installed (make install-eval)")
    cmp = compare_paired("metric", [0.5] * 10, [0.6] * 10, seed=1)
    assert not math.isnan(cmp.t_p_value)
    assert cmp.mean_difference == pytest.approx(0.1)


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_compare_paired_length_mismatch_raises() -> None:
    pytest.importorskip("scipy", reason="scipy not installed (make install-eval)")
    with pytest.raises(ValueError, match="same length"):
        compare_paired("metric", [1.0, 2.0], [1.0], seed=1)


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_compare_paired_too_few_samples_raises() -> None:
    pytest.importorskip("scipy", reason="scipy not installed (make install-eval)")
    with pytest.raises(ValueError, match="at least 2"):
        compare_paired("metric", [1.0], [2.0], seed=1)


@pytest.mark.unit
@pytest.mark.requires_eval_stack
def test_bootstrap_ci_contains_mean_difference() -> None:
    pytest.importorskip("scipy", reason="scipy not installed (make install-eval)")
    import random

    random.seed(7)
    base = [random.gauss(0.5, 0.1) for _ in range(60)]
    finetuned = [b + random.gauss(0.1, 0.03) for b in base]
    cmp = compare_paired("metric", base, finetuned, seed=7)
    assert cmp.bootstrap_ci_low <= cmp.mean_difference <= cmp.bootstrap_ci_high
