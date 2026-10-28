"""Statistical comparison of base vs. fine-tuned per-sample metric values.

Uses scipy's reference implementations for every test (paired t-test, Wilcoxon
signed-rank, and bootstrap confidence intervals) -- never a hand-rolled formula.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class DescriptiveStats:
    """Mean, standard deviation, and a 95% CI for one sample of values."""

    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class PairedComparison:
    """A paired comparison of the same metric between two models."""

    metric_name: str
    base: DescriptiveStats
    finetuned: DescriptiveStats
    mean_difference: float  # finetuned.mean - base.mean
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    bootstrap_ci_low: float  # 95% CI on the mean difference, via bootstrap
    bootstrap_ci_high: float
    significant_at_0_05: bool  # True iff the paired t-test p-value < 0.05


def describe(values: list[float]) -> DescriptiveStats:
    """Mean/std/95% CI (normal approximation) for one sample of values."""
    n = len(values)
    if n == 0:
        return DescriptiveStats(mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, n=0)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    margin = 1.96 * (std / (n**0.5)) if n > 1 else 0.0
    return DescriptiveStats(mean=mean, std=std, ci_low=mean - margin, ci_high=mean + margin, n=n)


def compare_paired(
    metric_name: str,
    base_values: list[float],
    finetuned_values: list[float],
    *,
    seed: int,
    bootstrap_resamples: int = 2000,
) -> PairedComparison:
    """Paired statistical comparison of one metric across the same samples.

    Raises :class:`ValueError` if the two value lists are not the same length
    (they must come from the same samples, in the same order).
    """
    import numpy as np
    from scipy import stats as scipy_stats

    if len(base_values) != len(finetuned_values):
        raise ValueError("base_values and finetuned_values must be the same length (paired)")
    if len(base_values) < 2:
        raise ValueError("at least 2 paired samples are required for a paired comparison")

    base_stats = describe(base_values)
    ft_stats = describe(finetuned_values)

    diffs = [f - b for f, b in zip(finetuned_values, base_values, strict=True)]
    has_variance = len(set(diffs)) > 1
    if has_variance:
        t_stat, t_p = scipy_stats.ttest_rel(finetuned_values, base_values)
    else:
        # Zero variance in the paired differences: scipy's t-test divides by that
        # variance and returns NaN, which is not JSON-serializable and not
        # meaningful anyway -- a constant difference has no dispersion to test.
        t_stat, t_p = 0.0, 1.0
    if any(d != 0 for d in diffs):
        w_stat, w_p = scipy_stats.wilcoxon(finetuned_values, base_values)
    else:
        w_stat, w_p = 0.0, 1.0  # identical paired samples: no evidence of a difference

    rng = np.random.default_rng(seed)
    diff_array = np.asarray(diffs)
    boot_result = scipy_stats.bootstrap(
        (diff_array,),
        np.mean,
        n_resamples=bootstrap_resamples,
        confidence_level=0.95,
        random_state=rng,
        method="percentile",
    )

    return PairedComparison(
        metric_name=metric_name,
        base=base_stats,
        finetuned=ft_stats,
        mean_difference=ft_stats.mean - base_stats.mean,
        t_statistic=float(t_stat),
        t_p_value=float(t_p),
        wilcoxon_statistic=float(w_stat),
        wilcoxon_p_value=float(w_p),
        bootstrap_ci_low=float(boot_result.confidence_interval.low),
        bootstrap_ci_high=float(boot_result.confidence_interval.high),
        significant_at_0_05=float(t_p) < 0.05,
    )
