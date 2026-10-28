"""Latency aggregation + system-info capture."""

from __future__ import annotations

import pytest

from plantdx.evaluation.latency import capture_system_info, compute_latency_stats


@pytest.mark.unit
def test_latency_stats_basic() -> None:
    stats = compute_latency_stats([100.0, 200.0, 300.0], [10, 20, 30], [8.0, 8.5, 9.0])
    assert stats.mean_ms == pytest.approx(200.0)
    assert stats.median_ms == pytest.approx(200.0)
    assert stats.sample_count == 3
    assert stats.max_peak_memory_gb == 9.0
    assert stats.tokens_per_sec > 0
    assert stats.images_per_sec > 0


@pytest.mark.unit
def test_latency_stats_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        compute_latency_stats([100.0], [10, 20], [8.0])


@pytest.mark.unit
def test_latency_stats_empty_raises() -> None:
    with pytest.raises(ValueError):
        compute_latency_stats([], [], [])


@pytest.mark.unit
def test_capture_system_info_has_required_fields() -> None:
    info = capture_system_info()
    assert "python_version" in info
    assert "platform" in info
    assert "package_versions" in info
