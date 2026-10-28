"""Benchmark placeholders: corpus-level diversity gates (doc 00 §7.7).

Run against a generated caption library once Milestone 3 exists. Skipped until
then. These assert the hard acceptance gates that a releasable corpus must meet.
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skip(reason="Milestone 3: requires a generated corpus"),
]


def test_distinct_ngrams_meet_targets() -> None:
    """distinct-1/2/3 >= 0.10 / 0.45 / 0.70 (global)."""


def test_self_bleu_below_threshold() -> None:
    """Mean self-BLEU per disease <= 0.35."""


def test_concept_coverage_complete() -> None:
    """Every valid concept appears at least once per disease (coverage == 1.0)."""


def test_no_template_dominates() -> None:
    """No single template exceeds 8% share within a disease."""
