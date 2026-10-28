"""Robust, crop-scoped disease extraction for the demo app.

These exercise the app-layer wrapper (``app.classification``), which reuses the
frozen production extractor and adds normalized/alias fallbacks. They need the
committed DKB but no model.
"""

from __future__ import annotations

import pytest
from app.classification import UNCLASSIFIED, classify

pytestmark = [pytest.mark.unit, pytest.mark.requires_dkb]


@pytest.mark.parametrize(
    ("text", "crop", "expected"),
    [
        ("This tomato leaf shows early blight, concentric rings.", "tomato", "tomato_early_blight"),
        ("This mango leaf shows mango dieback.", "mango", "mango_die_back"),  # one-word variant
        ("sooty mold coating the mango leaf", "mango", "mango_sooty_mould"),  # US spelling
        ("This mango leaf shows mango leaf gall midge.", "mango", "mango_gall_midge"),
        ("This mango anthracnose.", "mango", "mango_anthracnose"),
        ("bacterial black spot lesions on mango", "mango", "mango_bacterial_canker"),  # alias
        (
            "Tomato yellow leaf curl virus interveinal chlorosis.",
            "tomato",
            "tomato_yellow_leaf_curl_virus",
        ),
        ("TYLCV symptoms present.", "tomato", "tomato_yellow_leaf_curl_virus"),  # acronym
        ("This is a healthy tomato leaf.", "tomato", "tomato_healthy"),
        (
            "Septoria!! leaf-spot, gray centers.",
            "tomato",
            "tomato_septoria_leaf_spot",
        ),  # punctuation
    ],
)
def test_classify_named_diseases(text: str, crop: str, expected: str) -> None:
    assert classify(text, crop) == expected


@pytest.mark.parametrize(
    ("text", "crop"),
    [
        ("This tomato leaf.", "tomato"),  # terse, no disease named
        ("the sky is blue", "tomato"),  # unrelated
        ("", "mango"),  # empty
    ],
)
def test_unnamed_text_is_unclassified(text: str, crop: str) -> None:
    assert classify(text, crop) == UNCLASSIFIED


def test_never_returns_other_crop_disease() -> None:
    # A mango disease phrase must not classify under tomato, and vice versa.
    assert classify("mango anthracnose", "tomato") == UNCLASSIFIED
    assert classify("tomato early blight", "mango") == UNCLASSIFIED
