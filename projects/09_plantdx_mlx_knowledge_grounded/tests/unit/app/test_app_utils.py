"""Pure helpers in ``app.utils`` (no Streamlit, no model)."""

from __future__ import annotations

import pytest
from app.utils import (
    class_folder,
    confidence_band,
    crop_profile,
    is_supported_upload,
    pretty_disease,
    sanitize_filename,
    unique_filename,
)

pytestmark = pytest.mark.unit


def test_crop_profile_tomato_and_mango() -> None:
    t = crop_profile("tomato")
    assert t.run_name == "qwen25vl_tomato_qlora"
    assert "tomato leaf" in t.instruction
    m = crop_profile("MANGO")  # case-insensitive
    assert m.run_name == "qwen25vl_mango_qlora"
    assert "mango leaf" in m.instruction


def test_crop_profile_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported crop"):
        crop_profile("cotton")


@pytest.mark.parametrize(
    ("disease_id", "crop", "expected"),
    [
        ("tomato_early_blight", "tomato", "early_blight"),
        ("mango_anthracnose", "mango", "anthracnose"),
        ("unclassified", "tomato", "unknown"),
        ("", "tomato", "unknown"),
        ("mango_anthracnose", "tomato", "unknown"),  # crop mismatch
    ],
)
def test_class_folder(disease_id: str, crop: str, expected: str) -> None:
    assert class_folder(disease_id, crop) == expected


def test_confidence_band_colors() -> None:
    assert confidence_band(0.9)[0] == "high"
    assert confidence_band(0.6)[0] == "moderate"
    assert confidence_band(0.2)[0] == "low"
    assert confidence_band(None)[0] == "n/a"


def test_pretty_disease() -> None:
    assert pretty_disease("tomato_early_blight") == "Early Blight"
    assert pretty_disease("unclassified") == "Unclassified"


def test_supported_extensions() -> None:
    assert is_supported_upload("a.JPG")
    assert is_supported_upload("b.png")
    assert not is_supported_upload("c.gif")
    assert not is_supported_upload("d.pdf")


def test_unique_filename_is_unique_and_safe() -> None:
    a = unique_filename("My Leaf (1).JPG")
    b = unique_filename("My Leaf (1).JPG")
    assert a != b  # uuid component
    assert " " not in a and "(" not in a
    assert a.endswith(".JPG")


def test_sanitize_filename() -> None:
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("") == "image"
