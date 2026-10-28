"""Tests for per-image inspection (metadata + corrupt handling)."""

from __future__ import annotations

from typing import Any

import pytest

from plantdx.audit import images


@pytest.mark.unit
def test_inspect_valid_image_reads_metadata(sample_dataset: dict[str, Any]) -> None:
    root = sample_dataset["root"]
    record = images.inspect_image(
        root / "A" / "solid1.png", "sample", "A", root, compute_ahash=True, ahash_size=8
    )
    assert record.ok
    assert (record.width, record.height) == (16, 16)
    assert record.format == "PNG"
    assert record.aspect_ratio == 1.0
    assert record.sha256 and record.ahash


@pytest.mark.unit
def test_corrupt_image_is_recorded_not_raised(sample_dataset: dict[str, Any]) -> None:
    root = sample_dataset["root"]
    record = images.inspect_image(
        root / "B" / "broken.jpg", "sample", "B", root, compute_ahash=False, ahash_size=8
    )
    assert record.ok is False
    assert record.error is not None and record.error.startswith("corrupt")
    assert record.sha256  # bytes were still hashed even though decoding failed
