"""Tests for exact and near-duplicate detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from plantdx.audit import discovery, duplicates, images
from plantdx.audit.models import ImageRecord
from plantdx.config.schema import AuditConfig


def _inspect_all(root: Path, *, near: bool) -> list[ImageRecord]:
    found, _ = discovery.discover_images(root, AuditConfig().supported_extensions)
    return [
        images.inspect_image(p, "s", c, root, compute_ahash=near, ahash_size=8) for p, c in found
    ]


@pytest.mark.unit
def test_exact_duplicates_by_sha256(sample_dataset: dict[str, Any]) -> None:
    groups = duplicates.exact_duplicate_groups(_inspect_all(sample_dataset["root"], near=False))
    assert len(groups) == sample_dataset["exact_groups"]
    _sha, paths = groups[0]
    assert len(paths) == 2  # solid1.png and its byte-identical copy


@pytest.mark.unit
def test_near_duplicates_by_average_hash(sample_dataset: dict[str, Any]) -> None:
    groups = duplicates.near_duplicate_groups(_inspect_all(sample_dataset["root"], near=True))
    assert len(groups) == sample_dataset["near_groups"]
    _ahash, paths = groups[0]
    # PNG, its copy, and the BMP re-encoding are visually identical (same aHash).
    assert len(paths) == 3
