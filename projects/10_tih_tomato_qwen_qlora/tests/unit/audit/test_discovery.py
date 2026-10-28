"""Tests for dataset/image discovery and folder validation."""

from __future__ import annotations

from typing import Any

import pytest

from plantdx.audit import discovery
from plantdx.config.schema import AuditConfig


@pytest.mark.unit
def test_discovers_images_and_classes_from_folders(sample_dataset: dict[str, Any]) -> None:
    root = sample_dataset["root"]
    images, unsupported = discovery.discover_images(root, AuditConfig().supported_extensions)

    assert len(images) == sample_dataset["num_images"]
    # Class == immediate parent folder name; no disease names are hardcoded.
    assert sorted({class_name for _, class_name in images}) == sample_dataset["classes"]
    # The .txt file is reported as unsupported, not silently dropped.
    assert [p.name for p in unsupported] == ["notes.txt"]


@pytest.mark.unit
def test_flags_empty_folder(sample_dataset: dict[str, Any]) -> None:
    empty_dirs, _unexpected = discovery.find_dir_issues(sample_dataset["root"])
    assert any(d.name == sample_dataset["empty_dir"] for d in empty_dirs)
