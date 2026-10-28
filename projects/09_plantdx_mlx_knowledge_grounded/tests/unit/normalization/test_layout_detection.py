"""Tests for layout detection (flat vs nested train/val), no hardcoded split names."""

from __future__ import annotations

from typing import Any

import pytest

from plantdx.normalization.engine import find_class_dirs

EXTS = {".jpg"}


@pytest.mark.unit
def test_detects_nested_split_layout(plantvillage: dict[str, Any]) -> None:
    class_dirs = find_class_dirs(plantvillage["root"], EXTS)
    # Every class dir is discovered with its split (train/val), including the non-tomato crop.
    splits = {name.name: split for name, split in class_dirs}
    assert splits["Tomato___Early_blight"] in {"train", "val"}
    assert all(split in {"train", "val"} for _, split in class_dirs)
    assert "Corn_(maize)___healthy" in {p.name for p, _ in class_dirs}


@pytest.mark.unit
def test_detects_flat_layout(mango: dict[str, Any]) -> None:
    class_dirs = find_class_dirs(mango["root"], EXTS)
    assert {p.name for p, _ in class_dirs} == {"Anthracnose", "Bacterial Canker", "Healthy"}
    assert all(split is None for _, split in class_dirs)


@pytest.mark.unit
def test_missing_root_returns_empty(tmp_path: Any) -> None:
    assert find_class_dirs(tmp_path / "does_not_exist", EXTS) == []
