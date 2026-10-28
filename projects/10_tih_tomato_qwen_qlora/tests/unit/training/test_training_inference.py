"""Inference helpers that do not require MLX (discovery, model registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from plantdx.core.exceptions import PlantDxError
from plantdx.training.inference import discover_image_paths
from plantdx.training.models import get_model_spec, registered_models


@pytest.mark.unit
def test_discover_image_paths_sorted(tmp_path: Path) -> None:
    for name in ("b.JPG", "a.jpg", "c.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    paths = discover_image_paths(tmp_path)
    assert [p.name for p in paths] == ["a.jpg", "b.JPG", "c.png"]


@pytest.mark.unit
def test_discover_non_directory_errors(tmp_path: Path) -> None:
    with pytest.raises(PlantDxError, match="not a directory"):
        discover_image_paths(tmp_path / "nope")


@pytest.mark.unit
def test_model_registry() -> None:
    assert "qwen2_5_vl" in registered_models()
    spec = get_model_spec("qwen2_5_vl")
    assert spec.hf_repo == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    with pytest.raises(KeyError):
        get_model_spec("nonexistent")
