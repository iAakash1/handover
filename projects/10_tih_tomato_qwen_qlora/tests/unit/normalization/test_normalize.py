"""Tests for the core normalization: extraction, naming, merge, dedup, checksums."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from plantdx.config.schema import SourceSpec
from plantdx.normalization.engine import normalize_crop
from plantdx.utils.hashing import sha256_bytes

EXTS = {".jpg"}


def _run(spec_dict: dict[str, Any], tmp_path: Path, crop: str, dataset: str):
    source = SourceSpec(dataset=dataset, class_map=spec_dict["class_map"])
    processed_base = tmp_path / "datasets"
    return normalize_crop(
        crop,
        source,
        spec_dict["root"],
        processed_base,
        mode="copy",
        extensions=EXTS,
        disambiguate=True,
        base_dir=tmp_path,
    ), processed_base


@pytest.mark.unit
def test_extracts_only_mapped_classes_and_normalizes_names(
    plantvillage: dict[str, Any], tmp_path: Path
) -> None:
    (report, _images), processed_base = _run(plantvillage, tmp_path, "tomato", "PlantVillage")
    # Only tomato classes present, with canonical names; Corn ignored (not copied).
    assert report.class_counts == plantvillage["class_counts"]
    assert report.image_count == plantvillage["image_count"]
    assert report.layout == plantvillage["layout"]
    assert len(report.ignored_folders) == plantvillage["ignored_count"]
    processed = processed_base / "tomato" / "processed"
    assert sorted(p.name for p in processed.iterdir()) == ["early_blight", "healthy"]
    assert not (processed / "Corn_(maize)___healthy").exists()


@pytest.mark.unit
def test_merges_train_val_with_dedup_and_collision(
    plantvillage: dict[str, Any], tmp_path: Path
) -> None:
    (report, _images), processed_base = _run(plantvillage, tmp_path, "tomato", "PlantVillage")
    early = processed_base / "tomato" / "processed" / "early_blight"
    names = sorted(p.name for p in early.iterdir())
    # a,b,dup,same,v from train/val + one disambiguated val__dup.jpg; val/same.jpg deduped.
    assert names == ["a.jpg", "b.jpg", "dup.jpg", "same.jpg", "v.jpg", "val__dup.jpg"]
    assert len(report.duplicates_skipped) == plantvillage["duplicates_skipped"]
    assert len(report.disambiguated) == plantvillage["disambiguated"]


@pytest.mark.unit
def test_checksums_verified_and_manifest_matches_bytes(
    plantvillage: dict[str, Any], tmp_path: Path
) -> None:
    (report, images), _ = _run(plantvillage, tmp_path, "tomato", "PlantVillage")
    assert report.checksum_failures == []
    for image in images:
        placed = tmp_path / image.normalized_path
        assert sha256_bytes(placed.read_bytes()) == image.checksum
        assert image.split in {"train", "val"}


@pytest.mark.unit
def test_raw_dataset_is_not_modified(plantvillage: dict[str, Any], tmp_path: Path) -> None:
    root = plantvillage["root"]
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    _run(plantvillage, tmp_path, "tomato", "PlantVillage")
    after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after  # raw is immutable


@pytest.mark.unit
def test_flat_mango_layout(mango: dict[str, Any], tmp_path: Path) -> None:
    (report, _images), processed_base = _run(mango, tmp_path, "mango", "MangoLeafBD")
    assert report.class_counts == mango["class_counts"]  # "Bacterial Canker" -> bacterial_canker
    assert report.layout == mango["layout"]
    processed = processed_base / "mango" / "processed"
    assert sorted(p.name for p in processed.iterdir()) == [
        "anthracnose",
        "bacterial_canker",
        "healthy",
    ]
