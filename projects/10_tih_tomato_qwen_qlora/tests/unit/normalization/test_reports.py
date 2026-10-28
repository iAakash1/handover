"""Tests for normalization output files (mapping, manifest, card, run report)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from plantdx.config.schema import SourceSpec
from plantdx.normalization import report as report_writer
from plantdx.normalization.engine import normalize_crop

EXTS = {".jpg"}


def _normalize(spec_dict: dict[str, Any], tmp_path: Path):
    source = SourceSpec(dataset=spec_dict["dataset"], class_map=spec_dict["class_map"])
    processed_base = tmp_path / "datasets"
    report, images = normalize_crop(
        "tomato",
        source,
        spec_dict["root"],
        processed_base,
        mode="copy",
        extensions=EXTS,
        disambiguate=True,
        base_dir=tmp_path,
    )
    return source, report, images, processed_base


@pytest.mark.unit
def test_write_crop_outputs(plantvillage: dict[str, Any], tmp_path: Path) -> None:
    source, report, images, processed_base = _normalize(plantvillage, tmp_path)
    crop_dir = processed_base / "tomato"
    report_writer.write_crop_outputs(crop_dir, source, report, images)

    for name in ("class_mapping.json", "manifest.json", "dataset_card.md"):
        assert (crop_dir / name).is_file(), name

    mapping = json.loads((crop_dir / "class_mapping.json").read_text(encoding="utf-8"))
    assert mapping["mapping"]["Tomato___Early_blight"] == "early_blight"
    assert len(mapping["ignored_folders"]) == plantvillage["ignored_count"]

    manifest = json.loads((crop_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == plantvillage["image_count"]
    row = manifest["images"][0]
    assert {
        "source_path",
        "normalized_path",
        "checksum",
        "class_name",
        "split",
        "dataset",
    } <= row.keys()

    card = (crop_dir / "dataset_card.md").read_text(encoding="utf-8")
    for heading in (
        "Source dataset",
        "License",
        "Original citation",
        "Download URL",
        "Normalization timestamp",
        "Image count",
        "Class count",
        "Known limitations",
    ):
        assert heading in card


@pytest.mark.unit
def test_run_report_combines_crops(plantvillage: dict[str, Any], tmp_path: Path) -> None:
    _source, report, _images, processed_base = _normalize(plantvillage, tmp_path)
    path = processed_base / "normalization_report.json"
    report_writer.write_run_report(
        path, {"tomato": report}, plantdx_version="0.1.0", config_hash="abc", mode="copy"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tool"] == "plantdx.normalize"
    assert data["totals"]["images"] == plantvillage["image_count"]
    assert "tomato" in data["crops"]
