"""Split-integrity check: leakage detection, read-only guarantee."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.core.exceptions import DerivationError, InvariantViolation
from plantdx.evaluation.integrity import check_split_integrity


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.mark.unit
def test_clean_split_passes(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"image": "/a.JPG"}, {"image": "/b.JPG"}])
    _write_jsonl(tmp_path / "test.jsonl", [{"image": "/c.JPG"}])
    report = check_split_integrity(tmp_path, "test")
    assert report.overlap_count == 0
    assert report.train_image_count == 2
    assert report.eval_image_count == 1


@pytest.mark.unit
def test_leaked_image_fails_closed(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "train.jsonl", [{"image": "/a.JPG"}, {"image": "/b.JPG"}])
    _write_jsonl(tmp_path / "test.jsonl", [{"image": "/a.JPG"}])
    with pytest.raises(InvariantViolation, match="contaminated"):
        check_split_integrity(tmp_path, "test")


@pytest.mark.unit
def test_never_writes_to_either_file(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, [{"image": "/a.JPG"}])
    _write_jsonl(test_path, [{"image": "/b.JPG"}])
    before_train = train_path.read_bytes()
    before_test = test_path.read_bytes()
    check_split_integrity(tmp_path, "test")
    assert train_path.read_bytes() == before_train
    assert test_path.read_bytes() == before_test


@pytest.mark.unit
def test_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(DerivationError, match="not found"):
        check_split_integrity(tmp_path, "test")


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_real_frozen_tomato_dataset_has_no_leakage() -> None:
    dataset_dir = Path("artifacts/training/qwen25vl_tomato_qlora/dataset")
    if not dataset_dir.is_dir():
        pytest.skip("frozen tomato dataset not present in this checkout")
    report = check_split_integrity(dataset_dir, "test")
    assert report.overlap_count == 0
    assert report.train_image_count > 0
    assert report.eval_image_count > 0
