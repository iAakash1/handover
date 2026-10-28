"""Company dataset preparation: dedup, splits, leakage control, immutability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from plantdx.company.dataset import (
    DatasetError,
    build_rows,
    check_provenance,
    load_manifest,
    prepare_dataset,
    scan_dataset,
    split_path,
    split_records,
)

from .conftest import make_image


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.unit
def test_scan_finds_every_supported_extension(company_dataset) -> None:
    records, audit = scan_dataset(company_dataset, verify=True)
    assert audit.total_files == 18
    assert len(records) == 18
    assert audit.corrupt == []
    assert audit.skipped_extension == []
    assert set(audit.per_class) == {"aphid", "late_blight", "healthy_leaf"}


@pytest.mark.unit
def test_corrupt_file_is_reported_not_fatal(company_dataset, raw_dataset: Path) -> None:
    (raw_dataset / "Aphid" / "broken.jpg").write_bytes(b"not an image")
    records, audit = scan_dataset(company_dataset, verify=True)
    assert len(audit.corrupt) == 1
    assert "broken.jpg" in audit.corrupt[0]["path"]
    assert len(records) == 18


@pytest.mark.unit
def test_unsupported_extension_is_skipped_and_listed(
    company_dataset,
    raw_dataset: Path,
) -> None:
    (raw_dataset / "Aphid" / "notes.txt").write_text("hello", encoding="utf-8")
    _records, audit = scan_dataset(company_dataset, verify=True)
    assert any("notes.txt" in p for p in audit.skipped_extension)


@pytest.mark.unit
def test_byte_identical_duplicates_are_dropped(company_dataset, raw_dataset: Path) -> None:
    source = raw_dataset / "Aphid" / "img_0.Jpg"
    (raw_dataset / "Aphid" / "copy.jpg").write_bytes(source.read_bytes())
    records, audit = scan_dataset(company_dataset, verify=True)
    assert audit.duplicate_groups == 1
    assert audit.dropped_duplicates == 1
    assert len(records) == 18


@pytest.mark.unit
def test_cross_class_duplicate_is_excluded_not_assigned(
    company_dataset,
    raw_dataset: Path,
) -> None:
    """A hash that appears under two classes has a contradictory label."""
    source = raw_dataset / "Aphid" / "img_0.Jpg"
    (raw_dataset / "Late_Blight" / "conflict.jpg").write_bytes(source.read_bytes())
    records, audit = scan_dataset(company_dataset, verify=True)
    assert len(audit.cross_class_duplicate_groups) == 1
    assert audit.dropped_ambiguous == 2
    kept = {r.relative_path for r in records}
    assert not any("conflict" in p or p.endswith("Aphid/img_0.Jpg") for p in kept)


@pytest.mark.unit
def test_split_is_deterministic_for_a_seed(company_dataset) -> None:
    records, _ = scan_dataset(company_dataset, verify=True)
    first = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    second = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert {k: [r.path for r in v] for k, v in first.items()} == {
        k: [r.path for r in v] for k, v in second.items()
    }


@pytest.mark.unit
def test_a_different_seed_gives_a_different_split(company_dataset) -> None:
    records, _ = scan_dataset(company_dataset, verify=True)
    a = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1)
    b = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=2)
    assert [r.path for r in a["test"]] != [r.path for r in b["test"]]


@pytest.mark.unit
def test_no_image_appears_in_two_splits(company_dataset) -> None:
    records, _ = scan_dataset(company_dataset, verify=True)
    splits = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=7)
    seen: set[str] = set()
    for members in splits.values():
        paths = {r.path for r in members}
        assert not (paths & seen)
        seen |= paths
    assert len(seen) == len(records)


@pytest.mark.unit
def test_every_class_reaches_validation_and_test(company_dataset) -> None:
    records, _ = scan_dataset(company_dataset, verify=True)
    splits = split_records(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=7)
    for name in ("validation", "test"):
        assert {r.class_id for r in splits[name]} == set(company_dataset.class_ids)


@pytest.mark.unit
def test_prepare_writes_all_splits_and_a_manifest(company_config, company_dataset) -> None:
    prepared = prepare_dataset(company_config, company_dataset)
    for split in ("train", "validation", "test"):
        path = split_path(prepared.dataset_dir, split)
        assert path.is_file()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows
        for row in rows:
            assert set(row) >= {"image", "question", "answer", "class_id"}
            assert row["answer"] in company_dataset.labels
            assert Path(row["image"]).is_file()
    manifest = load_manifest(prepared.dataset_dir)
    assert manifest["split"]["seed"] == company_config.data.split_seed
    assert manifest["labels"] == list(company_dataset.labels)
    assert sum(prepared.counts.values()) == prepared.audit.usable


@pytest.mark.unit
def test_prepare_is_reproducible(company_config, company_dataset) -> None:
    first = prepare_dataset(company_config, company_dataset)
    first_rows = split_path(first.dataset_dir, "test").read_text(encoding="utf-8")
    second = prepare_dataset(company_config, company_dataset)
    assert split_path(second.dataset_dir, "test").read_text(encoding="utf-8") == first_rows
    assert second.manifest_hash == first.manifest_hash


@pytest.mark.unit
def test_raw_dataset_is_never_modified(
    company_config,
    company_dataset,
    raw_dataset: Path,
) -> None:
    before = _fingerprint(raw_dataset)
    prepare_dataset(company_config, company_dataset)
    assert _fingerprint(raw_dataset) == before


@pytest.mark.unit
def test_provenance_rejects_a_plantvillage_path(raw_dataset: Path) -> None:
    rows = [{"image": str(raw_dataset / "tomato" / "raw" / "PlantVillage" / "leaf.JPG")}]
    problems = check_provenance(rows, raw_dataset)
    assert problems and "plantvillage" in problems[0].lower()


@pytest.mark.unit
def test_provenance_rejects_a_mango_path(raw_dataset: Path) -> None:
    rows = [{"image": str(raw_dataset / "mango" / "leaf.JPG")}]
    problems = check_provenance(rows, raw_dataset)
    assert problems and "mango" in problems[0].lower()


@pytest.mark.unit
def test_provenance_rejects_a_path_outside_the_company_root(
    raw_dataset: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere" / "leaf.jpg"
    problems = check_provenance([{"image": str(outside)}], raw_dataset)
    assert problems and "outside the company dataset root" in problems[0]


@pytest.mark.unit
def test_provenance_accepts_company_paths(company_dataset) -> None:
    records, _ = scan_dataset(company_dataset, verify=True)
    rows = build_rows(records, "q")
    assert check_provenance(rows, company_dataset.root) == []


@pytest.mark.unit
def test_prepare_fails_closed_when_the_dataset_is_empty(
    company_config,
    company_dataset,
    raw_dataset: Path,
) -> None:
    for path in raw_dataset.rglob("*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(DatasetError, match="no usable images"):
        prepare_dataset(company_config, company_dataset)


@pytest.mark.unit
def test_row_ordering_seed_interleaves_classes(company_dataset) -> None:
    records, _ = scan_dataset(company_dataset, verify=True)
    ordered = build_rows(records, "q", seed=11)
    assert len({r["class_id"] for r in ordered[:6]}) > 1
    again = build_rows(records, "q", seed=11)
    assert [r["image"] for r in ordered] == [r["image"] for r in again]


@pytest.mark.unit
def test_max_per_class_bounds_the_scan(company_dataset) -> None:
    records, _ = scan_dataset(company_dataset, verify=True, max_per_class=2)
    assert len(records) == 6


@pytest.mark.unit
def test_missing_manifest_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="dataset manifest not found"):
        load_manifest(tmp_path)


@pytest.mark.unit
def test_unknown_split_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="unknown split"):
        split_path(tmp_path, "holdout")


@pytest.mark.unit
def test_class_with_no_images_is_recorded_as_empty(
    company_dataset,
    raw_dataset: Path,
    tmp_path: Path,
) -> None:
    make_image(tmp_path / "unused.jpg", seed=999)
    for path in (raw_dataset / "Healthy_Leaf").iterdir():
        path.unlink()
    _records, audit = scan_dataset(company_dataset, verify=True)
    assert audit.per_class["healthy_leaf"] == 0
    assert "Healthy_Leaf" in audit.empty_dirs
