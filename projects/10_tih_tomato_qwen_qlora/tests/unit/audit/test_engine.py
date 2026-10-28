"""End-to-end tests for the audit engine: reports, summaries, checksums."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from plantdx.audit import run_audit
from plantdx.audit.discovery import DatasetSpec
from plantdx.config.schema import AuditConfig


def _spec(root: Path, classes: int | None = 2) -> DatasetSpec:
    return DatasetSpec(key="sample", name="Sample", root=root, configured_classes=classes)


@pytest.mark.unit
def test_run_audit_writes_all_reports(sample_dataset: dict[str, Any], tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    manifest = run_audit([_spec(sample_dataset["root"])], AuditConfig(), reports)

    expected_files = [
        "dataset_card.md",
        "sample_summary.json",
        "class_distribution.csv",
        "image_statistics.csv",
        "duplicate_images.csv",
        "corrupt_images.csv",
        "audit_manifest.json",
        "audit.log",
    ]
    for name in expected_files:
        assert (reports / name).is_file(), f"missing report: {name}"

    summary = json.loads((reports / "sample_summary.json").read_text(encoding="utf-8"))
    assert summary["num_images"] == sample_dataset["num_images"]
    assert summary["num_ok"] == sample_dataset["num_ok"]
    assert summary["num_corrupt"] == sample_dataset["num_corrupt"]
    assert summary["num_classes"] == len(sample_dataset["classes"])
    assert summary["num_exact_duplicate_groups"] == sample_dataset["exact_groups"]
    assert manifest.totals["images"] == sample_dataset["num_images"]


@pytest.mark.unit
def test_manifest_json_parses_and_has_checksum(
    sample_dataset: dict[str, Any], tmp_path: Path
) -> None:
    reports = tmp_path / "reports"
    run_audit([_spec(sample_dataset["root"])], AuditConfig(), reports)
    manifest = json.loads((reports / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit_checksum"]
    assert manifest["datasets"]["sample"]
    assert manifest["splits"]["status"] == "not_performed"


@pytest.mark.unit
def test_checksum_is_deterministic(sample_dataset: dict[str, Any], tmp_path: Path) -> None:
    root = sample_dataset["root"]
    first = run_audit([_spec(root)], AuditConfig(), tmp_path / "r1")
    second = run_audit([_spec(root)], AuditConfig(), tmp_path / "r2")
    assert first.audit_checksum == second.audit_checksum
    assert first.datasets == second.datasets


@pytest.mark.unit
def test_class_count_mismatch_is_flagged(sample_dataset: dict[str, Any], tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    run_audit([_spec(sample_dataset["root"], classes=99)], AuditConfig(), reports)
    summary = json.loads((reports / "sample_summary.json").read_text(encoding="utf-8"))
    kinds = {issue["kind"] for issue in summary["issues"]}
    assert "class_count_mismatch" in kinds


@pytest.mark.unit
def test_corrupt_image_listed_in_csv(sample_dataset: dict[str, Any], tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    run_audit([_spec(sample_dataset["root"])], AuditConfig(), reports)
    assert "broken.jpg" in (reports / "corrupt_images.csv").read_text(encoding="utf-8")
