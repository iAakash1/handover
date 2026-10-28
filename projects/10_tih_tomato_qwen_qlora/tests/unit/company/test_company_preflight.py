"""Preflight: what gates the Train button, and what only warns."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plantdx.company.preflight import FAIL, OK, WARN, Check, PreflightReport, run_preflight


@pytest.mark.unit
def test_report_is_ready_only_when_no_critical_check_failed() -> None:
    ready = PreflightReport(checks=[Check("a", OK, ""), Check("b", WARN, "", critical=False)])
    assert ready.ready is True
    assert ready.blocking == []
    assert len(ready.warnings) == 1

    blocked = PreflightReport(checks=[Check("a", OK, ""), Check("b", FAIL, "boom")])
    assert blocked.ready is False
    assert [c.name for c in blocked.blocking] == ["b"]


@pytest.mark.unit
def test_a_non_critical_failure_does_not_block() -> None:
    report = PreflightReport(checks=[Check("optional", FAIL, "missing", critical=False)])
    assert report.ready is True
    assert len(report.warnings) == 1


@pytest.mark.unit
def test_preflight_runs_on_the_miniature_dataset(
    train_config_path: Path, profile_path: Path
) -> None:
    report = run_preflight(
        config_path=train_config_path, profile_path=profile_path, local_files_only=False
    )
    names = {c.name for c in report.checks}
    assert {"Training config", "Company dataset", "Class folders", "Images"} <= names
    by_name = {c.name: c for c in report.checks}
    assert by_name["Training config"].status == OK
    assert by_name["Company dataset"].status == OK
    assert by_name["Class folders"].status == FAIL  # miniature fixture has only three classes
    assert by_name["Images"].status == OK
    assert by_name["Dataset isolation"].status == OK


@pytest.mark.unit
def test_missing_dataset_root_fails_closed(tmp_path: Path, train_config_path: Path) -> None:
    payload = {
        "dataset": {"name": "x", "crop": "tomato", "root": str(tmp_path / "gone")},
        "classes": [{"raw": "Aphid", "class_id": "aphid", "label": "Aphid"}],
    }
    bad = tmp_path / "bad_profile.yaml"
    bad.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = run_preflight(config_path=train_config_path, profile_path=bad, local_files_only=False)
    assert report.ready is False
    assert any(c.name == "Company dataset" and c.status == FAIL for c in report.blocking)


@pytest.mark.unit
def test_an_empty_class_folder_blocks_training(
    train_config_path: Path, profile_path: Path, raw_dataset: Path
) -> None:
    for path in (raw_dataset / "Healthy_Leaf").iterdir():
        path.unlink()
    report = run_preflight(
        config_path=train_config_path, profile_path=profile_path, local_files_only=False
    )
    images = next(c for c in report.checks if c.name == "Images")
    assert images.status == FAIL
    assert "healthy_leaf" in images.detail


@pytest.mark.unit
def test_an_unmapped_folder_blocks(
    train_config_path: Path, profile_path: Path, raw_dataset: Path
) -> None:
    (raw_dataset / "Septoria_Leaf_Spot").mkdir()
    report = run_preflight(
        config_path=train_config_path, profile_path=profile_path, local_files_only=False
    )
    check = next(c for c in report.checks if c.name == "Unmapped folders")
    assert check.status == FAIL
    assert check.critical is True
    assert "Septoria_Leaf_Spot" in check.detail


@pytest.mark.unit
def test_unsupported_extensions_warn_but_do_not_block(
    train_config_path: Path, profile_path: Path, raw_dataset: Path
) -> None:
    (raw_dataset / "Aphid" / "readme.txt").write_text("x", encoding="utf-8")
    report = run_preflight(
        config_path=train_config_path, profile_path=profile_path, local_files_only=False
    )
    check = next(c for c in report.checks if c.name == "Image formats")
    assert check.status == WARN
    assert check.critical is False


@pytest.mark.unit
def test_a_broken_config_becomes_a_failed_check_not_an_exception(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("run_name: x\n", encoding="utf-8")
    report = run_preflight(config_path=bad, local_files_only=False)
    assert report.ready is False
    assert any(c.status == FAIL for c in report.checks)


@pytest.mark.unit
def test_prepared_dataset_checks_catch_a_leaked_path(
    train_config_path: Path, profile_path: Path, company_config, company_dataset
) -> None:
    """A PlantVillage row inside a prepared split must fail the provenance check."""
    import json

    from plantdx.company.dataset import prepare_dataset

    prepared = prepare_dataset(company_config, company_dataset)
    train_file = prepared.dataset_dir / "train.jsonl"
    rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines()]
    rows.append(
        {
            "image": str(company_dataset.root / "PlantVillage" / "leak.JPG"),
            "question": "q",
            "answer": "Aphid",
            "class_id": "aphid",
        }
    )
    train_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = run_preflight(
        config_path=train_config_path, profile_path=profile_path, local_files_only=False
    )
    check = next(c for c in report.checks if c.name == "Training-set provenance")
    assert check.status == FAIL
    assert check.critical is True
    assert report.ready is False


@pytest.mark.unit
def test_prepared_dataset_checks_catch_an_off_vocabulary_label(
    train_config_path: Path, profile_path: Path, company_config, company_dataset
) -> None:
    import json

    from plantdx.company.dataset import prepare_dataset

    prepared = prepare_dataset(company_config, company_dataset)
    train_file = prepared.dataset_dir / "train.jsonl"
    rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines()]
    rows[0]["answer"] = "Septoria Leaf Spot"
    train_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = run_preflight(
        config_path=train_config_path, profile_path=profile_path, local_files_only=False
    )
    check = next(c for c in report.checks if c.name == "Label vocabulary")
    assert check.status == FAIL
    assert report.ready is False
