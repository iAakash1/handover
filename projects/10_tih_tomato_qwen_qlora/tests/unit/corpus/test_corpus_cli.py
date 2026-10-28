"""End-to-end CLI tests for the M3 language-layer commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.cli import main

_REAL_DKB = Path(__file__).resolve().parents[3] / "knowledge_base" / "dkb.json"


def _skip_if_no_dkb() -> None:
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_concepts_stats_only(capsys: pytest.CaptureFixture[str]) -> None:
    _skip_if_no_dkb()
    assert main(["concepts", "--stats-only"]) == 0
    assert json.loads(capsys.readouterr().out)["disease_count"] == 18


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_templates_validate_only(capsys: pytest.CaptureFixture[str]) -> None:
    _skip_if_no_dkb()
    assert main(["templates", "--validate-only"]) == 0
    assert "templates valid" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_generate_writes_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _skip_if_no_dkb()
    assert main(["generate", "--output", str(tmp_path)]) == 0
    names = {p.name for p in tmp_path.iterdir()}
    assert {"captions.json", "captions.jsonl", "captions.csv", "checksum.txt"} <= names
    assert "Caption corpus generated" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_generate_condition_filter(capsys: pytest.CaptureFixture[str]) -> None:
    _skip_if_no_dkb()
    assert main(["generate", "--condition", "tomato_early_blight", "--stats-only"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["disease_count"] == 1


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_validate_reports(capsys: pytest.CaptureFixture[str]) -> None:
    _skip_if_no_dkb()
    assert main(["validate", "--crop", "mango"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "valid"
    assert report["accepted"] > 0


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_corpus_exports(tmp_path: Path) -> None:
    _skip_if_no_dkb()
    assert main(["corpus", "--output", str(tmp_path), "--format", "paligemma"]) == 0
    assert (tmp_path / "captions.jsonl").is_file()
