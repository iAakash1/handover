"""CLI surface + real end-to-end tests for ``plantdx vocabulary``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.cli import build_parser, main

_REAL_DKB = Path(__file__).resolve().parents[3] / "knowledge_base" / "dkb.json"


@pytest.mark.unit
def test_vocabulary_parser_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["vocabulary", "--output", "out", "--stats-only"])
    assert args.command == "vocabulary"
    assert args.output == "out"
    assert args.stats_only is True
    assert args.validate_only is False


@pytest.mark.unit
def test_vocabulary_no_longer_a_stub() -> None:
    """The old M1 stub raised NotImplementedError for every vocabulary invocation;
    it must now be a real, dispatched command (not exit code 2)."""
    parser = build_parser()
    args = parser.parse_args(["vocabulary", "--config", "does-not-exist.yaml"])
    assert args.command == "vocabulary"


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_vocabulary_stats_only_real_run(capsys: pytest.CaptureFixture[str]) -> None:
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")
    exit_code = main(["vocabulary", "--stats-only"])
    assert exit_code == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["validation_status"] == "valid"
    assert stats["vocabulary_item_count"] > 0
    assert stats["lexicon_item_count"] > 0


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_vocabulary_validate_only_real_run(capsys: pytest.CaptureFixture[str]) -> None:
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")
    exit_code = main(["vocabulary", "--validate-only"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "vocabulary valid" in out
    assert "checksum sha256:" in out


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_vocabulary_writes_all_six_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")
    out_dir = tmp_path / "vocabulary"
    exit_code = main(["vocabulary", "--output", str(out_dir)])
    assert exit_code == 0

    expected = {
        "vocabulary.json",
        "symptom_lexicon.json",
        "concept_index.json",
        "statistics.json",
        "checksum.txt",
        "validation_report.json",
    }
    assert {p.name for p in out_dir.iterdir()} == expected

    vocab_doc = json.loads((out_dir / "vocabulary.json").read_text())
    assert vocab_doc["kind"] == "plantdx.vocabulary"
    assert len(vocab_doc["items"]) > 0

    checksum_text = (out_dir / "checksum.txt").read_text().strip()
    assert checksum_text.startswith("sha256:")
    assert checksum_text in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_vocabulary_run_is_deterministic_across_invocations(tmp_path: Path) -> None:
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    assert main(["vocabulary", "--output", str(out1)]) == 0
    assert main(["vocabulary", "--output", str(out2)]) == 0
    for name in ("vocabulary.json", "symptom_lexicon.json", "checksum.txt"):
        assert (out1 / name).read_text() == (out2 / name).read_text()
