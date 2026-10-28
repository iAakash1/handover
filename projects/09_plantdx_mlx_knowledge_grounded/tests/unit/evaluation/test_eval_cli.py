"""`plantdx evaluate` CLI: argument parsing, integrity failure, missing predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.cli import build_parser, main


@pytest.mark.unit
def test_evaluate_parses_with_defaults() -> None:
    args = build_parser().parse_args(["evaluate"])
    assert args.command == "evaluate"
    assert args.stage == "all"
    assert args.split == "test"
    assert args.seed == 20260711


@pytest.mark.unit
def test_evaluate_parses_all_flags() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--stage",
            "analyze",
            "--adapter",
            "/a.safetensors",
            "--dataset",
            "/d",
            "--split",
            "validation",
            "--model",
            "m",
            "--output-dir",
            "/o",
            "--batch-size",
            "4",
            "--max-samples",
            "10",
            "--seed",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert args.stage == "analyze"
    assert args.adapter == "/a.safetensors"
    assert args.max_samples == 10


@pytest.mark.unit
def test_evaluate_rejects_invalid_stage() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "--stage", "bogus"])


@pytest.mark.unit
def test_evaluate_analyze_missing_predictions_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "evaluate",
            "--stage",
            "analyze",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "predictions file not found" in capsys.readouterr().err


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_evaluate_inference_fails_on_split_integrity_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    with (dataset_dir / "train.jsonl").open("w") as fh:
        fh.write(json.dumps({"image": "/a.JPG", "question": "q", "answer": "a"}) + "\n")
    with (dataset_dir / "test.jsonl").open("w") as fh:
        fh.write(json.dumps({"image": "/a.JPG", "question": "q", "answer": "a"}) + "\n")  # leaked

    code = main(
        [
            "evaluate",
            "--stage",
            "inference",
            "--dataset",
            str(dataset_dir),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 1
    assert "contaminated" in capsys.readouterr().err
