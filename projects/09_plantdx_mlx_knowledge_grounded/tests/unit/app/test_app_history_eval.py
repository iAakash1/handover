"""History index round-trip and evaluation-report reading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app import evaluation_view, history

pytestmark = pytest.mark.unit


def test_history_append_list_reopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(history, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(history, "_HISTORY_PATH", tmp_path / "history.json")

    pred = tmp_path / "2026-01-01_00-00-00.json"
    record = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "crop": "tomato",
        "image_path": "/x/leaf.jpg",
        "disease_name": "Early Blight",
        "disease_id": "tomato_early_blight",
        "confidence": 0.9,
        "prediction_json": str(pred),
    }
    pred.write_text(json.dumps(record), encoding="utf-8")
    history.append_history(record)

    entries = history.list_history()
    assert len(entries) == 1
    assert entries[0]["disease_id"] == "tomato_early_blight"
    reopened = history.load_record(entries[0]["prediction_json"])
    assert reopened is not None
    assert reopened["disease_name"] == "Early Blight"


def test_history_survives_missing_and_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(history, "_HISTORY_PATH", tmp_path / "history.json")
    assert history.list_history() == []  # missing
    (tmp_path / "history.json").write_text("{ not json", encoding="utf-8")
    assert history.list_history() == []  # corrupt -> [] not a crash
    assert history.load_record(tmp_path / "nope.json") is None


def test_eval_summary_reads_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation_view, "REPORTS_DIR", tmp_path)
    eval_dir = tmp_path / "qwen25vl_tomato_qlora" / "evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps(
            {
                "base": {
                    "classification": {"accuracy": 0.07, "f1_macro": 0.03, "sample_count": 910}
                },
                "finetuned": {"classification": {"accuracy": 0.94, "f1_macro": 0.92}},
            }
        ),
        encoding="utf-8",
    )
    summary = evaluation_view.load_eval_summary("tomato")
    assert summary is not None
    assert summary.sample_count == 910
    assert summary.finetuned_accuracy == pytest.approx(0.94)
    assert summary.base_accuracy == pytest.approx(0.07)


def test_eval_summary_absent_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation_view, "REPORTS_DIR", tmp_path)
    assert evaluation_view.load_eval_summary("mango") is None
