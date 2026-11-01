"""History index round-trip and evaluation-report reading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from app import evaluation_view, history, utils

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


def _write_eval(eval_dir: Path, accuracy: float, checksum: str | None = None) -> None:
    eval_dir.mkdir(parents=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps(
            {
                "base": {
                    "classification": {"accuracy": 0.07, "f1_macro": 0.03, "sample_count": 910}
                },
                "finetuned": {"classification": {"accuracy": accuracy, "f1_macro": 0.92}},
            }
        ),
        encoding="utf-8",
    )
    if checksum is not None:
        (eval_dir / "system_info.json").write_text(
            json.dumps({"adapter_checksum": f"sha256:{checksum}"}), encoding="utf-8"
        )


@pytest.fixture
def eval_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the July reports, the retraining record and the checkpoints at ``tmp_path``."""
    monkeypatch.setattr(evaluation_view, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(evaluation_view, "RETRAINED_DIR", tmp_path / "retrained")
    monkeypatch.setattr(utils, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    return tmp_path


def _write_weights(root: Path, data: bytes) -> str:
    weights = root / "checkpoints" / "qwen25vl_tomato_qlora" / "adapters.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_eval_summary_reads_metrics(eval_roots: Path) -> None:
    _write_eval(eval_roots / "reports" / "qwen25vl_tomato_qlora" / "evaluation", 0.94)
    summary = evaluation_view.load_eval_summary("tomato")
    assert summary is not None
    assert summary.sample_count == 910
    assert summary.finetuned_accuracy == pytest.approx(0.94)
    assert summary.base_accuracy == pytest.approx(0.07)
    assert summary.matches_weights is False  # no weights present: the July report, flagged


def test_eval_summary_follows_the_present_weights(eval_roots: Path) -> None:
    checksum = _write_weights(eval_roots, b"retrained weights")
    _write_eval(eval_roots / "reports" / "qwen25vl_tomato_qlora" / "evaluation", 0.94, "0" * 64)
    retrained = eval_roots / "retrained" / "qwen25vl_tomato_qlora"
    _write_eval(retrained / "evaluation", 0.91, checksum)
    _write_eval(retrained / "final_iteration_16201" / "evaluation", 0.87, "1" * 64)
    summary = evaluation_view.load_eval_summary("tomato")
    assert summary is not None
    assert summary.finetuned_accuracy == pytest.approx(0.91)
    assert summary.matches_weights is True
    assert summary.source.endswith("evaluation")


def test_eval_summary_unmatched_weights_fall_back_to_july(eval_roots: Path) -> None:
    _write_weights(eval_roots, b"some other weights")
    _write_eval(eval_roots / "reports" / "qwen25vl_tomato_qlora" / "evaluation", 0.94, "0" * 64)
    summary = evaluation_view.load_eval_summary("tomato")
    assert summary is not None
    assert summary.finetuned_accuracy == pytest.approx(0.94)
    assert summary.matches_weights is False


def test_eval_summary_absent_returns_none(eval_roots: Path) -> None:
    assert evaluation_view.load_eval_summary("mango") is None
