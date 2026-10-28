"""Upload organization, immutability, sidecars, and the prediction log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app import storage

pytestmark = pytest.mark.unit


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all app storage roots into a temp dir for the test."""
    monkeypatch.setattr(storage, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(storage, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(storage, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(storage, "PREDICTIONS_LOG", tmp_path / "logs" / "predictions.jsonl")
    return tmp_path


def test_upload_subfolder_rules() -> None:
    assert storage.upload_subfolder("tomato_early_blight", "tomato", "confident") == "early_blight"
    # Not confident -> unknown bucket, even if a class was guessed.
    assert storage.upload_subfolder("tomato_early_blight", "tomato", "low_confidence") == "unknown"
    assert storage.upload_subfolder("unclassified", "tomato", "unknown") == "unknown"


def test_save_upload_organizes_by_class_and_never_overwrites(sandbox: Path) -> None:
    p1 = storage.save_upload(b"\xff\xd8\xffdata1", "leaf.JPG", "tomato", "early_blight")
    p2 = storage.save_upload(b"\xff\xd8\xffdata2", "leaf.JPG", "tomato", "early_blight")
    assert p1.parent == sandbox / "uploads" / "tomato" / "early_blight"
    assert p1 != p2  # unique names, never overwritten
    assert p1.read_bytes() == b"\xff\xd8\xffdata1"
    assert p2.read_bytes() == b"\xff\xd8\xffdata2"


def test_unknown_bucket(sandbox: Path) -> None:
    p = storage.save_upload(b"x", "f.png", "mango", "unknown")
    assert p.parent == sandbox / "uploads" / "mango" / "unknown"


def test_sidecar_written_next_to_image(sandbox: Path) -> None:
    p = storage.save_upload(b"x", "f.jpg", "tomato", "healthy")
    rec = {"disease_id": "tomato_healthy", "confidence": 0.9}
    sidecar = storage.write_sidecar(p, rec)
    assert sidecar == p.with_suffix(".json")
    assert json.loads(sidecar.read_text())["disease_id"] == "tomato_healthy"


def test_build_record_and_prediction_and_log(sandbox: Path) -> None:
    p = storage.save_upload(b"x", "f.jpg", "tomato", "early_blight")
    result = {
        "model": "m",
        "adapter": "checkpoints/qwen25vl_tomato_qlora",
        "run_name": "qwen25vl_tomato_qlora",
        "disease_id": "tomato_early_blight",
        "disease_name": "Early Blight",
        "status": "confident",
        "confidence": 0.93,
        "confidence_threshold": 0.55,
        "caption": "early blight lesions",
        "inference_seconds": 1.2,
        "generation_tokens": 20,
    }
    rec = storage.build_record(crop="tomato", image_path=p, original_name="f.jpg", result=result)
    assert rec["filename"] == p.name
    assert rec["status"] == "confident"
    assert rec["run_name"] == "qwen25vl_tomato_qlora"

    pred = storage.write_prediction(rec)
    assert pred.is_file()
    assert json.loads(pred.read_text())["prediction_json"] == str(pred)

    storage.append_prediction_log(rec)
    log = sandbox / "logs" / "predictions.jsonl"
    entry = json.loads(log.read_text().strip())
    assert entry["prediction"] == "tomato_early_blight"
    assert entry["confidence"] == 0.93
    assert entry["adapter"] == "qwen25vl_tomato_qlora"
    assert entry["latency_seconds"] == 1.2


def test_prediction_never_overwrites_same_second(sandbox: Path) -> None:
    rec = {"timestamp_slug": "2026-01-01_00-00-00", "disease_id": "x"}
    a = storage.write_prediction(dict(rec))
    b = storage.write_prediction(dict(rec))
    assert a != b and a.is_file() and b.is_file()


def test_markdown_report_contains_key_fields(sandbox: Path) -> None:
    rec = {
        "disease_name": "Early Blight",
        "crop": "tomato",
        "status": "confident",
        "confidence": 0.9,
        "caption": "c",
        "run_name": "qwen25vl_tomato_qlora",
    }
    md = storage.result_to_markdown(rec)
    assert "Early Blight" in md
    assert "qwen25vl_tomato_qlora" in md
    assert "90.0%" in md
