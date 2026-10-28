"""Filesystem storage for the demo: organized immutable uploads, records, logs.

Layout (all gitignored, local-only):

    uploads/<crop>/<predicted_class|unknown>/<timestamp_uuid_name>.<ext>
    uploads/<crop>/<...>/<timestamp_uuid_name>.json      # metadata sidecar
    predictions/<timestamp>.json                          # full record (reopenable)
    predictions/history.json                              # compact sidebar index
    logs/predictions.jsonl                                # append-only debug log

Uploaded images are written exactly once, to the folder of their predicted class
(or ``unknown``), and never overwritten or moved. Nothing here touches datasets/,
artifacts/, or checkpoints/.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.logging_setup import get_logger
from app.utils import (
    LOGS_DIR,
    PREDICTIONS_DIR,
    PREDICTIONS_LOG,
    UPLOADS_DIR,
    class_folder,
    crop_profile,
    format_seconds,
    pretty_disease,
    timestamp_slug,
    unique_filename,
    utc_now,
)

_CROPS = ("tomato", "mango")
_log = get_logger()


def ensure_dirs() -> None:
    """Create the top-level upload/prediction/log directories (idempotent)."""
    for crop in _CROPS:
        (UPLOADS_DIR / crop).mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def upload_subfolder(disease_id: str, crop: str, status: str) -> str:
    """The class subfolder an upload belongs in: its class, or ``unknown``.

    Only confident predictions are filed under their disease class; low-confidence
    and unknown predictions go to ``unknown`` (we're not sure enough to label them).
    """
    if status == "confident":
        return class_folder(disease_id, crop)
    return "unknown"


def save_upload(data: bytes, original_name: str, crop: str, subfolder: str) -> Path:
    """Save uploaded bytes immutably under ``uploads/<crop>/<subfolder>/``.

    The filename is unique (timestamp + uuid + sanitized original), so an existing
    file is never overwritten.
    """
    ensure_dirs()
    crop_key = crop if crop in _CROPS else "unknown"
    target_dir = UPLOADS_DIR / crop_key / (subfolder or "unknown")
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / unique_filename(original_name)
    while path.exists():  # astronomically unlikely uuid collision — never clobber
        path = target_dir / unique_filename(original_name)
    path.write_bytes(data)
    return path


def write_sidecar(image_path: Path, record: dict[str, Any]) -> Path:
    """Write a ``<image>.json`` metadata sidecar next to a saved upload."""
    sidecar = image_path.with_suffix(".json")
    sidecar.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar


def build_record(
    *,
    crop: str,
    image_path: Path,
    original_name: str,
    result: dict[str, Any],
    adapter: dict[str, Any] | None = None,
    when: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a serializable prediction record from an inference result."""
    when = when or utc_now()
    profile = crop_profile(crop) if crop in _CROPS else None
    return {
        "timestamp": when.isoformat(),
        "timestamp_slug": timestamp_slug(when),
        "crop": crop,
        "original_name": original_name,
        "filename": image_path.name,
        "image_path": str(image_path),
        "model": result.get("model"),
        "adapter": result.get("adapter"),
        "run_name": result.get("run_name") or (profile.run_name if profile else None),
        "instruction": result.get("instruction"),
        "disease_id": result.get("disease_id"),
        "disease_name": result.get("disease_name"),
        "common_name": result.get("common_name"),
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "confidence_threshold": result.get("confidence_threshold"),
        "caption": result.get("caption"),
        "diagnosis": result.get("diagnosis"),
        "symptoms": result.get("symptoms", []),
        "inference_seconds": result.get("inference_seconds"),
        "generation_tokens": result.get("generation_tokens"),
        "peak_memory_gb": result.get("peak_memory_gb"),
        "adapter_info": adapter,
    }


def write_prediction(record: dict[str, Any]) -> Path:
    """Persist a full prediction record to ``predictions/<timestamp>.json``."""
    ensure_dirs()
    slug = record.get("timestamp_slug") or timestamp_slug()
    path = PREDICTIONS_DIR / f"{slug}.json"
    counter = 2
    while path.exists():
        path = PREDICTIONS_DIR / f"{slug}_{counter}.json"
        counter += 1
    record = {**record, "prediction_json": str(path)}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_prediction_log(record: dict[str, Any]) -> None:
    """Append one compact line to ``logs/predictions.jsonl`` (never fatal)."""
    entry = {
        "timestamp": record.get("timestamp"),
        "filename": record.get("filename"),
        "crop": record.get("crop"),
        "prediction": record.get("disease_id"),
        "status": record.get("status"),
        "confidence": record.get("confidence"),
        "generation": record.get("caption"),
        "adapter": record.get("run_name"),
        "latency_seconds": record.get("inference_seconds"),
        "model": record.get("model"),
    }
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        _log.warning("could not append prediction log: %s", exc)


def result_to_json(record: dict[str, Any]) -> str:
    """Pretty JSON string of a record, for the Download Result button."""
    return json.dumps(record, indent=2, ensure_ascii=False)


def result_to_markdown(record: dict[str, Any]) -> str:
    """A human-readable Markdown report of one prediction."""
    disease = record.get("disease_name") or pretty_disease(record.get("disease_id") or "")
    conf = record.get("confidence")
    conf_line = f"{conf * 100:.1f}%" if isinstance(conf, int | float) else "not available"
    secs = record.get("inference_seconds")
    time_line = format_seconds(secs) if isinstance(secs, int | float) else "—"
    symptoms = record.get("symptoms") or []

    lines = [
        f"# PlantDx Diagnosis Report — {disease}",
        "",
        f"- **Crop:** {str(record.get('crop', '')).capitalize()}",
        f"- **Predicted condition:** {disease}",
        f"- **Status:** {record.get('status', '—')}",
    ]
    if record.get("common_name"):
        lines.append(f"- **Common name:** {record['common_name']}")
    lines += [
        f"- **Confidence:** {conf_line}",
        f"- **Inference time:** {time_line}",
        f"- **Model:** {record.get('model', '—')}",
        f"- **Adapter (run):** {record.get('run_name', '—')}",
        f"- **Adapter path:** `{record.get('adapter', '—')}`",
        f"- **Timestamp (UTC):** {record.get('timestamp', '—')}",
        f"- **Image:** `{record.get('image_path', '—')}`",
        "",
        "## Generated diagnosis",
        "",
        record.get("diagnosis") or "_none_",
        "",
        "## Generated caption",
        "",
        record.get("caption") or "_none_",
        "",
        "## Documented symptoms (from the Disease Knowledge Base)",
        "",
    ]
    lines += [f"- {s}" for s in symptoms] or ["_none recorded_"]
    lines += [
        "",
        "---",
        "_Generated by the PlantDx demo application. Symptoms are read verbatim "
        "from the curated Disease Knowledge Base, not generated by the model._",
        "",
    ]
    return "\n".join(lines)
