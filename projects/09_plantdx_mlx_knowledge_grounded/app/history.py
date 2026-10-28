"""Lightweight prediction history.

``predictions/<timestamp>.json`` holds the full record for each run; this module
maintains a compact index at ``predictions/history.json`` so the sidebar can
list past predictions without opening every file, and can reopen any one. All
reads are defensive — a missing or corrupt index never breaks the app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.utils import PREDICTIONS_DIR

_HISTORY_PATH = PREDICTIONS_DIR / "history.json"
_MAX_ENTRIES = 200


def _read_index() -> list[dict[str, Any]]:
    if not _HISTORY_PATH.is_file():
        return []
    try:
        data = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def append_history(record: dict[str, Any]) -> None:
    """Add a compact entry for ``record`` to the front of the index."""
    entry = {
        "timestamp": record.get("timestamp"),
        "crop": record.get("crop"),
        "filename": Path(record.get("image_path", "")).name,
        "disease_name": record.get("disease_name"),
        "disease_id": record.get("disease_id"),
        "confidence": record.get("confidence"),
        "image_path": record.get("image_path"),
        "prediction_json": record.get("prediction_json"),
    }
    index = [entry, *_read_index()][:_MAX_ENTRIES]
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def list_history() -> list[dict[str, Any]]:
    """Return history entries, newest first (index order)."""
    return _read_index()


def load_record(prediction_json: str | Path) -> dict[str, Any] | None:
    """Load a full prediction record by its JSON path, or ``None`` if unreadable."""
    path = Path(prediction_json)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None
