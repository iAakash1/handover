"""Shared constants, paths, and small pure helpers for the demo app.

No Streamlit, no model, no side effects at import time — just configuration and
formatting so the rest of the app stays declarative.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (repo-relative; the app lives at <repo>/app, entry point at <repo>)
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = REPO_ROOT / "uploads"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
LOGS_DIR = REPO_ROOT / "logs"
DKB_PATH = REPO_ROOT / "knowledge_base" / "dkb.json"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
REPORTS_DIR = REPO_ROOT / "reports"

# Append-only JSONL debug log of every prediction, and a rolling text log.
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"
APP_LOG = LOGS_DIR / "plantdx_app.log"

# --------------------------------------------------------------------------- #
# Confidence / robustness policy (heuristic, and user-tunable in the sidebar)
# --------------------------------------------------------------------------- #

# Confidence = mean probability of the model's own selected tokens (see
# app.inference). Below this, a named prediction is treated as low-confidence
# (likely out-of-distribution) rather than asserted. This is a heuristic proxy
# for generation certainty, NOT a calibrated out-of-distribution detector.
DEFAULT_CONFIDENCE_THRESHOLD = 0.55

# --------------------------------------------------------------------------- #
# Model / crop configuration — mirrors the trained runs, never re-derives them
# --------------------------------------------------------------------------- #

MODEL_ID = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
MODEL_DISPLAY = "Qwen2.5-VL 7B (4-bit)"
FINE_TUNING = "QLoRA"
FRAMEWORK = "MLX / mlx-vlm"

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
ALLOWED_UPLOAD_TYPES = ["jpg", "jpeg", "png"]

# The set of crops the demo can serve — one trained adapter each. Adapter dir
# and instruction match exactly what the training/evaluation pipeline uses.
CROPS = ("tomato", "mango")


@dataclass(frozen=True)
class CropProfile:
    """Everything the demo needs to run one crop's adapter."""

    crop: str
    label: str
    emoji: str
    adapter_dir: Path
    instruction: str
    run_name: str


def crop_profile(crop: str) -> CropProfile:
    """Return the :class:`CropProfile` for ``crop`` (``tomato`` | ``mango``)."""
    key = crop.strip().lower()
    if key not in CROPS:
        raise ValueError(f"unsupported crop {crop!r}; expected one of {CROPS}")
    run_name = f"qwen25vl_{key}_qlora"
    return CropProfile(
        crop=key,
        label=key.capitalize(),
        emoji="🍅" if key == "tomato" else "🥭",
        adapter_dir=CHECKPOINTS_DIR / run_name,
        instruction=f"Describe the visible condition of this {key} leaf.",
        run_name=run_name,
    )


# --------------------------------------------------------------------------- #
# Filenames / timestamps
# --------------------------------------------------------------------------- #

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> datetime:
    """Timezone-aware current UTC time (single source for timestamps)."""
    return datetime.now(timezone.utc)


def timestamp_slug(when: datetime | None = None) -> str:
    """A filesystem- and sort-friendly timestamp, e.g. ``2026-07-16_17-20-31``."""
    return (when or utc_now()).strftime("%Y-%m-%d_%H-%M-%S")


def sanitize_filename(name: str) -> str:
    """Reduce an arbitrary upload name to a safe, bounded basename."""
    stem = Path(name).name.strip() or "image"
    cleaned = _SAFE_NAME.sub("_", stem).strip("._") or "image"
    return cleaned[:80]


def unique_filename(original_name: str, when: datetime | None = None) -> str:
    """``<timestamp>_<uuid8>_<original>`` — collision-free by construction."""
    return f"{timestamp_slug(when)}_{uuid.uuid4().hex[:8]}_{sanitize_filename(original_name)}"


def is_supported_upload(name: str) -> bool:
    """Whether ``name`` has a supported image extension."""
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #

# Display-only thresholds; not a scientific calibration.
_CONFIDENCE_HIGH = 0.75
_CONFIDENCE_MID = 0.50


def confidence_band(confidence: float | None) -> tuple[str, str]:
    """Return ``(label, hex_color)`` for a confidence value, or a neutral pair."""
    if confidence is None:
        return "n/a", "#6b7280"
    if confidence >= _CONFIDENCE_HIGH:
        return "high", "#16a34a"  # green
    if confidence >= _CONFIDENCE_MID:
        return "moderate", "#f59e0b"  # orange
    return "low", "#dc2626"  # red


def pretty_disease(disease_id: str) -> str:
    """Turn ``tomato_early_blight`` into ``Early Blight`` for display."""
    if not disease_id or disease_id == "unclassified":
        return "Unclassified"
    body = disease_id.split("_", 1)[1] if "_" in disease_id else disease_id
    return body.replace("_", " ").title()


def class_folder(disease_id: str, crop: str) -> str:
    """The upload subfolder for a predicted disease id.

    ``tomato_early_blight`` -> ``early_blight``; an unclassified/empty id, or an
    id whose crop prefix doesn't match ``crop``, -> ``unknown``.
    """
    if not disease_id or disease_id == "unclassified":
        return "unknown"
    prefix = f"{crop.lower()}_"
    if disease_id.startswith(prefix):
        return disease_id[len(prefix) :]
    return "unknown"


def strip_crop_suffix(disease_field: str) -> str:
    """Drop a trailing ``(tomato)``-style parenthetical from a DKB disease name."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", disease_field or "").strip()


def format_seconds(seconds: float) -> str:
    """Human-friendly elapsed time (``820 ms`` / ``3.4 s``)."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"
