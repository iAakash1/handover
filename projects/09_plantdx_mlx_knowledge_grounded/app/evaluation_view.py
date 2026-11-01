"""Read a crop's held-out evaluation results for display in the app.

Reads frozen ``plantdx evaluate`` artifacts (never recomputes them). A crop can
have more than one evaluation: the July 2026 run in ``reports/<run_name>/evaluation/``
and the November 2026 retraining in ``retrained_2026-11/<run_name>/``. The one shown
is the evaluation of the adapter weights present in ``checkpoints/<run_name>/``: the
folder whose recorded ``adapter_checksum`` (``system_info.json``) equals the weights'
SHA-256. Without a match (or without weights) it falls back to the July report and
says so. Returns a compact summary so the UI can show, honestly, how the adapter
scored on its own test split — and remind the viewer that those numbers are on
PlantVillage-style images.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.utils import REPO_ROOT, REPORTS_DIR, crop_profile

RETRAINED_DIR = REPO_ROOT / "retrained_2026-11"


@dataclass(frozen=True)
class EvalSummary:
    """A crop's base-vs-fine-tuned headline metrics on its held-out test split."""

    crop: str
    run_name: str
    sample_count: int
    base_accuracy: float
    finetuned_accuracy: float
    base_f1_macro: float
    finetuned_f1_macro: float
    per_disease_available: bool
    source: str = ""  # the evaluation folder shown, relative to the project when possible
    matches_weights: bool = False  # its recorded checksum equals the weights in checkpoints/


def evaluation_dir(crop: str) -> tuple[Path, bool]:
    """The evaluation folder to show for ``crop``, and whether it evaluated the present weights."""
    profile = crop_profile(crop)
    july = REPORTS_DIR / profile.run_name / "evaluation"
    weights = profile.adapter_dir / "adapters.safetensors"
    if weights.is_file():
        checksum = _sha256(weights)
        run_dir = RETRAINED_DIR / profile.run_name
        for candidate in (july, run_dir / "evaluation", *sorted(run_dir.glob("*/evaluation"))):
            if _recorded_checksum(candidate) == checksum:
                return candidate, True
    return july, False


def load_eval_summary(crop: str) -> EvalSummary | None:
    """Return the crop's evaluation summary, or ``None`` if it hasn't been run."""
    profile = crop_profile(crop)
    eval_dir, matches = evaluation_dir(crop)
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        base = metrics["base"]["classification"]
        ft = metrics["finetuned"]["classification"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    return EvalSummary(
        crop=crop,
        run_name=profile.run_name,
        sample_count=int(base.get("sample_count", 0)),
        base_accuracy=float(base.get("accuracy", 0.0)),
        finetuned_accuracy=float(ft.get("accuracy", 0.0)),
        base_f1_macro=float(base.get("f1_macro", 0.0)),
        finetuned_f1_macro=float(ft.get("f1_macro", 0.0)),
        per_disease_available=(eval_dir / "per_disease.csv").is_file(),
        source=_display_path(eval_dir),
        matches_weights=matches,
    )


def load_per_disease(crop: str) -> list[dict[str, Any]]:
    """Fine-tuned per-disease rows (disease_id, accuracy, f1, sample_count)."""
    import csv

    path = evaluation_dir(crop)[0] / "per_disease.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") != "finetuned":
                continue
            rows.append(
                {
                    "disease_id": row.get("disease_id", ""),
                    "sample_count": int(float(row.get("sample_count", 0) or 0)),
                    "accuracy": float(row.get("accuracy", 0) or 0),
                    "f1": float(row.get("f1", 0) or 0),
                }
            )
    return rows


def _recorded_checksum(eval_dir: Path) -> str | None:
    """The adapter SHA-256 an evaluation recorded in its ``system_info.json``."""
    try:
        info = json.loads((eval_dir / "system_info.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = str(info.get("adapter_checksum") or "").removeprefix("sha256:")
    return value or None


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    """SHA-256 of a file, cached per (path, size, mtime) so page reruns don't rehash 160 MB."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
