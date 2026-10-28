"""Read a crop's held-out evaluation results for display in the app.

Reads the frozen ``reports/<run_name>/evaluation/`` artifacts produced by
``plantdx evaluate`` (never recomputes them). Returns a compact summary so the
UI can show, honestly, how the fine-tuned adapter scored on its own test split —
and remind the viewer that those numbers are on PlantVillage-style images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.utils import REPORTS_DIR, crop_profile


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


def load_eval_summary(crop: str) -> EvalSummary | None:
    """Return the crop's evaluation summary, or ``None`` if it hasn't been run."""
    profile = crop_profile(crop)
    eval_dir = REPORTS_DIR / profile.run_name / "evaluation"
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
    )


def load_per_disease(crop: str) -> list[dict[str, Any]]:
    """Fine-tuned per-disease rows (disease_id, accuracy, f1, sample_count)."""
    import csv

    profile = crop_profile(crop)
    path = REPORTS_DIR / profile.run_name / "evaluation" / "per_disease.csv"
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
