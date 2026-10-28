#!/usr/bin/env python
"""Render presentation-quality README figures from EXISTING evaluation outputs.

Reads the real, already-generated CSV/JSON under ``reports/<run>/evaluation/``
(never recomputes a metric) and renders larger-font, opaque-white-background
figures — readable in both GitHub light and dark themes — into ``docs/images/``.

Plotted values are exactly those in the evaluation reports; only the rendering
(font size, DPI, background) differs from the pipeline's own CVD-safe figures.

Usage (in an environment with matplotlib, e.g. the ``[eval]`` extra):

    python scripts/render_readme_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "docs" / "images"
_CROPS = ("tomato", "mango")

# A calm, accessible blue ramp + two fixed series colors (base vs fine-tuned).
_HEAT = "Blues"
_BASE = "#94a3b8"
_FT = "#16a34a"


def _pretty(disease_id: str) -> str:
    body = disease_id.split("_", 1)[1] if "_" in disease_id else disease_id
    return body.replace("_", " ").title()


def _eval_dir(crop: str) -> Path:
    return _ROOT / "reports" / f"qwen25vl_{crop}_qlora" / "evaluation"


def _style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig: plt.Figure, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote docs/images/{name}")


def render_confusion_matrix(crop: str) -> None:
    """Heatmap of the fine-tuned confusion matrix for ``crop``."""
    path = _eval_dir(crop) / "confusion_matrix_finetuned.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    labels = [_pretty(c) for c in rows[0][1:]]
    matrix = [[int(v) for v in r[1:]] for r in rows[1:]]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.0 + n * 0.72, 1.0 + n * 0.72))
    _style(fig, ax)
    im = ax.imshow(matrix, cmap=_HEAT, aspect="equal")
    vmax = max((max(r) for r in matrix), default=1)
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if val > vmax * 0.55 else "#1f2937",
                    fontweight="bold",
                )
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(n), labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("True", fontsize=13, fontweight="bold")
    ax.set_title(f"{crop.capitalize()} — Confusion Matrix (fine-tuned)", fontsize=15, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, f"{crop}_confusion_matrix.png")


def render_per_disease_f1(crop: str) -> None:
    """Horizontal bar chart of fine-tuned per-disease F1 for ``crop``."""
    path = _eval_dir(crop) / "per_disease.csv"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") == "finetuned":
                rows.append((_pretty(row["disease_id"]), float(row["f1"])))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    f1s = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 0.6 + len(rows) * 0.52))
    _style(fig, ax)
    bars = ax.barh(names, f1s, color=_FT, height=0.62)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in f1s], padding=4, fontsize=11)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 score", fontsize=13, fontweight="bold")
    ax.set_title(f"{crop.capitalize()} — Per-Disease F1 (fine-tuned)", fontsize=15, pad=12)
    ax.tick_params(labelsize=11)
    _save(fig, f"{crop}_per_disease_f1.png")


def render_metrics_comparison(crop: str) -> None:
    """Grouped base-vs-fine-tuned bars (accuracy / balanced / macro / weighted F1)."""
    metrics = json.loads((_eval_dir(crop) / "metrics.json").read_text(encoding="utf-8"))
    b = metrics["base"]["classification"]
    f = metrics["finetuned"]["classification"]
    names = ["Accuracy", "Balanced\naccuracy", "Macro-F1", "Weighted-F1"]
    keys = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted"]
    base = [b[k] for k in keys]
    ft = [f[k] for k in keys]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _style(fig, ax)
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], base, w, label="Base", color=_BASE)
    b2 = ax.bar([i + w / 2 for i in x], ft, w, label="Fine-tuned", color=_FT)
    ax.bar_label(b1, labels=[f"{v:.2f}" for v in base], padding=3, fontsize=10)
    ax.bar_label(b2, labels=[f"{v:.2f}" for v in ft], padding=3, fontsize=10)
    ax.set_xticks(list(x), names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title(f"{crop.capitalize()} — Base vs Fine-tuned", fontsize=15, pad=12)
    ax.legend(fontsize=11, frameon=False)
    _save(fig, f"{crop}_metrics_comparison.png")


def main() -> None:
    """Render every README figure for every crop."""
    for crop in _CROPS:
        print(f"[{crop}]")
        render_confusion_matrix(crop)
        render_per_disease_f1(crop)
        render_metrics_comparison(crop)


if __name__ == "__main__":
    main()
