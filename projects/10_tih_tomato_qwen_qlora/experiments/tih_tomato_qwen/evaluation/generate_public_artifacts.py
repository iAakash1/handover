"""Build aggregate, publication-safe TIH benchmark tables and figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "tih_tomato_qwen"
INTERNAL_SOURCE = ROOT / "reports" / "tih_tomato_moreaccurate" / "test_finetuned"
EXTERNAL_SOURCE = ROOT / "reports" / "tih_tomato_moreaccurate_external"
TRAINING_SOURCE = ROOT / "logs" / "tih_tomato_moreaccurate" / "loss_history.jsonl"
TRAINING_SUMMARY_SOURCE = ROOT / "reports" / "tih_tomato_moreaccurate" / "training_summary.json"


def main() -> None:
    """Write aggregate metrics, per-class tables, and benchmark figures."""
    internal = _read_json(INTERNAL_SOURCE / "metrics.json")
    external = _read_json(EXTERNAL_SOURCE / "metrics.json")
    training_summary = _read_json(TRAINING_SUMMARY_SOURCE)

    internal_dir = EXPERIMENT / "results" / "internal"
    external_dir = EXPERIMENT / "results" / "external"
    graph_dir = EXPERIMENT / "graphs"
    for directory in (internal_dir, external_dir, graph_dir):
        directory.mkdir(parents=True, exist_ok=True)

    internal_public = _public_internal_metrics(internal)
    external_public = _public_external_metrics(external)
    _write_json(internal_dir / "metrics.json", internal_public)
    _write_json(external_dir / "metrics.json", external_public)
    _write_json(
        internal_dir / "training_summary.json",
        {
            "run_name": training_summary["run_name"],
            "steps": training_summary["steps"],
            "epochs": training_summary["epochs"],
            "final_loss": training_summary["final_loss"],
            "train_seconds": training_summary["train_seconds"],
            "train_samples": training_summary["train_samples"],
            "validation_samples": training_summary["validation_samples"],
            "test_samples": training_summary["test_samples"],
            "trainable_params": training_summary["trainable_params"],
            "total_params": training_summary["total_params"],
            "peak_memory_gb": training_summary["peak_memory_gb"],
        },
    )

    internal_rows = [_public_class_row(row) for row in internal["per_class"]]
    external_rows = _read_csv(EXTERNAL_SOURCE / "classification_report.csv")
    _write_csv(internal_dir / "per_class_metrics.csv", internal_rows)
    _write_csv(external_dir / "per_class_metrics.csv", external_rows)

    internal_labels = [row["label"] for row in internal["per_class"]]
    internal_matrix = np.asarray(internal["confusion_matrix"], dtype=int)[:, :-1]
    external_labels, external_matrix = _external_confusion(internal_labels)
    _write_matrix_csv(internal_dir / "confusion_matrix.csv", internal_labels, internal_matrix)
    _write_matrix_csv(external_dir / "confusion_matrix.csv", external_labels, external_matrix)

    _confusion_plot(
        internal_matrix,
        internal_labels,
        graph_dir / "tih_qwen_internal_confusion_matrix.png",
        "Internal test confusion matrix",
    )
    _confusion_plot(
        external_matrix,
        external_labels,
        graph_dir / "tih_qwen_external_confusion_matrix.png",
        "External domain-shift confusion matrix",
    )
    _per_class_plot(
        internal_rows,
        external_rows,
        graph_dir / "tih_qwen_per_class_f1.png",
    )
    _training_plot(graph_dir / "tih_qwen_training_metrics.png")
    _benchmark_plot(
        internal_public,
        external_public,
        graph_dir / "tih_qwen_benchmark_overview.png",
    )


def _public_internal_metrics(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": data["run_name"],
        "split": data["split"],
        "model": data["base_model"],
        "adapter": "checkpoints/tih_tomato_moreaccurate",
        "total_images": data["samples"],
        "number_of_classes": len(data["per_class"]),
        "accuracy": data["accuracy"],
        "macro_precision": data["precision_macro"],
        "macro_recall": data["recall_macro"],
        "macro_f1": data["f1_macro"],
        "weighted_f1": data["f1_weighted"],
        "average_latency_seconds": data["seconds_per_sample"],
        "unparseable_predictions": data["unparseable"],
    }


def _public_external_metrics(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_images",
        "number_of_classes",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "average_latency_seconds",
        "unparseable_predictions",
        "detected_classes",
        "folder_mapping",
        "ignored_folders",
    )
    public = {key: data[key] for key in keys}
    public.update(
        {
            "dataset": "Dataset/testing",
            "model": data["model"],
            "adapter": "checkpoints/tih_tomato_moreaccurate",
            "evaluation_role": "external domain-shift robustness benchmark",
        }
    )
    return public


def _public_class_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "class": row["label"],
        "support": row["support"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1-score": row["f1"],
    }


def _external_confusion(base_labels: list[str]) -> tuple[list[str], np.ndarray[Any, Any]]:
    rows = _read_csv(EXTERNAL_SOURCE / "predictions.csv")
    present = {row["ground_truth_label"] for row in rows} | {
        row["predicted_label"] for row in rows
    }
    labels = [label for label in base_labels if label in present]
    labels.extend(sorted(present - set(labels)))
    positions = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for row in rows:
        matrix[positions[row["ground_truth_label"]], positions[row["predicted_label"]]] += 1
    return labels, matrix


def _confusion_plot(
    matrix: np.ndarray[Any, Any], labels: list[str], path: Path, title: str
) -> None:
    side = max(9.0, len(labels) * 0.82)
    figure, axis = plt.subplots(figsize=(side, side))
    image = axis.imshow(matrix, cmap="Blues")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#17365d",
                fontsize=8,
            )
    axis.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _per_class_plot(
    internal: list[dict[str, Any]], external: list[dict[str, Any]], path: Path
) -> None:
    labels = [str(row["class"]) for row in internal]
    internal_f1 = [float(row["f1-score"]) for row in internal]
    external_by_class = {str(row["class"]): float(row["f1-score"]) for row in external}
    external_f1 = [external_by_class.get(label, np.nan) for label in labels]
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(13, 6.5))
    axis.bar(positions - width / 2, internal_f1, width, label="Internal test", color="#2563eb")
    axis.bar(
        positions + width / 2,
        external_f1,
        width,
        label="External benchmark",
        color="#f59e0b",
    )
    axis.set_xticks(positions, labels=labels, rotation=42, ha="right")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("F1 score")
    axis.set_title("Per-class performance under domain shift")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _training_plot(path: Path) -> None:
    rows = [json.loads(line) for line in TRAINING_SOURCE.read_text(encoding="utf-8").splitlines()]
    training = [row for row in rows if row.get("loss") is not None]
    steps = np.asarray([row["step"] for row in training], dtype=float)
    losses = np.asarray([row["loss"] for row in training], dtype=float)
    learning_rates = np.asarray(
        [np.nan if row.get("learning_rate") is None else row["learning_rate"] for row in training]
    )
    memory = np.asarray([row["memory"]["max_reserved_gb"] for row in training], dtype=float)
    window = 50
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    smooth_steps = steps[window - 1 :]
    validation: dict[int, float] = {}
    for row in rows:
        if row.get("validation_loss") is not None:
            validation[int(row["step"])] = float(row["validation_loss"])

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(steps, losses, color="#93c5fd", alpha=0.35, linewidth=0.7, label="Step loss")
    axes[0].plot(smooth_steps, smooth, color="#1d4ed8", linewidth=2, label="50-step mean")
    axes[0].scatter(
        list(validation),
        list(validation.values()),
        color="#dc2626",
        zorder=3,
        label="Validation loss",
    )
    axes[0].set_ylabel("Loss")
    axes[0].set_title("TIH Qwen2.5-VL QLoRA training metrics")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(steps, learning_rates, color="#7c3aed", label="Learning rate")
    axes[1].set_ylabel("Learning rate", color="#7c3aed")
    axes[1].tick_params(axis="y", labelcolor="#7c3aed")
    axes[1].set_xlabel("Optimizer step")
    axes[1].grid(alpha=0.2)
    memory_axis = axes[1].twinx()
    memory_axis.plot(steps, memory, color="#059669", alpha=0.8, label="Peak reserved VRAM")
    memory_axis.set_ylabel("Peak reserved VRAM (GB)", color="#059669")
    memory_axis.tick_params(axis="y", labelcolor="#059669")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _benchmark_plot(internal: dict[str, Any], external: dict[str, Any], path: Path) -> None:
    metric_names = ["Accuracy", "Macro F1", "Weighted F1"]
    internal_values = [internal["accuracy"], internal["macro_f1"], internal["weighted_f1"]]
    external_values = [external["accuracy"], external["macro_f1"], external["weighted_f1"]]
    positions = np.arange(len(metric_names))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(positions - width / 2, internal_values, width, label="Internal", color="#2563eb")
    axes[0].bar(positions + width / 2, external_values, width, label="External", color="#f59e0b")
    axes[0].set_xticks(positions, labels=metric_names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification benchmark")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(
        ["Internal", "External"],
        [internal["average_latency_seconds"], external["average_latency_seconds"]],
        color=["#2563eb", "#f59e0b"],
    )
    axes[1].set_ylabel("Seconds per image")
    axes[1].set_title("Average inference latency")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray[Any, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", *labels])
        for label, row in zip(labels, matrix.tolist(), strict=True):
            writer.writerow([label, *row])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
