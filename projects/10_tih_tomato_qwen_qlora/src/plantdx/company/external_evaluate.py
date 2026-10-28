"""Evaluate a trained company adapter on a folder-labelled external dataset."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plantdx.company.classes import ClassEntry, CompanyDataset
from plantdx.company.config import CompanyTrainConfig
from plantdx.company.evaluate import UNPARSEABLE
from plantdx.company.modeling import ModelingError
from plantdx.company.predict import Prediction, load_predictor, predict_image

# The supplied external benchmark uses ``Healthy`` for images of healthy leaves.
# Keep this benchmark-specific alias here so the training class profile remains immutable.
EXTERNAL_FOLDER_ALIASES = {"healthy": "healthy_leaf"}


@dataclass(frozen=True)
class ExternalImage:
    """One external image and its ground-truth class from the parent folder."""

    path: Path
    class_id: str
    label: str
    folder: str


@dataclass(frozen=True)
class ExternalPrediction:
    """One serializable external benchmark prediction."""

    image: str
    ground_truth_label: str
    ground_truth_class_id: str
    predicted_label: str
    predicted_class_id: str
    confidence: float | None
    latency_seconds: float
    generated: str


@dataclass(frozen=True)
class ExternalBenchmarkResult:
    """Summary and artifact paths for one external benchmark run."""

    dataset_root: Path
    model: str
    adapter: str
    output_dir: Path
    total_images: int
    number_of_classes: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    average_latency_seconds: float
    ignored_folders: tuple[str, ...]


def resolve_external_root(value: str | Path, base_dir: Path) -> Path:
    """Resolve a benchmark root from the CWD, repository, or repository parent."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        choices = [candidate]
    else:
        choices = [Path.cwd() / candidate, base_dir / candidate, base_dir.parent / candidate]
    seen: set[Path] = set()
    for choice in choices:
        resolved = choice.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            return resolved
    tried = "; ".join(str(path.resolve()) for path in choices)
    raise ModelingError(f"external dataset root does not exist; tried: {tried}")


def discover_external_images(
    root: Path,
    dataset: CompanyDataset,
    *,
    limit: int | None = None,
) -> tuple[list[ExternalImage], dict[str, str], list[str]]:
    """Map top-level class folders by name and return a deterministic image list."""
    if limit is not None and limit <= 0:
        raise ModelingError("external evaluation limit must be positive")

    lookup: dict[str, ClassEntry] = {}
    for entry in dataset.classes:
        for name in (entry.raw, entry.class_id, entry.label):
            key = _normalize(name)
            previous = lookup.get(key)
            if previous is not None and previous.class_id != entry.class_id:
                raise ModelingError(f"ambiguous normalized class name {name!r}")
            lookup[key] = entry

    allowed = set(dataset.extensions)
    per_class: dict[str, list[ExternalImage]] = {}
    folder_mapping: dict[str, str] = {}
    ignored: list[str] = []
    for folder in sorted((item for item in root.iterdir() if item.is_dir()), key=_path_key):
        key = _normalize(folder.name)
        entry = lookup.get(key)
        if entry is None:
            alias_class_id = EXTERNAL_FOLDER_ALIASES.get(key)
            entry = dataset.by_class_id(alias_class_id) if alias_class_id else None
        if entry is None:
            ignored.append(folder.name)
            continue
        folder_mapping[folder.name] = entry.class_id
        images = [
            ExternalImage(
                path=image.resolve(),
                class_id=entry.class_id,
                label=entry.label,
                folder=folder.name,
            )
            for image in sorted(
                (
                    item
                    for item in folder.rglob("*")
                    if item.is_file() and item.suffix.lower() in allowed
                ),
                key=_path_key,
            )
        ]
        per_class.setdefault(entry.class_id, []).extend(images)

    rows = _round_robin(per_class, dataset, limit=limit)
    if not rows:
        raise ModelingError(f"no mapped images found under external dataset root: {root}")
    return rows, folder_mapping, ignored


def run_external_benchmark(
    cfg: CompanyTrainConfig,
    dataset: CompanyDataset,
    dataset_root: str | Path,
    *,
    adapter_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> ExternalBenchmarkResult:
    """Run inference-only evaluation and write all benchmark artifacts."""
    import torch

    root = resolve_external_root(dataset_root, cfg.base_dir)
    rows, folder_mapping, ignored = discover_external_images(root, dataset, limit=limit)
    adapter = Path(adapter_dir).resolve() if adapter_dir else cfg.checkpoint_dir.resolve()
    report_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else cfg.base_dir / "reports" / f"{cfg.run_name}_external"
    )
    if not report_dir.is_absolute():
        report_dir = cfg.base_dir / report_dir
    report_dir = report_dir.resolve()

    loaded = load_predictor(cfg, adapter_dir=adapter)
    loaded.model.requires_grad_(False)
    loaded.model.eval()
    if loaded.model.training:
        raise ModelingError("external evaluation requires model.eval()")

    predictions: list[ExternalPrediction] = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            prediction = predict_image(loaded, row.path, cfg=cfg, dataset=dataset)
            predictions.append(_prediction_row(row, prediction))
            if on_progress is not None:
                on_progress(index, len(rows))

    return write_external_report(
        predictions,
        dataset=dataset,
        dataset_root=root,
        model=loaded.model_id,
        adapter=loaded.adapter_path or str(adapter),
        output_dir=report_dir,
        folder_mapping=folder_mapping,
        ignored_folders=ignored,
    )


def write_external_report(
    predictions: list[ExternalPrediction],
    *,
    dataset: CompanyDataset,
    dataset_root: Path,
    model: str,
    adapter: str,
    output_dir: Path,
    folder_mapping: dict[str, str],
    ignored_folders: list[str],
) -> ExternalBenchmarkResult:
    """Compute metrics and write JSON, CSV, and confusion-matrix artifacts."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if not predictions:
        raise ModelingError("cannot report an empty external benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = [row.ground_truth_class_id for row in predictions]
    predicted = [row.predicted_class_id for row in predictions]
    detected_ids = [entry.class_id for entry in dataset.classes if entry.class_id in set(truth)]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=detected_ids, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        truth, predicted, labels=detected_ids, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        truth, predicted, labels=detected_ids, average="weighted", zero_division=0
    )
    accuracy = float(accuracy_score(truth, predicted))
    average_latency = sum(row.latency_seconds for row in predictions) / len(predictions)

    per_class = []
    for index, class_id in enumerate(detected_ids):
        entry = dataset.by_class_id(class_id)
        per_class.append(
            {
                "class": entry.label if entry else class_id,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1-score": float(f1[index]),
            }
        )

    result = ExternalBenchmarkResult(
        dataset_root=dataset_root,
        model=model,
        adapter=adapter,
        output_dir=output_dir,
        total_images=len(predictions),
        number_of_classes=len(detected_ids),
        accuracy=accuracy,
        macro_precision=float(macro[0]),
        macro_recall=float(macro[1]),
        macro_f1=float(macro[2]),
        weighted_f1=float(weighted[2]),
        average_latency_seconds=average_latency,
        ignored_folders=tuple(ignored_folders),
    )

    metrics: dict[str, Any] = {
        "total_images": result.total_images,
        "number_of_classes": result.number_of_classes,
        "accuracy": result.accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "average_latency": result.average_latency_seconds,
        "average_latency_seconds": result.average_latency_seconds,
        "dataset": str(dataset_root),
        "model": model,
        "adapter": adapter,
        "detected_classes": detected_ids,
        "folder_mapping": folder_mapping,
        "ignored_folders": ignored_folders,
        "unparseable_predictions": sum(
            row.predicted_class_id == UNPARSEABLE for row in predictions
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "classification_report.csv", per_class)
    _write_csv(
        output_dir / "predictions.csv",
        [asdict(row) for row in predictions],
    )
    _write_csv(
        output_dir / "misclassified_examples.csv",
        [
            {
                "image": row.image,
                "true_label": row.ground_truth_label,
                "predicted_label": row.predicted_label,
            }
            for row in predictions
            if row.ground_truth_class_id != row.predicted_class_id
        ],
        fieldnames=["image", "true_label", "predicted_label"],
    )
    _write_confusion_matrix(
        output_dir / "confusion_matrix.png", truth, predicted, dataset=dataset
    )
    return result


def _prediction_row(row: ExternalImage, prediction: Prediction) -> ExternalPrediction:
    return ExternalPrediction(
        image=str(row.path),
        ground_truth_label=row.label,
        ground_truth_class_id=row.class_id,
        predicted_label=prediction.label,
        predicted_class_id=prediction.class_id,
        confidence=prediction.confidence,
        latency_seconds=prediction.seconds,
        generated=prediction.generated,
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix(
    path: Path,
    truth: list[str],
    predicted: list[str],
    *,
    dataset: CompanyDataset,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    present = set(truth) | set(predicted)
    class_ids = [entry.class_id for entry in dataset.classes if entry.class_id in present]
    if UNPARSEABLE in present:
        class_ids.append(UNPARSEABLE)
    labels = []
    for class_id in class_ids:
        entry = dataset.by_class_id(class_id)
        labels.append(entry.label if entry else "Unrecognized")
    matrix = confusion_matrix(truth, predicted, labels=class_ids)
    side = max(9.0, 0.85 * len(class_ids))
    figure, axis = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False, values_format="d"
    )
    axis.set_title("External benchmark confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _round_robin(
    per_class: dict[str, list[ExternalImage]],
    dataset: CompanyDataset,
    *,
    limit: int | None,
) -> list[ExternalImage]:
    ordered = [per_class[class_id] for class_id in dataset.class_ids if per_class.get(class_id)]
    rows: list[ExternalImage] = []
    depth = 0
    while ordered:
        next_ordered: list[list[ExternalImage]] = []
        for images in ordered:
            if depth < len(images):
                rows.append(images[depth])
                if limit is not None and len(rows) >= limit:
                    return rows
            if depth + 1 < len(images):
                next_ordered.append(images)
        ordered = next_ordered
        depth += 1
    return rows


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _path_key(path: Path) -> str:
    return path.as_posix().casefold()
