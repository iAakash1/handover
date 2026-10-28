"""Held-out evaluation of a company adapter.

Generates one answer per image, parses it back to a class, and reports the usual
classification metrics plus a confusion matrix. Parsing is deliberately
forgiving about wrapping text and punctuation but never guesses: an answer that
matches no class is counted as unparseable rather than silently reassigned.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from plantdx.company.classes import CompanyDataset
from plantdx.company.config import CompanyTrainConfig
from plantdx.company.dataset import split_path
from plantdx.company.modeling import JsonlRows, generate_label, load_for_inference
from plantdx.core.exceptions import PlantDxError
from plantdx.utils.io import write_json, write_jsonl

UNPARSEABLE = "unparseable"
PREDICTIONS = "predictions.jsonl"
METRICS = "metrics.json"

_PREFIXES = (
    "diagnosis",
    "condition",
    "disease",
    "answer",
    "label",
    "prediction",
    "the condition is",
    "this is",
)


class EvaluationError(PlantDxError):
    """Evaluation could not run."""


@dataclass
class ClassRow:
    """Per-class precision, recall, F1, and support."""

    class_id: str
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvalResult:
    """Everything one evaluation produced."""

    run_name: str
    split: str
    adapter: str | None
    base_model: str
    samples: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    unparseable: int
    seconds: float
    seconds_per_sample: float
    labels: list[str] = field(default_factory=list)
    per_class: list[ClassRow] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    confusions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        payload = asdict(self)
        payload["per_class"] = [asdict(row) for row in self.per_class]
        return payload

    @property
    def best_class(self) -> ClassRow | None:
        """Highest-F1 class with non-zero support."""
        rows = [r for r in self.per_class if r.support]
        return max(rows, key=lambda r: r.f1) if rows else None

    @property
    def worst_class(self) -> ClassRow | None:
        """Lowest-F1 class with non-zero support."""
        rows = [r for r in self.per_class if r.support]
        return min(rows, key=lambda r: r.f1) if rows else None


def normalize_answer(text: str) -> str:
    """Lowercase, strip wrapping prefixes and punctuation, collapse whitespace."""
    cleaned = str(text).strip()
    cleaned = cleaned.split("\n", 1)[0].strip()
    lowered = cleaned.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            remainder = cleaned[len(prefix) :].lstrip()
            if remainder.startswith((":", "-", "=")):
                cleaned = remainder[1:].strip()
                lowered = cleaned.lower()
                break
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def parse_prediction(text: str, dataset: CompanyDataset) -> str:
    """Map generated text onto a class id, or :data:`UNPARSEABLE`.

    Tries an exact normalized match first, then the longest class label that
    appears as a whole-word phrase in the answer. Longest-first matching keeps
    ``healthy leaf`` from being captured by a shorter overlapping label.
    """
    normalized = normalize_answer(text)
    if not normalized:
        return UNPARSEABLE

    by_norm = {normalize_answer(entry.label): entry.class_id for entry in dataset.classes}
    if normalized in by_norm:
        return by_norm[normalized]

    by_id = {entry.class_id.replace("_", " "): entry.class_id for entry in dataset.classes}
    if normalized in by_id:
        return by_id[normalized]

    # "mealybugs" and "whitefly" are the same conditions as "Mealy Bugs" and
    # "White Fly"; no two class labels collide once spaces are removed, so this
    # closes a real gap without letting one class absorb another.
    despaced = {phrase.replace(" ", ""): class_id for phrase, class_id in by_norm.items()}
    if len(despaced) == len(by_norm) and normalized.replace(" ", "") in despaced:
        return despaced[normalized.replace(" ", "")]

    candidates = sorted(
        list(by_norm.items()) + list(by_id.items()), key=lambda kv: len(kv[0]), reverse=True
    )
    padded = f" {normalized} "
    for phrase, class_id in candidates:
        if f" {phrase} " in padded:
            return class_id

    if len(despaced) == len(by_norm):
        collapsed = normalized.replace(" ", "")
        for phrase, class_id in sorted(despaced.items(), key=lambda kv: -len(kv[0])):
            if phrase in collapsed:
                return class_id
    return UNPARSEABLE


def evaluate_predictions(
    truths: list[str], predictions: list[str], dataset: CompanyDataset
) -> tuple[dict[str, float], list[ClassRow], list[list[int]]]:
    """Compute overall metrics, per-class rows, and the confusion matrix."""
    from sklearn.metrics import precision_recall_fscore_support

    class_ids = list(dataset.class_ids)
    correct = sum(1 for t, p in zip(truths, predictions, strict=True) if t == p)
    accuracy = correct / len(truths) if truths else 0.0

    macro = precision_recall_fscore_support(
        truths, predictions, labels=class_ids, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        truths, predictions, labels=class_ids, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        truths, predictions, labels=class_ids, average=None, zero_division=0
    )

    overall = {
        "accuracy": float(accuracy),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]),
        "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]),
    }

    rows: list[ClassRow] = []
    for index, class_id in enumerate(class_ids):
        entry = dataset.by_class_id(class_id)
        rows.append(
            ClassRow(
                class_id=class_id,
                label=entry.label if entry else class_id,
                support=int(per_class[3][index]),
                precision=float(per_class[0][index]),
                recall=float(per_class[1][index]),
                f1=float(per_class[2][index]),
            )
        )

    matrix = confusion_matrix(truths, predictions, class_ids)
    return overall, rows, matrix


def confusion_matrix(
    truths: list[str], predictions: list[str], class_ids: list[str]
) -> list[list[int]]:
    """Rows are true classes, columns predicted; the last column is unparseable."""
    index = {class_id: i for i, class_id in enumerate(class_ids)}
    size = len(class_ids)
    matrix = [[0] * (size + 1) for _ in range(size)]
    for truth, prediction in zip(truths, predictions, strict=True):
        if truth not in index:
            continue
        column = index.get(prediction, size)
        matrix[index[truth]][column] += 1
    return matrix


def top_confusions(
    matrix: list[list[int]], class_ids: list[str], *, limit: int = 10
) -> list[dict[str, Any]]:
    """The most frequent off-diagonal cells, largest first."""
    pairs: list[dict[str, Any]] = []
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            if count == 0 or i == j:
                continue
            predicted = class_ids[j] if j < len(class_ids) else UNPARSEABLE
            pairs.append({"true": class_ids[i], "predicted": predicted, "count": int(count)})
    pairs.sort(key=lambda p: (-int(p["count"]), str(p["true"]), str(p["predicted"])))
    return pairs[:limit]


def run_evaluation(
    cfg: CompanyTrainConfig,
    dataset: CompanyDataset,
    *,
    split: str = "validation",
    adapter_dir: str | Path | None,
    dataset_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    limit: int | None = None,
    local_files_only: bool = False,
    on_progress: Any = None,
    loaded: Any = None,
) -> EvalResult:
    """Generate predictions for one split and write metrics + predictions.

    Args:
        cfg: The run configuration.
        dataset: Resolved company dataset profile (supplies the label vocabulary).
        split: ``train``, ``validation`` or ``test``.
        adapter_dir: Trained adapter, or ``None`` to evaluate the base model.
        dataset_dir: Prepared dataset directory; defaults to ``cfg.dataset_dir``.
        report_dir: Where to write results; defaults to ``cfg.report_dir/<split>``.
        limit: Evaluate only the first N rows.
        local_files_only: Never reach the network for model files.
        on_progress: Called as ``(done, total)`` after each sample.

    Raises:
        EvaluationError: The split is missing or the model could not be loaded.
    """
    if split == "test" and dataset_dir is None:
        import hashlib
        import json

        if adapter_dir is None or not (Path(adapter_dir) / "best_checkpoint.json").is_file():
            raise EvaluationError("The test split is reserved for the selected final adapter")
        if limit is not None or cfg.evaluation.limit is not None:
            raise EvaluationError("Final testing cannot be truncated")
        stamp = {
            "adapter_sha256": hashlib.sha256(
                (Path(adapter_dir) / "adapter_model.safetensors").read_bytes()
            ).hexdigest(),
            "test_sha256": hashlib.sha256(
                split_path(cfg.dataset_dir, "test").read_bytes()
            ).hexdigest(),
        }
        marker = cfg.dataset_dir / "final_test_identity.json"
        if marker.is_file():
            if json.loads(marker.read_text(encoding="utf-8")) != stamp:
                raise EvaluationError(
                    "Test split already used by another adapter; tuning on test is forbidden"
                )
            saved = cfg.report_dir / "test_finetuned" / METRICS
            if saved.is_file():
                return load_result(saved)
        else:
            write_json(marker, stamp)
    data_dir = Path(dataset_dir) if dataset_dir is not None else cfg.dataset_dir
    rows = JsonlRows(split_path(data_dir, split), limit=limit or cfg.evaluation.limit)

    tag = "base" if adapter_dir is None else "finetuned"
    out_dir = Path(report_dir) if report_dir is not None else cfg.report_dir / f"{split}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = loaded or load_for_inference(cfg, adapter_dir, local_files_only=local_files_only)

    truths: list[str] = []
    predictions: list[str] = []
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    # Persist each final test prediction so an interruption never repeats completed samples.
    prediction_file = out_dir / PREDICTIONS
    if split == "test" and prediction_file.is_file():
        import json

        records = [
            json.loads(line)
            for line in prediction_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for index, record in enumerate(records):
            if (
                record["image"] != rows.rows[index]["image"]
                or record["true_class"] != rows.rows[index]["class_id"]
            ):
                raise EvaluationError("Saved test predictions do not match frozen test rows")
            truths.append(record["true_class"])
            predictions.append(record["predicted_class"])

    for position in range(len(records), len(rows.rows)):
        row = rows.rows[position]
        raw = generate_label(
            loaded,
            row["image"],
            str(row["question"]),
            max_side=cfg.model.image_max_side,
            max_new_tokens=cfg.evaluation.max_new_tokens,
        )
        predicted = parse_prediction(raw, dataset)
        truth = str(row["class_id"])
        truths.append(truth)
        predictions.append(predicted)
        records.append(
            {
                "image": row["image"],
                "true_class": truth,
                "predicted_class": predicted,
                "generated": raw,
                "correct": predicted == truth,
            }
        )
        if split == "test":
            import json

            temporary = prediction_file.with_suffix(".tmp")
            write_jsonl(temporary, records)
            temporary.replace(prediction_file)
        if on_progress is not None:
            on_progress(position + 1, len(rows))

    elapsed = time.perf_counter() - started
    overall, per_class, matrix = evaluate_predictions(truths, predictions, dataset)
    unparseable = sum(1 for p in predictions if p == UNPARSEABLE)

    result = EvalResult(
        run_name=cfg.run_name,
        split=split,
        adapter=str(adapter_dir) if adapter_dir else None,
        base_model=cfg.model.model_id,
        samples=len(truths),
        unparseable=unparseable,
        seconds=round(elapsed, 1),
        seconds_per_sample=round(elapsed / len(truths), 3) if truths else 0.0,
        labels=list(dataset.class_ids),
        per_class=per_class,
        confusion_matrix=matrix,
        confusions=top_confusions(matrix, list(dataset.class_ids)),
        **{key: round(value, 4) for key, value in overall.items()},
    )

    write_jsonl(out_dir / PREDICTIONS, records)
    write_json(out_dir / METRICS, result.to_dict())
    write_json(
        out_dir / "confusion_matrix.json",
        {
            "rows": list(dataset.class_ids),
            "columns": [*dataset.class_ids, UNPARSEABLE],
            "matrix": matrix,
        },
    )
    return result


def compare(base: EvalResult | None, tuned: EvalResult | None) -> list[dict[str, Any]]:
    """Side-by-side rows for the headline metrics of two evaluations."""
    keys = ("accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro")
    rows: list[dict[str, Any]] = []
    for key in keys:
        base_value = getattr(base, key) if base else None
        tuned_value = getattr(tuned, key) if tuned else None
        delta = (
            round(tuned_value - base_value, 4)
            if base_value is not None and tuned_value is not None
            else None
        )
        rows.append({"metric": key, "base": base_value, "fine_tuned": tuned_value, "delta": delta})
    return rows


def load_result(path: str | Path) -> EvalResult:
    """Read a metrics.json back into an :class:`EvalResult`.

    Raises:
        EvaluationError: The file is missing or malformed.
    """
    import json

    source = Path(path)
    if source.is_dir():
        source = source / METRICS
    if not source.is_file():
        raise EvaluationError(f"evaluation metrics not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise EvaluationError(f"malformed evaluation metrics: {source} ({exc})") from exc
    rows = [ClassRow(**row) for row in raw.pop("per_class", [])]
    return EvalResult(per_class=rows, **raw)


def label_distribution(predictions: list[str]) -> dict[str, int]:
    """How often each class was predicted."""
    return dict(sorted(Counter(predictions).items()))


def finalize_run(cfg, dataset, adapter_dir):
    """Reload the selected adapter, validate, then evaluate the final test once."""
    import json
    import gc
    import torch
    from plantdx.company.dataset import load_manifest

    loaded = load_for_inference(cfg, adapter_dir, local_files_only=True)
    print("Adapter reload verified. Evaluating validation...")
    validation = run_evaluation(
        cfg, dataset, split="validation", adapter_dir=adapter_dir, loaded=loaded
    )
    print("Evaluating final test...")
    test = run_evaluation(cfg, dataset, split="test", adapter_dir=adapter_dir, loaded=loaded)
    metadata = json.loads((cfg.report_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    warnings = []
    if abs(validation.f1_macro - test.f1_macro) > 0.10:
        warnings.append("Validation/test macro F1 gap exceeds 0.10")
    counts = load_manifest(cfg.dataset_dir)["split"]["per_class"]["train"]
    rare = {k for k, n in counts.items() if n <= sorted(counts.values())[len(counts) // 4]}
    for result in (validation, test):
        for row in result.per_class:
            if row.class_id in rare and row.recall < 0.1:
                warnings.append(
                    f"{result.split}: rare class {row.label} recall is {row.recall:.3f}"
                )
        if result.unparseable:
            warnings.append(f"{result.split}: {result.unparseable} invalid answers")
        majority = max(counts, key=counts.get)
        col = list(dataset.class_ids).index(majority)
        share = sum(row[col] for row in result.confusion_matrix) / result.samples
        if share > counts[majority] / sum(counts.values()) + 0.15:
            warnings.append(f"{result.split}: excessive majority-class predictions ({share:.1%})")
    history = []
    history_file = cfg.log_dir / "loss_history.jsonl"
    if history_file.is_file():
        history = [
            json.loads(line)
            for line in history_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    losses = [r for r in history if r.get("validation_loss") is not None]
    if any(
        r.get("loss") is not None
        and r["loss"] < 0.05
        and r["validation_loss"] > metadata["eval_loss"]
        for r in losses
    ):
        warnings.append("Very low training loss while validation loss worsens")
    report = {
        "best_epoch": metadata["epoch"],
        "training_loss_at_best_epoch": metadata["training_loss"],
        "validation_loss_at_best_epoch": metadata["eval_loss"],
        "validation_macro_f1": validation.f1_macro,
        "test_macro_f1": test.f1_macro,
        "adapter_reload_verified": True,
        "warnings": warnings,
    }
    write_json(cfg.report_dir / "quality_summary.json", report)
    # Reuse a few actual held-out predictions from the final evaluation.
    predictions = (
        (cfg.report_dir / "test_finetuned" / PREDICTIONS).read_text(encoding="utf-8").splitlines()
    )
    write_json(
        cfg.report_dir / "held_out_predictions.json", [json.loads(line) for line in predictions[:5]]
    )
    del loaded
    gc.collect()
    torch.cuda.empty_cache()
    return validation, test
