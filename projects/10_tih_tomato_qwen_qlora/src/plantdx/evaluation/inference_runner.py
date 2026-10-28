"""Stage 1 (inference): base vs. fine-tuned generation over the eval split.

Requires mlx-vlm (lazy-imported, Apple Silicon). Never computes metrics — its
only job is to produce `predictions.jsonl` + `metadata.json`, the frozen
artifact contract that stage 2 (`--stage analyze`, a different, MLX-free
environment) consumes. Deterministic: temperature 0, no sampling, a fixed seed.
Reuses :mod:`plantdx.training.inference` for model loading (no duplicated
load/adapter logic); adds only what that module does not provide — per-call
latency, token counts, and peak memory telemetry via the full mlx-vlm
``GenerationResult``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plantdx.core.exceptions import PlantDxError
from plantdx.evaluation.checkpoint import resolve_adapter_dir
from plantdx.evaluation.config import EvalConfig, resolve_crop
from plantdx.training import seeds
from plantdx.training.data.label_map import load_label_map
from plantdx.training.inference import LoadedModel, load_model
from plantdx.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl

_MAX_NEW_TOKENS = 128
_TEMPERATURE = 0.0  # greedy, deterministic decoding (no sampling), per spec


@dataclass(frozen=True)
class PredictionRow:
    """One eval sample's ground truth plus both models' generations + telemetry."""

    image_id: str
    image_path: str
    disease_id: str
    class_name: str
    instruction: str
    ground_truth: str
    base_prediction: str
    finetuned_prediction: str
    base_runtime_ms: float
    finetuned_runtime_ms: float
    base_prompt_tokens: int
    base_generation_tokens: int
    finetuned_prompt_tokens: int
    finetuned_generation_tokens: int
    base_peak_memory_gb: float
    finetuned_peak_memory_gb: float
    base_confidence: float | None  # mean token probability from logprobs, if exposed
    finetuned_confidence: float | None


def run_inference(cfg: EvalConfig) -> Path:
    """Run stage 1 end to end; write predictions.jsonl + metadata.json.

    Returns the path to the written predictions file. Raises
    :class:`~plantdx.core.exceptions.PlantDxError` (via the lazy mlx-vlm loader)
    if mlx-vlm is not importable in the current environment.
    """
    seeds.apply(cfg.seed)
    rows = _load_split_rows(cfg)
    crop = resolve_crop(cfg.dataset_dir)
    label_map = load_label_map(crop)
    adapter_dir = resolve_adapter_dir(cfg.adapter_path)

    base = load_model(cfg.model_path, adapter_path=None)
    finetuned = load_model(cfg.model_path, adapter_path=str(adapter_dir))

    predictions: list[PredictionRow] = []
    for row in rows:
        predictions.append(_predict_one(row, label_map, base, finetuned))

    out_path = Path(cfg.predictions_path)
    ensure_dir(out_path.parent)
    write_jsonl(out_path, [asdict(p) for p in predictions])
    _write_metadata(cfg, out_path, len(predictions), crop)
    return out_path


def _load_split_rows(cfg: EvalConfig) -> list[dict[str, Any]]:
    path = Path(cfg.dataset_dir) / f"{cfg.split}.jsonl"
    if not path.is_file():
        raise PlantDxError(f"eval split not found: {path}")
    rows = sorted(read_jsonl(path), key=lambda r: str(r["image"]))
    if cfg.max_samples is not None:
        rows = rows[: cfg.max_samples]
    return rows


def _predict_one(
    row: dict[str, Any],
    label_map: dict[str, str],
    base_model: LoadedModel,
    finetuned_model: LoadedModel,
) -> PredictionRow:
    image_path = str(row["image"])
    class_name = Path(image_path).parent.name
    disease_id = label_map.get(class_name, f"unknown:{class_name}")
    instruction = str(row["question"])

    base = _generate(base_model, image_path, instruction)
    ft = _generate(finetuned_model, image_path, instruction)

    return PredictionRow(
        image_id=f"{class_name}/{Path(image_path).name}",
        image_path=image_path,
        disease_id=disease_id,
        class_name=class_name,
        instruction=instruction,
        ground_truth=str(row["answer"]),
        base_prediction=base.text,
        finetuned_prediction=ft.text,
        base_runtime_ms=base.elapsed_ms,
        finetuned_runtime_ms=ft.elapsed_ms,
        base_prompt_tokens=base.prompt_tokens,
        base_generation_tokens=base.generation_tokens,
        finetuned_prompt_tokens=ft.prompt_tokens,
        finetuned_generation_tokens=ft.generation_tokens,
        base_peak_memory_gb=base.peak_memory_gb,
        finetuned_peak_memory_gb=ft.peak_memory_gb,
        base_confidence=base.confidence,
        finetuned_confidence=ft.confidence,
    )


@dataclass(frozen=True)
class _GenerationTelemetry:
    text: str
    elapsed_ms: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_gb: float
    confidence: float | None


def _generate(loaded: LoadedModel, image_path: str, instruction: str) -> _GenerationTelemetry:
    """Run one deterministic generation and capture full mlx-vlm telemetry."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(loaded.processor, loaded.config, instruction, num_images=1)
    start = time.perf_counter()
    result = generate(
        loaded.model,
        loaded.processor,
        prompt,
        image=image_path,
        max_tokens=_MAX_NEW_TOKENS,
        temperature=_TEMPERATURE,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return _GenerationTelemetry(
        text=str(getattr(result, "text", result)).strip(),
        elapsed_ms=elapsed_ms,
        prompt_tokens=int(getattr(result, "prompt_tokens", 0) or 0),
        generation_tokens=int(getattr(result, "generation_tokens", 0) or 0),
        peak_memory_gb=float(getattr(result, "peak_memory", 0.0) or 0.0),
        confidence=_mean_token_confidence(getattr(result, "logprobs", None)),
    )


def _mean_token_confidence(logprobs: Any) -> float | None:
    """Best-effort mean per-token probability from mlx-vlm's ``logprobs`` field.

    ``GenerationResult.logprobs`` shape varies by mlx-vlm version (a single
    float, a per-token list, or absent); this degrades to ``None`` rather than
    guessing at an unsupported shape.
    """
    import math

    try:
        if logprobs is None:
            return None
        if isinstance(logprobs, int | float):
            return math.exp(float(logprobs))
        values = [float(v) for v in logprobs]
        if not values:
            return None
        return math.exp(sum(values) / len(values))
    except (TypeError, ValueError, OverflowError):
        return None


def _write_metadata(cfg: EvalConfig, predictions_path: Path, sample_count: int, crop: str) -> None:
    write_json(
        predictions_path.parent / "metadata.json",
        {
            "stage": "inference",
            "crop": crop,
            "model_path": cfg.model_path,
            "adapter_path": cfg.adapter_path,
            "dataset_dir": cfg.dataset_dir,
            "split": cfg.split,
            "sample_count": sample_count,
            "max_samples": cfg.max_samples,
            "seed": cfg.seed,
            "device": cfg.device,
            "generation": {
                "max_new_tokens": _MAX_NEW_TOKENS,
                "temperature": _TEMPERATURE,
                "sampling": False,
            },
            "predictions_file": predictions_path.name,
        },
    )
