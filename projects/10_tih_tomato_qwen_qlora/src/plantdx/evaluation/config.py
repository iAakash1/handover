"""Evaluation run configuration: resolved from CLI flags, validated fail-closed.

An evaluation run is driven entirely by `plantdx evaluate` flags (no YAML layer
like training's `configs/train/` — the eval surface is small enough that flags
with sensible frozen-run defaults are simpler and just as reproducible, since
every resolved value is recorded into the reproducibility manifest anyway).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plantdx.core.exceptions import ConfigError
from plantdx.evaluation.checkpoint import run_name_from_adapter_path

_STAGES = frozenset({"inference", "analyze", "all"})
_SPLITS = frozenset({"train", "validation", "test"})
_DEVICES = frozenset({"auto", "cpu", "gpu"})

_DEFAULT_MODEL_PATH = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
# The checkpoint DIRECTORY (mlx-vlm's apply_lora_layers expects a directory
# containing adapter_config.json + adapters.safetensors side by side, not the
# weights file alone) -- see evaluation/checkpoint.py.
#
# These two are the only genuinely hardcoded defaults left: `plantdx evaluate`
# with no flags at all must still resolve to *some* run, and the frozen tomato
# run is that documented default. Everything downstream (output dir, crop,
# ontology/vocabulary/lexicon) is derived from whichever adapter/dataset is
# actually configured -- never re-hardcoded per crop.
_DEFAULT_ADAPTER = "checkpoints/qwen25vl_tomato_qlora"
_DEFAULT_DATASET_DIR = "artifacts/training/qwen25vl_tomato_qlora/dataset"


@dataclass(frozen=True)
class EvalConfig:
    """Fully-resolved settings for one `plantdx evaluate` invocation."""

    stage: str  # inference | analyze | all
    model_path: str  # base model repo id/path (never modified, never retrained)
    adapter_path: str  # trained LoRA adapter checkpoint DIRECTORY (frozen artifact)
    dataset_dir: str  # dir containing the frozen train/validation/test.jsonl
    split: str  # which split to evaluate (default: test)
    output_dir: str  # reports/<run>/evaluation/
    batch_size: int
    max_samples: int | None  # None = evaluate every sample in the split
    seed: int
    device: str  # auto | cpu | gpu (mlx picks its own device; recorded for the manifest)
    predictions_path: str  # the stage-1 -> stage-2 artifact contract


def resolve_eval_config(
    *,
    stage: str = "all",
    model: str | None = None,
    adapter: str | None = None,
    dataset: str | None = None,
    split: str = "test",
    output_dir: str | None = None,
    batch_size: int = 1,
    max_samples: int | None = None,
    seed: int = 20260711,
    device: str = "auto",
    predictions_path: str | None = None,
) -> EvalConfig:
    """Resolve and validate an :class:`EvalConfig` from CLI-flag values.

    Every parameter has a default matching the frozen tomato/Qwen2.5-VL run, so
    ``plantdx evaluate`` with no flags evaluates that exact run's test split.
    """
    errors: list[str] = []
    if stage not in _STAGES:
        errors.append(f"--stage must be one of {sorted(_STAGES)} (got {stage!r})")
    if split not in _SPLITS:
        errors.append(f"--split must be one of {sorted(_SPLITS)} (got {split!r})")
    if device not in _DEVICES:
        errors.append(f"--device must be one of {sorted(_DEVICES)} (got {device!r})")
    if batch_size <= 0:
        errors.append(f"--batch-size must be > 0 (got {batch_size})")
    if max_samples is not None and max_samples <= 0:
        errors.append(f"--max-samples must be > 0 (got {max_samples})")
    if errors:
        raise ConfigError("evaluation config invalid:\n  " + "\n  ".join(errors))

    resolved_adapter = adapter or _DEFAULT_ADAPTER
    out_dir = output_dir or f"reports/{run_name_from_adapter_path(resolved_adapter)}/evaluation"
    return EvalConfig(
        stage=stage,
        model_path=model or _DEFAULT_MODEL_PATH,
        adapter_path=resolved_adapter,
        dataset_dir=dataset or _DEFAULT_DATASET_DIR,
        split=split,
        output_dir=out_dir,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
        device=device,
        predictions_path=predictions_path or str(Path(out_dir) / "raw" / "predictions.jsonl"),
    )


def resolve_crop(dataset_dir: str | Path) -> str:
    """The crop ``dataset_dir`` was built for, read from its frozen manifest.

    The training data pipeline (``plantdx train`` / ``prepare-training``)
    always writes ``crop`` into ``<dataset_dir>/manifest.json`` (see
    ``training/data/builder.py::_write_manifest``) -- this is the single
    source of truth for which crop's ontology/vocabulary/lexicon an evaluation
    run must use. Never guessed, never hardcoded, never a separate flag that
    could silently drift from the dataset actually being evaluated.
    """
    manifest_path = Path(dataset_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ConfigError(
            f"dataset manifest not found: {manifest_path}. The training data "
            f"pipeline (`plantdx train` / `plantdx prepare-training`) must be run "
            f"first; evaluation reads the crop from its manifest, never guesses it."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    crop = data.get("crop")
    if not crop:
        raise ConfigError(f"dataset manifest at {manifest_path} has no 'crop' field")
    return str(crop)
