"""Training configuration: typed, composed, validated (no side effects).

A training run is a pure function of one YAML config (plus the frozen corpus and
the normalized images). The config composes three reusable pieces — a model
(``configs/models/``), a LoRA method (``configs/lora/``), and the run itself
(``configs/train/``) — so a new model or adapter method is one file, never a
code change. Plain dataclasses + an explicit loader (same style as the ontology
and vocabulary policies); no pydantic, no hidden defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plantdx.core.exceptions import ConfigError
from plantdx.utils.io import read_yaml

_LORA_METHODS = frozenset({"lora", "qlora", "dora"})
_SCHEDULERS = frozenset({"cosine", "linear", "constant"})
_RESPONSE_FORMATS = frozenset({"qa", "messages"})


@dataclass(frozen=True)
class ModelConfig:
    """A supported VLM and how mlx-vlm loads it."""

    name: str  # logical key, e.g. "qwen2_5_vl"
    model_path: str  # mlx model id or local path (already downloaded)
    family: str  # mlx-vlm model family (drives chat formatting)
    max_seq_length: int
    quantization_bits: int  # 4 for the *-4bit MLX model
    image_resize: int | None = None  # optional square resize (px)
    assistant_id: int = 77091  # tokenizer assistant-marker id (Qwen default)


@dataclass(frozen=True)
class LoRAConfig:
    """Adapter method + hyperparameters (config-driven, never hardcoded)."""

    method: str  # lora | qlora | dora
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    train_vision: bool = False


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer + schedule + batching."""

    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    scheduler: str  # cosine | linear | constant
    grad_clip: float
    gradient_accumulation_steps: int
    batch_size: int
    epochs: int
    weight_decay: float = 0.0
    train_on_completions: bool = True  # mask the prompt; loss on the caption only


@dataclass(frozen=True)
class DataConfig:
    """How the image x corpus training set is built."""

    crop: str
    classes: tuple[str, ...]
    processed_dir: str  # datasets/<crop>/processed
    corpus_path: str  # artifacts/corpus/captions.jsonl
    captions_per_image: int
    max_captions_per_disease: int
    response_format: str  # qa | messages
    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_seed: int
    image_glob: str = "*.JPG"
    # Optional override for the instruction-paraphrase bank (default: the
    # tomato-worded assets/training/instructions.json). Crops whose leaf noun
    # differs from "tomato" must set this to a matching asset — the instruction
    # text is literal model input, not just a label.
    instructions_path: str | None = None


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint policy (best / latest / per-epoch, resume)."""

    output_dir: str  # checkpoints/<run_name>
    steps_per_save: int
    keep_last: int
    keep_best: bool
    resume: bool


@dataclass(frozen=True)
class LoggingConfig:
    """Where run logs, curves, and reports go."""

    log_dir: str  # logs/<run_name>
    report_dir: str  # reports/<run_name>
    steps_per_report: int
    steps_per_eval: int
    val_batches: int
    tensorboard: bool


@dataclass(frozen=True)
class TrainingConfig:
    """The complete, composed training configuration for one run."""

    run_name: str
    backend: str  # "mlx_vlm"
    seed: int
    grad_checkpoint: bool
    model: ModelConfig
    lora: LoRAConfig
    optim: OptimConfig
    data: DataConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    provenance: dict[str, str] = field(default_factory=dict)


def load_training_config(path: str | Path, *, base_dir: str | Path | None = None) -> TrainingConfig:
    """Load and validate a composed training config from ``path``.

    The train YAML may reference a model file (``model: qwen25vl`` ->
    ``configs/models/qwen25vl.yaml``) and a LoRA file (``lora: qlora`` ->
    ``configs/lora/qlora.yaml``); inline blocks override the referenced file.
    """
    root = Path(base_dir) if base_dir else Path.cwd()
    train_path = Path(path)
    raw = _read(train_path)

    model = _resolve("model", raw, root / "configs" / "models")
    lora = _resolve("lora", raw, root / "configs" / "lora")

    try:
        cfg = TrainingConfig(
            run_name=str(raw["run_name"]),
            backend=str(raw.get("backend", "mlx_vlm")),
            seed=int(raw.get("seed", 20260711)),
            grad_checkpoint=bool(raw.get("grad_checkpoint", True)),
            model=_model(model),
            lora=_lora(lora),
            optim=_optim(raw["optim"]),
            data=_data(raw["data"]),
            checkpoint=_checkpoint(raw["checkpoint"], raw["run_name"]),
            logging=_logging(raw["logging"], raw["run_name"]),
            provenance={"config_path": str(train_path)},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid training config {train_path}: {exc}") from exc
    _validate(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"training config not found: {path}")
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ConfigError(f"training config must be a mapping: {path}")
    return data


def _resolve(key: str, raw: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Resolve a ``key: <name>`` reference to ``<directory>/<name>.yaml``.

    An inline ``key`` mapping is used directly; a ``key_override`` mapping is
    merged on top of the referenced file.
    """
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        ref = directory / f"{value}.yaml"
        base = _read(ref)
        override = raw.get(f"{key}_override", {})
        return {**base, **(override if isinstance(override, dict) else {})}
    raise ConfigError(f"config '{key}' must be a name or an inline mapping")


def _model(d: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=str(d["name"]),
        model_path=str(d["model_path"]),
        family=str(d["family"]),
        max_seq_length=int(d["max_seq_length"]),
        quantization_bits=int(d["quantization_bits"]),
        image_resize=int(d["image_resize"]) if d.get("image_resize") is not None else None,
        assistant_id=int(d.get("assistant_id", 77091)),
    )


def _lora(d: dict[str, Any]) -> LoRAConfig:
    return LoRAConfig(
        method=str(d["method"]),
        rank=int(d["rank"]),
        alpha=int(d["alpha"]),
        dropout=float(d["dropout"]),
        target_modules=tuple(d["target_modules"]),
        train_vision=bool(d.get("train_vision", False)),
    )


def _optim(d: dict[str, Any]) -> OptimConfig:
    return OptimConfig(
        learning_rate=float(d["learning_rate"]),
        min_learning_rate=float(d.get("min_learning_rate", float(d["learning_rate"]) / 100)),
        warmup_steps=int(d["warmup_steps"]),
        scheduler=str(d.get("scheduler", "cosine")),
        grad_clip=float(d.get("grad_clip", 1.0)),
        gradient_accumulation_steps=int(d["gradient_accumulation_steps"]),
        batch_size=int(d["batch_size"]),
        epochs=int(d["epochs"]),
        weight_decay=float(d.get("weight_decay", 0.0)),
        train_on_completions=bool(d.get("train_on_completions", True)),
    )


def _data(d: dict[str, Any]) -> DataConfig:
    return DataConfig(
        crop=str(d["crop"]),
        classes=tuple(d["classes"]),
        processed_dir=str(d["processed_dir"]),
        corpus_path=str(d["corpus_path"]),
        captions_per_image=int(d["captions_per_image"]),
        max_captions_per_disease=int(d["max_captions_per_disease"]),
        response_format=str(d.get("response_format", "qa")),
        train_ratio=float(d["train_ratio"]),
        val_ratio=float(d["val_ratio"]),
        test_ratio=float(d["test_ratio"]),
        split_seed=int(d["split_seed"]),
        image_glob=str(d.get("image_glob", "*.JPG")),
        instructions_path=str(d["instructions_path"]) if d.get("instructions_path") else None,
    )


def _checkpoint(d: dict[str, Any], run: str) -> CheckpointConfig:
    return CheckpointConfig(
        output_dir=str(d.get("output_dir", f"checkpoints/{run}")),
        steps_per_save=int(d["steps_per_save"]),
        keep_last=int(d.get("keep_last", 3)),
        keep_best=bool(d.get("keep_best", True)),
        resume=bool(d.get("resume", False)),
    )


def _logging(d: dict[str, Any], run: str) -> LoggingConfig:
    return LoggingConfig(
        log_dir=str(d.get("log_dir", f"logs/{run}")),
        report_dir=str(d.get("report_dir", f"reports/{run}")),
        steps_per_report=int(d["steps_per_report"]),
        steps_per_eval=int(d["steps_per_eval"]),
        val_batches=int(d.get("val_batches", 4)),
        tensorboard=bool(d.get("tensorboard", False)),
    )


def _validate(cfg: TrainingConfig) -> None:
    """Fail closed on out-of-range or inconsistent values."""
    errors: list[str] = []
    if cfg.backend != "mlx_vlm":
        errors.append(f"unsupported backend {cfg.backend!r} (only 'mlx_vlm')")
    if cfg.lora.method not in _LORA_METHODS:
        errors.append(f"lora.method must be one of {sorted(_LORA_METHODS)}")
    if cfg.optim.scheduler not in _SCHEDULERS:
        errors.append(f"optim.scheduler must be one of {sorted(_SCHEDULERS)}")
    if cfg.data.response_format not in _RESPONSE_FORMATS:
        errors.append(f"data.response_format must be one of {sorted(_RESPONSE_FORMATS)}")
    ratio_sum = cfg.data.train_ratio + cfg.data.val_ratio + cfg.data.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        errors.append(f"data split ratios must sum to 1.0 (got {ratio_sum})")
    for name, value in (
        ("lora.rank", cfg.lora.rank),
        ("optim.batch_size", cfg.optim.batch_size),
        ("optim.epochs", cfg.optim.epochs),
        ("optim.gradient_accumulation_steps", cfg.optim.gradient_accumulation_steps),
        ("data.captions_per_image", cfg.data.captions_per_image),
    ):
        if value <= 0:
            errors.append(f"{name} must be > 0 (got {value})")
    if cfg.lora.method == "qlora" and cfg.model.quantization_bits != 4:
        errors.append("qlora requires a 4-bit quantized model (model.quantization_bits: 4)")
    if errors:
        raise ConfigError("training config invalid:\n  " + "\n  ".join(errors))
