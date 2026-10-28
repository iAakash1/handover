"""Training configuration for the company tomato run.

Plain frozen dataclasses loaded from one YAML file. Separate from
``plantdx.training.config`` (the MLX research path) so neither constrains the
other. Validation is fail-closed: a bad value raises before anything is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from plantdx.company.classes import repo_root
from plantdx.core.exceptions import ConfigError
from plantdx.utils.io import read_yaml

DEFAULT_CONFIG = "configs/train/qwen25vl_tih_tomato_strong.yaml"

_SCHEDULERS = frozenset({"cosine", "linear", "constant"})
_DTYPES = frozenset({"bfloat16", "float16"})
_QUANT_TYPES = frozenset({"nf4", "fp4"})


@dataclass(frozen=True)
class ModelSpec:
    """Base model and how it is quantized and fed images."""

    model_id: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    compute_dtype: str
    image_max_side: int
    max_seq_length: int


@dataclass(frozen=True)
class LoRASpec:
    """QLoRA adapter capacity and placement."""

    rank: int
    alpha: int
    dropout: float
    train_vision: bool
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class OptimSpec:
    """Optimizer, schedule, and batching."""

    learning_rate: float
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    warmup_ratio: float
    scheduler: str
    grad_clip: float
    grad_checkpoint: bool
    optimizer: str

    @property
    def effective_batch_size(self) -> int:
        """Micro-batch times accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class DataSpec:
    """Dataset profile, prompt, and split policy."""

    profile: str
    instruction: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    split_seed: int
    deduplicate: bool


@dataclass(frozen=True)
class EvalSpec:
    """Held-out evaluation settings."""

    max_new_tokens: int
    limit: int | None


@dataclass(frozen=True)
class OutputSpec:
    """Where the run writes its artifacts (all repo-root relative or absolute)."""

    dataset_dir: str
    checkpoint_dir: str
    log_dir: str
    report_dir: str


@dataclass(frozen=True)
class CompanyTrainConfig:
    """One company training run, fully specified."""

    run_name: str
    seed: int
    model: ModelSpec
    lora: LoRASpec
    optim: OptimSpec
    data: DataSpec
    evaluation: EvalSpec
    output: OutputSpec
    base_dir: Path
    config_path: Path | None = None
    provenance: dict[str, str] = field(default_factory=dict)

    def path(self, value: str) -> Path:
        """Resolve a configured output path against the repository root."""
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else self.base_dir / candidate

    @property
    def dataset_dir(self) -> Path:
        """Directory holding train/validation/test JSONL plus the manifest."""
        return self.path(self.output.dataset_dir)

    @property
    def checkpoint_dir(self) -> Path:
        """Directory holding the trained adapter and its checkpoints."""
        return self.path(self.output.checkpoint_dir)

    @property
    def log_dir(self) -> Path:
        """Directory holding train.log and the loss history."""
        return self.path(self.output.log_dir)

    @property
    def report_dir(self) -> Path:
        """Directory holding evaluation reports and the run manifest."""
        return self.path(self.output.report_dir)


def load_company_train_config(
    path: str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> CompanyTrainConfig:
    """Load and validate the company training config.

    Args:
        path: Config YAML; defaults to ``configs/train/qwen25vl_tih_tomato.yaml``.
        base_dir: Repository root for resolving relative paths. Defaults to the
            repository this package lives in.
        overrides: Flat keys applied on top of the parsed config. Recognized:
            ``run_name``, ``seed``, ``epochs``, ``learning_rate``, ``lora_rank``,
            ``lora_alpha``, ``lora_dropout``, ``gradient_accumulation_steps``,
            ``batch_size``, ``image_max_side``, ``grad_checkpoint``,
            ``eval_limit``, ``max_new_tokens``.

    Raises:
        ConfigError: The file is missing or any value is invalid.
    """
    base = Path(base_dir).resolve() if base_dir is not None else repo_root()
    cfg_path = Path(path) if path is not None else base / DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = base / cfg_path
    if not cfg_path.is_file():
        raise ConfigError(f"company training config not found: {cfg_path}")

    raw = read_yaml(cfg_path)
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: expected a mapping at the top level")

    cfg = _build(raw, base, cfg_path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    _validate(cfg)
    return cfg


def apply_overrides(cfg: CompanyTrainConfig, overrides: dict[str, Any]) -> CompanyTrainConfig:
    """Return a copy of ``cfg`` with the recognized override keys applied."""
    known = {
        "run_name",
        "seed",
        "epochs",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "gradient_accumulation_steps",
        "batch_size",
        "image_max_side",
        "grad_checkpoint",
        "eval_limit",
        "max_new_tokens",
    }
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ConfigError(f"unknown override(s): {', '.join(unknown)}")

    model = cfg.model
    lora = cfg.lora
    optim = cfg.optim
    evaluation = cfg.evaluation
    run_name = cfg.run_name
    seed = cfg.seed
    output = cfg.output

    if "image_max_side" in overrides:
        model = replace(model, image_max_side=int(overrides["image_max_side"]))
    if "lora_rank" in overrides:
        lora = replace(lora, rank=int(overrides["lora_rank"]))
    if "lora_alpha" in overrides:
        lora = replace(lora, alpha=int(overrides["lora_alpha"]))
    if "lora_dropout" in overrides:
        lora = replace(lora, dropout=float(overrides["lora_dropout"]))
    if "epochs" in overrides:
        optim = replace(optim, epochs=float(overrides["epochs"]))
    if "learning_rate" in overrides:
        optim = replace(optim, learning_rate=float(overrides["learning_rate"]))
    if "gradient_accumulation_steps" in overrides:
        optim = replace(
            optim, gradient_accumulation_steps=int(overrides["gradient_accumulation_steps"])
        )
    if "batch_size" in overrides:
        optim = replace(optim, batch_size=int(overrides["batch_size"]))
    if "grad_checkpoint" in overrides:
        optim = replace(optim, grad_checkpoint=bool(overrides["grad_checkpoint"]))
    if "eval_limit" in overrides:
        limit = overrides["eval_limit"]
        evaluation = replace(evaluation, limit=None if limit in (None, 0) else int(limit))
    if "max_new_tokens" in overrides:
        evaluation = replace(evaluation, max_new_tokens=int(overrides["max_new_tokens"]))
    if "seed" in overrides:
        seed = int(overrides["seed"])
    if "run_name" in overrides:
        new_name = str(overrides["run_name"]).strip()
        if not new_name:
            raise ConfigError("run_name override must not be empty")
        output = _retarget(output, cfg.run_name, new_name)
        run_name = new_name

    return replace(
        cfg,
        run_name=run_name,
        seed=seed,
        model=model,
        lora=lora,
        optim=optim,
        evaluation=evaluation,
        output=output,
    )


def _retarget(output: OutputSpec, old: str, new: str) -> OutputSpec:
    """Re-point default output paths that embed the old run name."""

    def swap(value: str) -> str:
        return value.replace(old, new) if old and old in value else value

    return OutputSpec(
        dataset_dir=swap(output.dataset_dir),
        checkpoint_dir=swap(output.checkpoint_dir),
        log_dir=swap(output.log_dir),
        report_dir=swap(output.report_dir),
    )


def _build(raw: dict[str, Any], base: Path, cfg_path: Path) -> CompanyTrainConfig:
    try:
        run_name = str(raw["run_name"]).strip()
        model_raw = _section(raw, "model")
        lora_raw = _section(raw, "lora")
        optim_raw = _section(raw, "optim")
        data_raw = _section(raw, "data")
        eval_raw = raw.get("evaluation") or {}
        output_raw = raw.get("output") or {}

        model = ModelSpec(
            model_id=str(model_raw["model_id"]),
            load_in_4bit=bool(model_raw.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(model_raw.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_raw.get("bnb_4bit_use_double_quant", True)),
            compute_dtype=str(model_raw.get("compute_dtype", "bfloat16")),
            image_max_side=int(model_raw.get("image_max_side", 448)),
            max_seq_length=int(model_raw.get("max_seq_length", 1024)),
        )
        lora = LoRASpec(
            rank=int(lora_raw["rank"]),
            alpha=int(lora_raw["alpha"]),
            dropout=float(lora_raw.get("dropout", 0.05)),
            train_vision=bool(lora_raw.get("train_vision", False)),
            target_modules=tuple(str(m) for m in lora_raw.get("target_modules", ())),
        )
        optim = OptimSpec(
            learning_rate=float(optim_raw["learning_rate"]),
            epochs=float(optim_raw["epochs"]),
            batch_size=int(optim_raw.get("batch_size", 1)),
            gradient_accumulation_steps=int(optim_raw.get("gradient_accumulation_steps", 8)),
            weight_decay=float(optim_raw.get("weight_decay", 0.0)),
            warmup_ratio=float(optim_raw.get("warmup_ratio", 0.03)),
            scheduler=str(optim_raw.get("scheduler", "cosine")),
            grad_clip=float(optim_raw.get("grad_clip", 1.0)),
            grad_checkpoint=bool(optim_raw.get("grad_checkpoint", True)),
            optimizer=str(optim_raw.get("optimizer", "paged_adamw_8bit")),
        )
        data = DataSpec(
            profile=str(data_raw.get("profile", "configs/company_tomato_classes.yaml")),
            instruction=str(data_raw["instruction"]),
            train_ratio=float(data_raw.get("train_ratio", 0.8)),
            val_ratio=float(data_raw.get("val_ratio", 0.1)),
            test_ratio=float(data_raw.get("test_ratio", 0.1)),
            split_seed=int(data_raw.get("split_seed", 20260911)),
            deduplicate=bool(data_raw.get("deduplicate", True)),
        )
        limit_raw = eval_raw.get("limit")
        evaluation = EvalSpec(
            max_new_tokens=int(eval_raw.get("max_new_tokens", 24)),
            limit=None if limit_raw in (None, 0) else int(limit_raw),
        )
        output = OutputSpec(
            dataset_dir=str(output_raw.get("dataset_dir", f"artifacts/company/{run_name}/dataset")),
            checkpoint_dir=str(output_raw.get("checkpoint_dir", f"checkpoints/{run_name}")),
            log_dir=str(output_raw.get("log_dir", f"logs/{run_name}")),
            report_dir=str(output_raw.get("report_dir", f"reports/{run_name}")),
        )
    except KeyError as exc:
        raise ConfigError(f"{cfg_path}: missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{cfg_path}: {exc}") from exc

    return CompanyTrainConfig(
        run_name=run_name,
        seed=int(raw.get("seed", 20260911)),
        model=model,
        lora=lora,
        optim=optim,
        data=data,
        evaluation=evaluation,
        output=output,
        base_dir=base,
        config_path=cfg_path,
        provenance={"config_path": str(cfg_path)},
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or malformed `{key}:` section")
    return value


def _validate(cfg: CompanyTrainConfig) -> None:
    errors: list[str] = []
    if not cfg.run_name:
        errors.append("run_name must not be empty")
    if cfg.model.compute_dtype not in _DTYPES:
        errors.append(f"compute_dtype must be one of {sorted(_DTYPES)}")
    if cfg.model.bnb_4bit_quant_type not in _QUANT_TYPES:
        errors.append(f"bnb_4bit_quant_type must be one of {sorted(_QUANT_TYPES)}")
    if cfg.model.image_max_side < 112 or cfg.model.image_max_side % 28:
        errors.append("image_max_side must be >= 112 and a multiple of 28 (Qwen patch size)")
    if cfg.model.max_seq_length < 64:
        errors.append("max_seq_length must be >= 64")
    if cfg.lora.rank <= 0 or cfg.lora.alpha <= 0:
        errors.append("lora rank and alpha must be positive")
    if not 0.0 <= cfg.lora.dropout < 1.0:
        errors.append("lora dropout must be in [0, 1)")
    if not cfg.lora.target_modules:
        errors.append("lora target_modules must not be empty")
    if cfg.optim.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    if cfg.optim.epochs <= 0:
        errors.append("epochs must be positive")
    if cfg.optim.batch_size <= 0 or cfg.optim.gradient_accumulation_steps <= 0:
        errors.append("batch_size and gradient_accumulation_steps must be positive")
    if cfg.optim.scheduler not in _SCHEDULERS:
        errors.append(f"scheduler must be one of {sorted(_SCHEDULERS)}")
    if not 0.0 <= cfg.optim.warmup_ratio < 1.0:
        errors.append("warmup_ratio must be in [0, 1)")
    ratios = (cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.test_ratio)
    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        errors.append("data train/val/test ratios must be non-negative and sum to 1.0")
    if cfg.data.train_ratio <= 0:
        errors.append("data train_ratio must be positive")
    if not cfg.data.instruction.strip():
        errors.append("data instruction must not be empty")
    if cfg.evaluation.max_new_tokens <= 0:
        errors.append("evaluation max_new_tokens must be positive")
    if errors:
        raise ConfigError("invalid company training config:\n  - " + "\n  - ".join(errors))
