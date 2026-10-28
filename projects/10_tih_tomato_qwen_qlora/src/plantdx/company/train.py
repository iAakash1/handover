"""QLoRA fine-tuning of Qwen2.5-VL on the company tomato dataset.

Uses the Hugging Face ``Trainer`` with a custom collator; nothing here
reimplements optimization, scheduling, or checkpointing. The smoke test calls
:func:`run_training` with the same arguments as the real run, so a passing smoke
test exercises the identical code path.
"""

from __future__ import annotations

import json
import math
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plantdx.company.classes import CompanyDataset
from plantdx.company.config import CompanyTrainConfig
from plantdx.company.dataset import load_manifest, split_path
from plantdx.company.environment import environment_facts, memory_snapshot
from plantdx.company.modeling import (
    CompanyCollator,
    JsonlRows,
    ModelingError,
    attach_lora,
    count_params,
    load_base_model,
    load_processor,
)
from plantdx.core.exceptions import PlantDxError
from plantdx.utils.io import write_json

LOSS_HISTORY = "loss_history.jsonl"
RUN_CONFIG = "run_config.json"
TRAIN_SUMMARY = "training_summary.json"


class TrainingError(PlantDxError):
    """Training could not start or did not complete."""


@dataclass
class TrainProgress:
    """Live training state, refreshed on every logging step."""

    step: int = 0
    total_steps: int = 0
    epoch: float = 0.0
    total_epochs: float = 0.0
    loss: float | None = None
    validation_loss: float | None = None
    learning_rate: float | None = None
    elapsed_seconds: float = 0.0
    memory: dict[str, float] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "epoch": round(self.epoch, 3),
            "total_epochs": self.total_epochs,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "validation_loss": self.validation_loss,
            "estimated_remaining_seconds": (
                self.elapsed_seconds / self.step * (self.total_steps - self.step)
                if self.step
                else None
            ),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "memory": self.memory,
            "message": self.message,
        }


@dataclass(frozen=True)
class TrainResult:
    """What one completed training run produced."""

    run_name: str
    adapter_dir: Path
    steps: int
    epochs: float
    final_loss: float | None
    train_seconds: float
    train_samples: int
    validation_samples: int
    test_samples: int
    trainable_params: int
    total_params: int
    peak_memory_gb: float | None
    summary_path: Path

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "run_name": self.run_name,
            "adapter_dir": str(self.adapter_dir),
            "steps": self.steps,
            "epochs": round(self.epochs, 3),
            "final_loss": self.final_loss,
            "train_seconds": round(self.train_seconds, 1),
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "peak_memory_gb": self.peak_memory_gb,
        }


ProgressHook = Callable[[TrainProgress], None]


def planned_steps(n_rows: int, *, batch_size: int, accumulation: int, epochs: float) -> int:
    """Optimizer updates the Trainer will perform for this dataset and schedule."""
    per_epoch = max(1, math.ceil(math.ceil(n_rows / batch_size) / accumulation))
    return max(1, int(per_epoch * epochs))


def run_training(
    cfg: CompanyTrainConfig,
    dataset: CompanyDataset,
    *,
    dataset_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    max_train_samples: int | None = None,
    max_steps: int | None = None,
    stop_after_steps: int | None = None,
    on_progress: ProgressHook | None = None,
    resume: bool = False,
    local_files_only: bool = False,
) -> TrainResult:
    """Fine-tune the adapter and save it.

    Args:
        cfg: The run configuration.
        dataset: The resolved company dataset profile (recorded for provenance).
        dataset_dir: Prepared dataset directory; defaults to ``cfg.dataset_dir``.
        output_dir: Adapter destination; defaults to ``cfg.checkpoint_dir``.
        log_dir: Loss-history destination; defaults to ``cfg.log_dir``.
        report_dir: Run-config/summary destination; defaults to ``cfg.report_dir``.
        max_train_samples: Train on only the first N rows (smoke test).
        max_steps: Stop after N optimizer steps (smoke test).
        stop_after_steps: Interrupt smoke after this step while preserving its full schedule.
        on_progress: Called on every logging step with the current state.
        resume: Resume from the newest checkpoint in ``output_dir``.
        local_files_only: Never reach the network for model files.

    Raises:
        TrainingError: The dataset is missing, or training failed.
    """
    import torch
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

    data_dir = Path(dataset_dir) if dataset_dir is not None else cfg.dataset_dir
    target = Path(output_dir) if output_dir is not None else cfg.checkpoint_dir
    logs = Path(log_dir) if log_dir is not None else cfg.log_dir
    reports = Path(report_dir) if report_dir is not None else cfg.report_dir
    target.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    from plantdx.company.integrity import identity, require_budget
    from plantdx.company.dataset import validate_prepared

    if max_steps is None:
        require_budget(cfg, dataset)
    manifest = validate_prepared(cfg, dataset, data_dir)
    signature = identity(cfg, dataset, data_dir)
    signature = json.loads(json.dumps(signature))
    identity_path = target / "run_identity.json"
    checkpoint = latest_checkpoint(target)
    if resume:
        if checkpoint is None or not identity_path.is_file():
            raise TrainingError("No compatible resumable checkpoint exists")
        if json.loads(identity_path.read_text(encoding="utf-8")) != signature:
            raise TrainingError("Resume rejected: run settings, split, or implementation changed")
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (checkpoint / name).is_file():
                raise TrainingError(f"Incomplete checkpoint: missing {name}")
    elif checkpoint is not None or (target / "adapter_model.safetensors").exists():
        raise TrainingError("Run already exists; use --resume with identical settings")
    write_json(identity_path, signature)
    set_seed(cfg.seed)
    train_rows = JsonlRows(split_path(data_dir, "train"), limit=max_train_samples)
    counts = manifest.get("split", {}).get("counts", {})
    validation_rows = JsonlRows(split_path(data_dir, "validation"))

    _emit(on_progress, TrainProgress(message="loading base model"))
    processor = load_processor(cfg, local_files_only=local_files_only)
    model = load_base_model(cfg, local_files_only=local_files_only, for_training=True)
    model = attach_lora(model, cfg)
    trainable, total = count_params(model)
    _emit(
        on_progress,
        TrainProgress(message=f"adapter attached ({trainable:,} trainable / {total:,} params)"),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    collator = CompanyCollator(
        processor,
        max_side=cfg.model.image_max_side,
        max_seq_length=cfg.model.max_seq_length,
    )

    expected = planned_steps(
        len(train_rows),
        batch_size=cfg.optim.batch_size,
        accumulation=cfg.optim.gradient_accumulation_steps,
        epochs=cfg.optim.epochs,
    )
    if max_steps is not None:
        expected = min(expected, max_steps)
    use_bf16 = cfg.model.compute_dtype == "bfloat16" and _bf16_ok()
    if not use_bf16:
        raise TrainingError("Company training requires BF16-capable CUDA")

    args = TrainingArguments(
        output_dir=str(target),
        per_device_train_batch_size=cfg.optim.batch_size,
        gradient_accumulation_steps=cfg.optim.gradient_accumulation_steps,
        num_train_epochs=cfg.optim.epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
        warmup_steps=max(0, int(expected * cfg.optim.warmup_ratio)),
        lr_scheduler_type=cfg.optim.scheduler,
        max_grad_norm=cfg.optim.grad_clip,
        gradient_checkpointing=cfg.optim.grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=not use_bf16 and cfg.model.compute_dtype == "float16",
        optim=cfg.optim.optimizer,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        restore_callback_states_from_checkpoint=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=cfg.seed,
        data_seed=cfg.seed,
        disable_tqdm=True,
        label_names=["labels"],
    )

    state = _ProgressState(total_epochs=float(cfg.optim.epochs))
    callback = _make_callback(
        state, on_progress, logs / LOSS_HISTORY, stop_after_steps=stop_after_steps
    )

    class CompanyTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return balanced_sampler(train_dataset or self.train_dataset, cfg.seed)

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            from plantdx.company.evaluate import run_evaluation
            from plantdx.company.modeling import LoadedModel

            was_training = self.model.training
            self.model.eval()
            old_cache = self.model.config.use_cache
            self.model.config.use_cache = True
            try:
                measured = run_evaluation(
                    cfg,
                    dataset,
                    split="validation",
                    adapter_dir=target,
                    dataset_dir=data_dir,
                    report_dir=reports / f"epoch-{self.state.epoch:.3f}-validation",
                    loaded=LoadedModel(
                        self.model,
                        self.processing_class,
                        cfg.model.model_id,
                        str(target),
                        trainable,
                        total,
                    ),
                )
            finally:
                self.model.config.use_cache = old_cache
                self.model.train(was_training)
            metrics.update(eval_f1_macro=measured.f1_macro, eval_accuracy=measured.accuracy)
            self.log(
                {
                    "eval_f1_macro": measured.f1_macro,
                    "eval_accuracy": measured.accuracy,
                    "eval_loss": metrics["eval_loss"],
                }
            )
            return metrics

    trainer = CompanyTrainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=validation_rows,
        processing_class=processor,
        data_collator=collator,
        callbacks=[callback, EarlyStoppingCallback(early_stopping_patience=1)],
    )

    _write_run_config(cfg, dataset, manifest, target, reports, trainable, total, expected)

    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=_resume_target(target) if resume else None)
    except ModelingError:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        raise TrainingError(
            "CUDA ran out of memory. Lower model.image_max_side (for example 336), "
            "keep optim.batch_size at 1, and raise gradient_accumulation_steps "
            f"to hold the effective batch size. ({exc})"
        ) from exc
    except Exception as exc:
        raise TrainingError(f"training failed: {type(exc).__name__}: {exc}") from exc
    elapsed = time.perf_counter() - started

    # Early stopping uses loss; final checkpoint selection uses validation macro F1.
    candidates = [
        row
        for row in trainer.state.log_history
        if "eval_f1_macro" in row and (target / f"checkpoint-{row['step']}").is_dir()
    ]
    if not candidates:
        raise TrainingError("No validated checkpoint was saved")
    best = max(
        candidates, key=lambda row: (row["eval_f1_macro"], row["eval_accuracy"], -row["eval_loss"])
    )
    trainer.state.best_model_checkpoint = str(target / f"checkpoint-{best['step']}")
    trainer._load_best_model()
    best_metadata = dict(
        best,
        checkpoint=trainer.state.best_model_checkpoint,
        selection_metric="validation macro F1",
        early_stopping_metric="eval_loss",
    )
    history = [
        row for row in trainer.state.log_history if "loss" in row and row["step"] <= best["step"]
    ]
    best_metadata["training_loss"] = history[-1]["loss"] if history else None
    write_json(reports / "best_checkpoint.json", best_metadata)
    write_json(target / "best_checkpoint.json", best_metadata)
    write_json(target / "class_mapping.json", manifest["class_map"])
    write_json(target / "dataset_manifest.json", manifest)
    trainer.save_model(str(target))
    processor.save_pretrained(str(target))

    memory = memory_snapshot()
    summary = TrainResult(
        run_name=cfg.run_name,
        adapter_dir=target,
        steps=int(result.global_step),
        epochs=state.last_epoch,
        final_loss=state.last_loss if state.last_loss is not None else float(result.training_loss),
        train_seconds=elapsed,
        train_samples=len(train_rows),
        validation_samples=int(counts.get("validation", 0)),
        test_samples=int(counts.get("test", 0)),
        trainable_params=trainable,
        total_params=total,
        peak_memory_gb=memory.get("max_reserved_gb"),
        summary_path=reports / TRAIN_SUMMARY,
    )
    write_json(
        summary.summary_path, {**summary.to_dict(), "optimizer_step_seconds": state.step_seconds}
    )
    _emit(
        on_progress,
        TrainProgress(
            step=summary.steps,
            total_steps=summary.steps,
            epoch=summary.epochs,
            total_epochs=float(cfg.optim.epochs),
            loss=summary.final_loss,
            elapsed_seconds=elapsed,
            memory=memory,
            message="training complete",
        ),
    )
    del trainer, model, processor
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _bf16_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def _resume_target(output_dir: Path) -> str | bool:
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    numbered = sorted(
        (p for p in checkpoints if p.name.rsplit("-", 1)[-1].isdigit()),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    return str(numbered[-1]) if numbered else False


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Newest ``checkpoint-N`` directory under ``output_dir``, or ``None``."""
    resolved = _resume_target(Path(output_dir))
    return Path(resolved) if isinstance(resolved, str) else None


def _emit(hook: ProgressHook | None, progress: TrainProgress) -> None:
    if hook is not None:
        hook(progress)


@dataclass
class _ProgressState:
    """Mutable state shared with the Trainer callback."""

    total_epochs: float
    last_loss: float | None = None
    last_epoch: float = 0.0
    step_started: float = 0.0
    step_seconds: list[float] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)


def _make_callback(
    shared: _ProgressState,
    on_progress: ProgressHook | None,
    history_path: Path,
    *,
    stop_after_steps=None,
) -> Any:
    """Build a ``TrainerCallback`` that mirrors log events to a hook and a file."""
    from transformers import TrainerCallback

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.touch(exist_ok=True)

    class _Callback(TrainerCallback):
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            shared.started = time.perf_counter()

        def on_step_begin(self, args, state, control, **kwargs):
            import torch

            torch.cuda.synchronize()
            shared.step_started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            import torch

            torch.cuda.synchronize()
            shared.step_seconds.append(time.perf_counter() - shared.step_started)
            # Stop with evaluation/checkpointing while reserving time for final reports.
            if time.perf_counter() - shared.started >= 3 * 3600 or (
                stop_after_steps is not None and state.global_step >= stop_after_steps
            ):
                control.should_training_stop = True
                control.should_evaluate = True
                control.should_save = True

        def on_save(self, args, state, control, **kwargs):
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": state.global_step,
                            "epoch": state.epoch,
                            "checkpoint_path": str(
                                Path(args.output_dir) / f"checkpoint-{state.global_step}"
                            ),
                        }
                    )
                    + "\n"
                )

        def on_log(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            logs = kwargs.get("logs") or {}
            trainer_state = _pick_state(state, kwargs)
            if "loss" in logs:
                shared.last_loss = float(logs["loss"])
            shared.last_epoch = float(getattr(trainer_state, "epoch", 0.0) or 0.0)
            progress = TrainProgress(
                step=int(getattr(trainer_state, "global_step", 0) or 0),
                total_steps=int(getattr(trainer_state, "max_steps", 0) or 0),
                epoch=shared.last_epoch,
                total_epochs=shared.total_epochs,
                loss=shared.last_loss,
                validation_loss=logs.get("eval_loss"),
                learning_rate=(float(logs["learning_rate"]) if "learning_rate" in logs else None),
                elapsed_seconds=time.perf_counter() - shared.started,
                memory=memory_snapshot(),
                message="training",
            )
            with history_path.open("a", encoding="utf-8") as handle:
                payload = progress.to_dict()
                if logs.get("eval_samples_per_second"):
                    payload["validation_seconds_per_sample"] = 1.0 / logs["eval_samples_per_second"]
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
            _emit(on_progress, progress)

    return _Callback()


def _pick_state(candidate: Any, kwargs: dict[str, Any]) -> Any:
    """Return whichever argument carries ``global_step`` (hook order varies)."""
    if hasattr(candidate, "global_step"):
        return candidate
    for value in kwargs.values():
        if hasattr(value, "global_step"):
            return value
    return candidate


def _write_run_config(
    cfg: CompanyTrainConfig,
    dataset: CompanyDataset,
    manifest: dict[str, Any],
    target: Path,
    reports: Path,
    trainable: int,
    total: int,
    expected_steps: int,
) -> None:
    """Freeze exactly what this run used, next to its adapter and in the report dir."""
    facts = environment_facts()
    payload = {
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "base_model": cfg.model.model_id,
        "quantization": (
            f"4-bit {cfg.model.bnb_4bit_quant_type}" if cfg.model.load_in_4bit else "none"
        ),
        "compute_dtype": cfg.model.compute_dtype,
        "image_max_side": cfg.model.image_max_side,
        "max_seq_length": cfg.model.max_seq_length,
        "method": "QLoRA" if cfg.model.load_in_4bit else "LoRA",
        "planned_optimizer_steps": expected_steps,
        "lora": {
            "rank": cfg.lora.rank,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "train_vision": cfg.lora.train_vision,
            "target_modules": list(cfg.lora.target_modules),
        },
        "optim": {
            "learning_rate": cfg.optim.learning_rate,
            "epochs": cfg.optim.epochs,
            "batch_size": cfg.optim.batch_size,
            "gradient_accumulation_steps": cfg.optim.gradient_accumulation_steps,
            "effective_batch_size": cfg.optim.effective_batch_size,
            "weight_decay": cfg.optim.weight_decay,
            "warmup_ratio": cfg.optim.warmup_ratio,
            "scheduler": cfg.optim.scheduler,
            "grad_clip": cfg.optim.grad_clip,
            "grad_checkpoint": cfg.optim.grad_checkpoint,
            "optimizer": cfg.optim.optimizer,
        },
        "dataset": {
            "name": dataset.name,
            "root": str(dataset.root),
            "instruction": cfg.data.instruction,
            "labels": list(dataset.labels),
            "split": manifest.get("split", {}).get("counts", {}),
            "split_seed": cfg.data.split_seed,
            "content_checksum": manifest.get("content_checksum", ""),
        },
        "params": {"trainable": trainable, "total": total},
        "environment": facts.to_dict(),
        "host": platform.node(),
        "config_path": str(cfg.config_path) if cfg.config_path else "",
    }
    write_json(target / RUN_CONFIG, payload)
    write_json(reports / RUN_CONFIG, payload)


def balanced_sampler(rows, seed):
    """Mild inverse-square-root weighting, reproducible at every epoch/resume."""
    from collections import Counter
    import torch
    from torch.utils.data import WeightedRandomSampler

    counts = Counter(row["class_id"] for row in rows.rows)
    weights = [1.0 / math.sqrt(counts[row["class_id"]]) for row in rows.rows]

    class EpochSampler(WeightedRandomSampler):
        epoch = 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            generator = torch.Generator().manual_seed(seed + self.epoch)
            return iter(
                torch.multinomial(
                    self.weights, self.num_samples, self.replacement, generator=generator
                ).tolist()
            )

    return EpochSampler(weights, len(rows), replacement=True)
