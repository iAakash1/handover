"""Build the exact ``mlx_vlm.lora`` command from a validated TrainingConfig.

This is the single source of truth for how a PlantDx config maps onto the
backend's CLI flags. The command is launched as a subprocess of the *current*
interpreter (``sys.executable``), so running ``plantdx train`` inside the env
that has mlx-vlm uses that same env for training. Building the argv has no side
effects and never launches anything.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from plantdx.training.config import TrainingConfig
from plantdx.training.lora import check_method_supported


def build_command(
    cfg: TrainingConfig,
    *,
    dataset_dir: str | Path,
    output_path: str | Path,
    resume_adapter: str | Path | None = None,
    split: str = "train",
) -> list[str]:
    """Return the full argv for ``python -m mlx_vlm.lora ...``.

    Raises :class:`~plantdx.core.exceptions.ConfigError` if the configured adapter
    method is not runnable by the installed backend (fail closed).
    """
    check_method_supported(cfg.lora)

    argv: list[str] = [
        sys.executable,
        "-m",
        "mlx_vlm.lora",
        "--model-path",
        cfg.model.model_path,
        "--dataset",
        str(dataset_dir),
        "--split",
        split,
        "--train-mode",
        "sft",
        "--learning-rate",
        _num(cfg.optim.learning_rate),
        "--batch-size",
        str(cfg.optim.batch_size),
        "--epochs",
        str(cfg.optim.epochs),
        "--max-seq-length",
        str(cfg.model.max_seq_length),
        "--gradient-accumulation-steps",
        str(cfg.optim.gradient_accumulation_steps),
        "--grad-clip",
        _num(cfg.optim.grad_clip),
        "--steps-per-report",
        str(cfg.logging.steps_per_report),
        "--steps-per-eval",
        str(cfg.logging.steps_per_eval),
        "--steps-per-save",
        str(cfg.checkpoint.steps_per_save),
        "--val-batches",
        str(cfg.logging.val_batches),
        "--lora-rank",
        str(cfg.lora.rank),
        "--lora-alpha",
        _num(cfg.lora.alpha),
        "--lora-dropout",
        _num(cfg.lora.dropout),
        "--assistant-id",
        str(cfg.model.assistant_id),
        "--output-path",
        str(output_path),
    ]
    if cfg.model.image_resize is not None:
        argv += ["--image-resize-shape", str(cfg.model.image_resize), str(cfg.model.image_resize)]
    if cfg.grad_checkpoint:
        argv.append("--grad-checkpoint")
    if cfg.lora.train_vision:
        argv.append("--train-vision")
    if cfg.optim.train_on_completions:
        argv.append("--train-on-completions")
    if resume_adapter is not None:
        argv += ["--adapter-path", str(resume_adapter)]
    return argv


def render_command(argv: list[str]) -> str:
    """A copy-pasteable, shell-quoted single line for the report."""
    return " ".join(shlex.quote(a) for a in argv)


def _num(value: float) -> str:
    """Format a number without a trailing ``.0`` where it is integral."""
    if isinstance(value, int):
        return str(value)
    if value == int(value):
        return str(int(value))
    return repr(value)
