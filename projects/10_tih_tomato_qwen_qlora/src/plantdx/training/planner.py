"""Pre-flight training plan: iterations, checkpoints, and resource estimates.

Pure computation over the config + dataset stats — launches nothing. Estimates
are deliberately conservative ranges (clearly labelled) so the user can decide
whether to start a run, not precise guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass

from plantdx.training.config import TrainingConfig
from plantdx.training.data import DatasetStats

# Rough per-iteration wall-clock on an M4 Pro (24 GB) for a 7B 4-bit VLM with
# batch_size=1, gradient checkpointing, and 448px images. A range, not a promise.
_SEC_PER_ITER_LOW = 1.5
_SEC_PER_ITER_HIGH = 3.5

# Estimated peak unified-memory range for the same configuration (GB).
_PEAK_MEM_LOW_GB = 8.0
_PEAK_MEM_HIGH_GB = 14.0

# LoRA adapter checkpoint size grows with rank; ~1.5 MB per rank unit is a safe
# upper bound for this model's adapted linear layers.
_ADAPTER_MB_PER_RANK = 1.5


@dataclass(frozen=True)
class PlanEstimate:
    """Everything the report needs to describe a run without launching it."""

    train_rows: int
    val_rows: int
    test_rows: int
    effective_batch_size: int
    iters: int
    optimizer_updates: int
    num_checkpoints: int
    est_minutes_low: float
    est_minutes_high: float
    est_peak_mem_low_gb: float
    est_peak_mem_high_gb: float
    est_disk_mb: float
    warnings: tuple[str, ...]


def build_plan(cfg: TrainingConfig, stats: DatasetStats) -> PlanEstimate:
    """Compute the training plan from the config and the built dataset stats."""
    train_rows = stats.per_split.get("train", 0)
    bs = cfg.optim.batch_size
    ga = cfg.optim.gradient_accumulation_steps
    iters = (train_rows // bs) * cfg.optim.epochs if bs else 0
    updates = iters // ga if ga else iters
    num_checkpoints = iters // cfg.checkpoint.steps_per_save if cfg.checkpoint.steps_per_save else 0

    est_low_min = iters * _SEC_PER_ITER_LOW / 60.0
    est_high_min = iters * _SEC_PER_ITER_HIGH / 60.0

    adapter_mb = cfg.lora.rank * _ADAPTER_MB_PER_RANK
    kept = cfg.checkpoint.keep_last + (1 if cfg.checkpoint.keep_best else 0)
    est_disk_mb = adapter_mb * max(1, kept) + _jsonl_mb(stats)

    warnings = _warnings(cfg, stats)
    return PlanEstimate(
        train_rows=train_rows,
        val_rows=stats.per_split.get("validation", 0),
        test_rows=stats.per_split.get("test", 0),
        effective_batch_size=bs * ga,
        iters=iters,
        optimizer_updates=updates,
        num_checkpoints=num_checkpoints,
        est_minutes_low=round(est_low_min, 1),
        est_minutes_high=round(est_high_min, 1),
        est_peak_mem_low_gb=_PEAK_MEM_LOW_GB,
        est_peak_mem_high_gb=_PEAK_MEM_HIGH_GB,
        est_disk_mb=round(est_disk_mb, 1),
        warnings=warnings,
    )


def _jsonl_mb(stats: DatasetStats) -> float:
    """Approximate on-disk size of the three JSONL files (paths + text)."""
    approx_bytes_per_row = 400
    return stats.row_count * approx_bytes_per_row / 1_000_000.0


def _warnings(cfg: TrainingConfig, stats: DatasetStats) -> tuple[str, ...]:
    out: list[str] = []
    if cfg.optim.batch_size > 1:
        out.append("batch_size > 1 may exceed 24 GB for a 7B VLM; 1 is recommended.")
    if not cfg.grad_checkpoint:
        out.append("grad_checkpoint is off; enabling it lowers peak memory.")
    if cfg.model.image_resize is None:
        out.append("no image_resize set; large images can spike visual-token memory.")
    if stats.per_split.get("train", 0) == 0:
        out.append("the training split is empty; check discovery + splits.")
    return tuple(out)
