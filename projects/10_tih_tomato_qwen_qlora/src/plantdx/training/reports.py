"""Render the pre-flight training report (Markdown + JSON).

Assembles the config, the built dataset, the resource plan, the LR-schedule
preview, and the exact launch command into one human-readable report and a
machine-readable JSON twin. Pure I/O over already-computed values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plantdx.training.command import render_command
from plantdx.training.config import TrainingConfig
from plantdx.training.data import DatasetStats
from plantdx.training.planner import PlanEstimate
from plantdx.training.scheduler import preview_curve
from plantdx.utils.io import ensure_dir


def build_report(
    cfg: TrainingConfig,
    stats: DatasetStats,
    plan: PlanEstimate,
    argv: list[str],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write ``training_plan.md`` + ``training_plan.json``; return their paths."""
    out = ensure_dir(output_dir)
    md_path = out / "training_plan.md"
    json_path = out / "training_plan.json"
    md_path.write_text(_markdown(cfg, stats, plan, argv), encoding="utf-8")
    json_path.write_text(_json(cfg, stats, plan, argv) + "\n", encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path)}


def _markdown(cfg: TrainingConfig, stats: DatasetStats, plan: PlanEstimate, argv: list[str]) -> str:
    curve = preview_curve(plan.iters, cfg.optim)
    lines = [
        f"# Training plan — {cfg.run_name}",
        "",
        "> Pre-flight only. **No training has been run.** Launch it yourself with the",
        "> command at the bottom, on the Apple Silicon machine with mlx-vlm installed.",
        "",
        "## Model & method",
        f"- Model: `{cfg.model.model_path}` ({cfg.model.quantization_bits}-bit MLX)",
        f"- Adapter: **{cfg.lora.method}** (rank {cfg.lora.rank}, alpha {cfg.lora.alpha}, "
        f"dropout {cfg.lora.dropout}, train_vision={cfg.lora.train_vision})",
        f"- Objective: SFT, train_on_completions={cfg.optim.train_on_completions}",
        f"- Seed: {cfg.seed} | grad_checkpoint: {cfg.grad_checkpoint} "
        f"| image_resize: {cfg.model.image_resize}",
        "",
        f"## Dataset ({cfg.data.crop})",
        f"- Images discovered: **{stats.image_count}**",
        f"- Training rows: **{stats.row_count}** "
        f"(train {plan.train_rows} / val {plan.val_rows} / test {plan.test_rows})",
        f"- Response pool: frozen corpus `{stats.corpus_checksum}`",
        f"- Instructions: {stats.instruction_count} | "
        f"captions/image: {cfg.data.captions_per_image}",
        "",
        "| disease | rows |",
        "|---|---|",
    ]
    lines += [f"| {d} | {n} |" for d, n in stats.per_disease.items()]
    lines += [
        "",
        "## Schedule & budget (estimates)",
        f"- Effective batch size: **{plan.effective_batch_size}** "
        f"(micro {cfg.optim.batch_size} x grad-accum {cfg.optim.gradient_accumulation_steps})",
        f"- Iterations: **{plan.iters}** over {cfg.optim.epochs} epoch(s) "
        f"(~{plan.optimizer_updates} optimizer updates)",
        f"- Estimated wall-clock: **{plan.est_minutes_low:.0f}-{plan.est_minutes_high:.0f} min** "
        f"({plan.est_minutes_low / 60:.1f}-{plan.est_minutes_high / 60:.1f} h)",
        f"- Estimated peak memory: **{plan.est_peak_mem_low_gb:.0f}-"
        f"{plan.est_peak_mem_high_gb:.0f} GB** of 24 GB",
        f"- Checkpoints: every {cfg.checkpoint.steps_per_save} steps "
        f"(~{plan.num_checkpoints}), keep_last={cfg.checkpoint.keep_last}, "
        f"keep_best={cfg.checkpoint.keep_best}",
        f"- Estimated disk: **~{plan.est_disk_mb:.0f} MB** (adapters + dataset JSONL)",
        "",
        "### LR schedule preview",
        "| step | lr |",
        "|---|---|",
    ]
    lines += [f"| {s} | {lr:.2e} |" for s, lr in curve]
    if plan.warnings:
        lines += ["", "## Warnings"] + [f"- {w}" for w in plan.warnings]
    lines += [
        "",
        "## Outputs",
        f"- Checkpoints: `{cfg.checkpoint.output_dir}/`",
        f"- Logs (CSV/JSON/Markdown): `{cfg.logging.log_dir}/`",
        f"- This report: `{cfg.logging.report_dir}/`",
        "",
        "## Note on evaluation",
        "mlx-vlm 0.6.x trains on the `train` split; the `validation`/`test` splits are",
        "written and reserved for a separate evaluation milestone (no in-loop eval set).",
        "",
        "## The one command to run",
        "```bash",
        render_command(argv),
        "```",
        "",
    ]
    return "\n".join(lines)


def _json(cfg: TrainingConfig, stats: DatasetStats, plan: PlanEstimate, argv: list[str]) -> str:
    payload: dict[str, Any] = {
        "run_name": cfg.run_name,
        "model": cfg.model.model_path,
        "adapter_method": cfg.lora.method,
        "seed": cfg.seed,
        "dataset": {
            "image_count": stats.image_count,
            "row_count": stats.row_count,
            "per_split": stats.per_split,
            "per_disease": stats.per_disease,
            "corpus_checksum": stats.corpus_checksum,
        },
        "plan": {
            "iters": plan.iters,
            "effective_batch_size": plan.effective_batch_size,
            "optimizer_updates": plan.optimizer_updates,
            "num_checkpoints": plan.num_checkpoints,
            "est_minutes_low": plan.est_minutes_low,
            "est_minutes_high": plan.est_minutes_high,
            "est_peak_mem_gb": [plan.est_peak_mem_low_gb, plan.est_peak_mem_high_gb],
            "est_disk_mb": plan.est_disk_mb,
            "warnings": list(plan.warnings),
        },
        "command": render_command(argv),
        "command_argv": argv,
    }
    return json.dumps(payload, indent=2)
