"""Prepare a run (deterministic, side-effect-light) and, only on request, launch it.

``prepare_run`` does everything up to but not including training: load + validate
config, build the tomato dataset, compute the plan, resolve checkpoints, build the
exact command, and write the report. ``launch`` runs that command as a subprocess
of the current interpreter — it is the ONLY function that starts training and is
never called by preparation or dry-run paths.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from plantdx.training import seeds
from plantdx.training.checkpoints import CheckpointLayout, ensure_root, resolve_layout
from plantdx.training.command import build_command
from plantdx.training.config import TrainingConfig, load_training_config
from plantdx.training.data import DatasetStats, build_training_dataset
from plantdx.training.lora import check_method_supported
from plantdx.training.planner import PlanEstimate, build_plan
from plantdx.training.reports import build_report

_DEFAULT_DATASET_ROOT = Path("artifacts/training")


@dataclass(frozen=True)
class PreparedRun:
    """The complete, ready-to-launch description of a run — nothing started yet."""

    cfg: TrainingConfig
    dataset_dir: Path
    stats: DatasetStats
    plan: PlanEstimate
    layout: CheckpointLayout
    argv: list[str]
    report_paths: dict[str, str]


def prepare_run(
    config_path: str | Path,
    *,
    base_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
) -> PreparedRun:
    """Build everything needed to train, without training. Fails closed early."""
    cfg = load_training_config(config_path, base_dir=base_dir)
    check_method_supported(cfg.lora)  # reject e.g. dora before doing any work

    ds_dir = Path(dataset_dir) if dataset_dir else _DEFAULT_DATASET_ROOT / cfg.run_name / "dataset"
    stats = build_training_dataset(
        cfg.data, output_dir=ds_dir, instructions_asset=cfg.data.instructions_path
    )
    plan = build_plan(cfg, stats)

    layout = resolve_layout(cfg.checkpoint)
    ensure_root(layout)
    argv = build_command(
        cfg,
        dataset_dir=ds_dir,
        output_path=layout.adapter_path,
        resume_adapter=layout.resume_from,
    )
    report_paths = build_report(cfg, stats, plan, argv, output_dir=cfg.logging.report_dir)
    return PreparedRun(
        cfg=cfg,
        dataset_dir=ds_dir,
        stats=stats,
        plan=plan,
        layout=layout,
        argv=argv,
        report_paths=report_paths,
    )


def launch(prepared: PreparedRun, *, log_path: str | Path | None = None) -> int:
    """Run the training subprocess. THIS STARTS TRAINING. Returns the exit code.

    Streams the backend's stdout/stderr to the console and, if ``log_path`` is
    given, tees it to that file. Only ever called by the CLI's execute path.
    """
    env = dict(os.environ)
    env.update(seeds.resolve_env(prepared.cfg.seed))

    if log_path is None:
        completed = subprocess.run(prepared.argv, env=env, check=False)
        return completed.returncode

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_file.open("w", encoding="utf-8") as sink,
        subprocess.Popen(
            prepared.argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc,
    ):
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sink.write(line)
        return proc.wait()
