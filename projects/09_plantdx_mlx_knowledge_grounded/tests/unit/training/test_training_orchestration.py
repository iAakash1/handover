"""Planner, scheduler, seeds, checkpoints, metrics — pure/local orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from plantdx.training import seeds
from plantdx.training.checkpoints import list_snapshots, prune, resolve_layout
from plantdx.training.config import CheckpointConfig, TrainingConfig
from plantdx.training.data import build_training_dataset
from plantdx.training.metrics import MetricsLogger, StepRecord
from plantdx.training.planner import build_plan
from plantdx.training.scheduler import lr_at, preview_curve


@pytest.mark.unit
def test_plan_iters_and_effective_batch(training_config: TrainingConfig, tmp_path: Path) -> None:
    stats = build_training_dataset(training_config.data, output_dir=tmp_path / "ds")
    plan = build_plan(training_config, stats)
    train_rows = stats.per_split["train"]
    assert plan.iters == (train_rows // 1) * 1
    assert plan.effective_batch_size == 8
    assert plan.optimizer_updates == plan.iters // 8


@pytest.mark.unit
def test_scheduler_warmup_and_decay(training_config: TrainingConfig) -> None:
    optim = training_config.optim  # warmup_steps=5, cosine, peak 1e-4, floor 1e-6
    assert lr_at(0, 100, optim) == pytest.approx(1e-4 * 1 / 5)
    assert lr_at(4, 100, optim) == pytest.approx(1e-4)  # end of warmup
    assert lr_at(99, 100, optim) == pytest.approx(1e-6, abs=1e-7)  # decays to floor
    curve = preview_curve(100, optim, points=5)
    assert len(curve) == 5


@pytest.mark.unit
def test_seed_derivation_is_deterministic() -> None:
    a = seeds.derive_seed(20260711, "split", "tomato")
    b = seeds.derive_seed(20260711, "split", "tomato")
    c = seeds.derive_seed(20260711, "split", "mango")
    assert a == b
    assert a != c
    env = seeds.resolve_env(20260711)
    assert env["PYTHONHASHSEED"] == "20260711"


@pytest.mark.unit
def test_checkpoint_layout_and_resume(tmp_path: Path) -> None:
    ckpt = CheckpointConfig(
        output_dir=str(tmp_path / "ck"), steps_per_save=10, keep_last=2, keep_best=True, resume=True
    )
    # no adapter yet -> nothing to resume from
    assert resolve_layout(ckpt).resume_from is None
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "adapters.safetensors").write_bytes(b"x")
    assert resolve_layout(ckpt).resume_from == tmp_path / "ck" / "adapters.safetensors"


@pytest.mark.unit
def test_prune_keeps_last_n(tmp_path: Path) -> None:
    root = tmp_path / "ck"
    root.mkdir()
    for step in (100, 200, 300, 400):
        (root / f"{step:07d}_adapters.safetensors").write_bytes(b"x")
    (root / "adapters.safetensors").write_bytes(b"x")  # final, never pruned
    removed = prune(root, keep_last=2)
    assert len(removed) == 2
    remaining = {p.name for p in list_snapshots(root)}
    assert remaining == {"0000300_adapters.safetensors", "0000400_adapters.safetensors"}
    assert (root / "adapters.safetensors").exists()


@pytest.mark.unit
def test_metrics_logger_tracks_best_and_writes(tmp_path: Path) -> None:
    logger = MetricsLogger(log_dir=tmp_path / "logs")
    assert logger.log_step(StepRecord(10, 0.1, 2.0, 1e-4, val_loss=1.0)) is True
    assert logger.log_step(StepRecord(20, 0.2, 1.5, 9e-5, val_loss=1.2)) is False
    assert logger.log_step(StepRecord(30, 0.3, 1.2, 8e-5, val_loss=0.8)) is True
    assert logger.best_val_loss == 0.8
    assert logger.best_step == 30
    paths = logger.write()
    assert Path(paths["csv"]).exists()
    assert Path(paths["json"]).exists()
    assert "Training metrics" in Path(paths["markdown"]).read_text()
