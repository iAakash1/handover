"""Tiny end-to-end training smoke test.

Runs :func:`plantdx.company.train.run_training` — the same function the Train
button calls — over a handful of images for one or two optimizer steps, into a
throwaway directory. Proves image load, processor, forward, loss, backward,
optimizer, and adapter save all work before a multi-hour run starts.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from plantdx.company.classes import CompanyDataset, load_company_dataset
from plantdx.company.config import CompanyTrainConfig, load_company_train_config
from plantdx.company.dataset import prepare_dataset
from plantdx.company.modeling import adapter_is_valid
from plantdx.company.train import TrainingError, run_training
from plantdx.utils.io import write_json

SMOKE_DIRNAME = "_smoke"
RESULT_NAME = "smoke_result.json"


@dataclass(frozen=True)
class SmokeResult:
    """What the smoke test proved (or did not)."""

    passed: bool
    detail: str
    steps: int = 0
    final_loss: float | None = None
    seconds: float = 0.0
    peak_memory_gb: float | None = None
    adapter_dir: str = ""
    fingerprint: str = ""
    seconds_per_optimizer_step: float | None = None
    estimated_total_seconds: float | None = None
    resume_verified: bool = False
    validation_batch_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return asdict(self)


def smoke_marker(cfg: CompanyTrainConfig) -> Path:
    """Where the last smoke result is recorded for this run."""
    return cfg.report_dir / RESULT_NAME


def last_smoke(cfg: CompanyTrainConfig) -> SmokeResult | None:
    """Read the previous smoke result, or ``None`` if there is not one."""
    import json

    path = smoke_marker(cfg)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return SmokeResult(
        passed=bool(raw.get("passed")),
        detail=str(raw.get("detail", "")),
        steps=int(raw.get("steps", 0)),
        final_loss=raw.get("final_loss"),
        seconds=float(raw.get("seconds", 0.0)),
        peak_memory_gb=raw.get("peak_memory_gb"),
        adapter_dir=str(raw.get("adapter_dir", "")),
        fingerprint=str(raw.get("fingerprint", "")),
        seconds_per_optimizer_step=raw.get("seconds_per_optimizer_step"),
        estimated_total_seconds=raw.get("estimated_total_seconds"),
        resume_verified=bool(raw.get("resume_verified")),
        validation_batch_verified=bool(raw.get("validation_batch_verified")),
    )


def run_smoke_test(
    cfg: CompanyTrainConfig | None = None,
    dataset: CompanyDataset | None = None,
    *,
    images_per_class: int = 2,
    steps: int = 2,
    keep_artifacts: bool = False,
    local_files_only: bool = True,
    on_progress: Any = None,
) -> SmokeResult:
    """Train for a couple of steps on a few images and report the outcome.

    Args:
        cfg: Run config; loaded from the default path when omitted.
        dataset: Company dataset profile; loaded from ``cfg`` when omitted.
        images_per_class: Images sampled per class for the throwaway dataset.
        steps: Optimizer steps to take.
        keep_artifacts: Keep the throwaway directory instead of deleting it.
        local_files_only: Never reach the network for model files.
        on_progress: Forwarded to the training loop.
    """
    if steps != 2:
        raise TrainingError("The readiness smoke test requires exactly two optimizer steps")
    cfg = cfg or load_company_train_config()
    dataset = dataset or load_company_dataset(cfg.data.profile)

    from plantdx.company.dataset import load_manifest, split_path
    from plantdx.company.modeling import (
        JsonlRows,
        load_for_inference,
        generate_label,
        CompanyCollator,
    )
    from plantdx.company.integrity import fingerprint
    from plantdx.company.train import planned_steps, latest_checkpoint
    from plantdx.utils.io import write_jsonl
    import json
    import gc
    import torch

    prepare_dataset(cfg, dataset)
    scratch = cfg.dataset_dir.parent / SMOKE_DIRNAME
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    data_dir = scratch / "dataset"
    adapter_dir = scratch / "adapter"

    started = time.perf_counter()
    try:
        # Sample only the frozen training/validation splits; the test split stays unused.
        data_dir.mkdir(parents=True, exist_ok=True)
        manifest = load_manifest(cfg.dataset_dir)
        tiny = json.loads(json.dumps(manifest))
        from collections import Counter

        selected_train_hashes = set()
        for split in ("train", "validation", "test"):
            rows = JsonlRows(split_path(cfg.dataset_dir, split)).rows
            if split == "test":
                # A disjoint train-only placeholder for the unused smoke test split.
                rows = [
                    r
                    for r in JsonlRows(split_path(cfg.dataset_dir, "train")).rows
                    if r["sha256"] not in selected_train_hashes
                ][-11:]
            else:
                chosen = []
                for entry in dataset.classes:
                    chosen.extend(
                        [r for r in rows if r["class_id"] == entry.class_id][
                            : images_per_class if split == "train" else 1
                        ]
                    )
                rows = chosen
            if split == "train":
                selected_train_hashes = {r["sha256"] for r in rows}
            write_jsonl(data_dir / f"{split}.jsonl", rows)
            tiny["split"]["counts"][split] = len(rows)
            tiny["split"]["per_class"][split] = dict(Counter(r["class_id"] for r in rows))
        write_json(data_dir / "manifest.json", tiny)
        result = run_training(
            cfg,
            dataset,
            dataset_dir=data_dir,
            output_dir=adapter_dir,
            log_dir=scratch / "logs",
            report_dir=scratch / "reports",
            max_steps=2,
            stop_after_steps=1,
            on_progress=on_progress,
            local_files_only=local_files_only,
        )
        timings = json.loads(result.summary_path.read_text(encoding="utf-8"))[
            "optimizer_step_seconds"
        ]
        result = run_training(
            cfg,
            dataset,
            dataset_dir=data_dir,
            output_dir=adapter_dir,
            log_dir=scratch / "logs",
            report_dir=scratch / "reports",
            max_steps=2,
            resume=True,
            on_progress=on_progress,
            local_files_only=local_files_only,
        )
        timings += json.loads(result.summary_path.read_text(encoding="utf-8"))[
            "optimizer_step_seconds"
        ]
        if result.steps != 2 or len(timings) != 2:
            raise TrainingError("Resume did not complete exactly two total optimizer steps")
        # Reload the actual saved adapter and exercise a held-out validation batch/prediction.
        loaded = load_for_inference(cfg, adapter_dir, local_files_only=local_files_only)
        validation = JsonlRows(split_path(data_dir, "validation"))
        batch = CompanyCollator(
            loaded.processor,
            max_side=cfg.model.image_max_side,
            max_seq_length=cfg.model.max_seq_length,
        )([validation.rows[0]])
        assert int((batch["labels"] != -100).sum()) > 0
        started_eval = time.perf_counter()
        for row in validation.rows:
            generate_label(
                loaded,
                row["image"],
                row["question"],
                max_side=cfg.model.image_max_side,
                max_new_tokens=cfg.evaluation.max_new_tokens,
            )
        torch.cuda.synchronize()
        generation_seconds = (time.perf_counter() - started_eval) / len(validation)
        del loaded, batch
        gc.collect()
        torch.cuda.empty_cache()
        # Verify the resumed checkpoint includes optimizer, schedule and RNG state.
        checkpoint = latest_checkpoint(adapter_dir)
        if checkpoint is None:
            raise TrainingError("Smoke run did not save a resumable checkpoint")
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            if not (checkpoint / name).is_file():
                raise TrainingError(f"Smoke checkpoint missing {name}")
        seconds_per_step = max(timings)
        val_logs = [
            json.loads(line)
            for line in (scratch / "logs" / "loss_history.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        # Teacher-forced validation time is measured separately from autoregressive generation.
        val_loss_seconds = max(
            (r.get("validation_seconds_per_sample", 0) for r in val_logs), default=0
        )
        if val_loss_seconds <= 0:
            raise TrainingError("Validation throughput was not measured")
        n = manifest["split"]["counts"]
        updates = planned_steps(
            n["train"],
            batch_size=cfg.optim.batch_size,
            accumulation=cfg.optim.gradient_accumulation_steps,
            epochs=cfg.optim.epochs,
        )
        # Conservative: full max epochs, epoch validation loss + generation, final val/test,
        # checkpoint/loading overhead, and 15% throughput allowance. An additional 30min is reserved.
        estimate = (
            1.15
            * (
                updates * seconds_per_step
                + cfg.optim.epochs * n["validation"] * (generation_seconds + val_loss_seconds)
                + (n["validation"] + n["test"]) * generation_seconds
            )
            + 300
        )
        ok, reason = adapter_is_valid(adapter_dir)
        if not ok:
            outcome = SmokeResult(
                passed=False,
                detail=f"training ran but no usable adapter was written: {reason}",
                steps=result.steps,
                seconds=time.perf_counter() - started,
            )
        else:
            outcome = SmokeResult(
                passed=True,
                detail=(
                    f"{result.steps} optimizer step(s) on {result.train_samples} images; "
                    "adapter written and validated"
                ),
                steps=result.steps,
                final_loss=result.final_loss,
                seconds=time.perf_counter() - started,
                peak_memory_gb=result.peak_memory_gb,
                adapter_dir=str(adapter_dir),
                fingerprint=fingerprint(cfg, dataset),
                seconds_per_optimizer_step=seconds_per_step,
                estimated_total_seconds=estimate,
                resume_verified=True,
                validation_batch_verified=True,
            )
    except TrainingError as exc:
        outcome = SmokeResult(passed=False, detail=str(exc), seconds=time.perf_counter() - started)
    except Exception as exc:
        outcome = SmokeResult(
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            seconds=time.perf_counter() - started,
        )
    finally:
        if not keep_artifacts:
            shutil.rmtree(scratch, ignore_errors=True)

    if not keep_artifacts:
        outcome = replace(outcome, adapter_dir="")
    write_json(smoke_marker(cfg), outcome.to_dict())
    return outcome
