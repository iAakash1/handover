"""``plantdx company`` subcommands.

Kept in its own module so the company training path adds one subparser and one
dispatch line to the main CLI rather than growing it. Heavy imports stay inside
the handlers, matching the rest of the CLI.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
from pathlib import Path
from typing import Any

COMMAND = "company"


def add_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``plantdx company`` and its subcommands."""
    parser = sub.add_parser(
        COMMAND,
        help="Company (TIH) tomato dataset: prepare, fine-tune, evaluate, serve",
    )
    inner = parser.add_subparsers(dest="subcommand", metavar="<subcommand>")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config",
            default=None,
            help="Training config (default: configs/train/qwen25vl_tih_tomato.yaml)",
        )
        p.add_argument(
            "--profile", default=None, help="Class profile (default: from the training config)"
        )
        p.add_argument("--dataset-root", default=None, help="Override the raw dataset root")

    p_audit = inner.add_parser("audit", help="Inventory the raw company dataset (read-only)")
    common(p_audit)
    p_audit.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    p_prepare = inner.add_parser(
        "prepare", help="Build train/validation/test JSONL from the raw dataset"
    )
    common(p_prepare)

    p_pre = inner.add_parser("preflight", help="Check everything training needs")
    common(p_pre)
    p_pre.add_argument(
        "--deep", action="store_true", help="Also load the processor and encode one sample"
    )
    p_pre.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    p_smoke = inner.add_parser("smoke", help="Tiny real training run that must produce an adapter")
    common(p_smoke)
    p_smoke.add_argument("--steps", type=int, default=2, help="Optimizer steps to take")
    p_smoke.add_argument("--images-per-class", type=int, default=2, help="Images sampled per class")
    p_smoke.add_argument("--keep", action="store_true", help="Keep the throwaway artifacts")

    p_train = inner.add_parser("train", help="Fine-tune the adapter with QLoRA on the GPU")
    common(p_train)
    p_train.add_argument("--run-name", default=None, help="Override the run name")
    p_train.add_argument("--epochs", type=float, default=None)
    p_train.add_argument("--learning-rate", type=float, default=None)
    p_train.add_argument("--lora-rank", type=int, default=None)
    p_train.add_argument("--lora-alpha", type=int, default=None)
    p_train.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p_train.add_argument("--image-max-side", type=int, default=None)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument(
        "--max-train-samples", type=int, default=None, help="Train on only the first N rows"
    )
    p_train.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps")
    p_train.add_argument("--resume", action="store_true", help="Resume from the newest checkpoint")
    p_train.add_argument(
        "--skip-preflight", action="store_true", help="Do not gate on the preflight checks"
    )
    p_train.add_argument(
        "--no-evaluate", action="store_true", help="Do not evaluate after training"
    )

    p_eval = inner.add_parser("evaluate", help="Evaluate an adapter on a held-out split")
    common(p_eval)
    p_eval.add_argument("--run-name", default=None)
    p_eval.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p_eval.add_argument("--adapter", default=None, help="Adapter directory (default: this run's)")
    p_eval.add_argument(
        "--base", action="store_true", help="Evaluate the untouched base model instead"
    )
    p_eval.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows")

    p_external = inner.add_parser(
        "evaluate-external", help="Benchmark an adapter on a folder-labelled external dataset"
    )
    common(p_external)
    p_external.add_argument(
        "--adapter", default=None, help="Adapter directory (default: this run's)"
    )
    p_external.add_argument(
        "--output-dir",
        default=None,
        help="Report directory (default: reports/<run-name>_external)",
    )
    p_external.add_argument(
        "--limit", type=int, default=None, help="Evaluate a balanced sample of at most N images"
    )

    p_predict = inner.add_parser("predict", help="Diagnose one image or a folder")
    common(p_predict)
    p_predict.add_argument("--image", default=None, help="Path to one image")
    p_predict.add_argument("--folder", default=None, help="Path to a folder of images")
    p_predict.add_argument("--adapter", default=None, help="Adapter directory (default: active)")
    p_predict.add_argument("--json", action="store_true", help="Emit JSON instead of text")

    p_activate = inner.add_parser("activate", help="Point the app at a trained adapter")
    common(p_activate)
    p_activate.add_argument("--run-name", default=None)
    p_activate.add_argument("--adapter", default=None, help="Adapter directory to activate")
    p_activate.add_argument(
        "--metrics", default=None, help="metrics.json to record alongside the pointer"
    )

    p_status = inner.add_parser("status", help="Show the active adapter and available runs")
    common(p_status)


def run(args: argparse.Namespace) -> int:
    """Dispatch a ``plantdx company`` subcommand. Returns a process exit code."""
    from plantdx.core.exceptions import PlantDxError

    subcommand = getattr(args, "subcommand", None)
    if subcommand is None:
        print("plantdx company: pick a subcommand (see --help)")
        return 0

    handlers = {
        "audit": _audit,
        "prepare": _prepare,
        "preflight": _preflight,
        "smoke": _smoke,
        "train": _train,
        "evaluate": _evaluate,
        "evaluate-external": _evaluate_external,
        "predict": _predict,
        "activate": _activate,
        "status": _status,
    }
    handler = handlers.get(subcommand)
    if handler is None:  # pragma: no cover - argparse rejects unknown names first
        print(f"plantdx company: unknown subcommand {subcommand!r}")
        return 2
    try:
        return handler(args)
    except PlantDxError as exc:
        print(f"plantdx company: {exc}")
        return 2


def _load(args: argparse.Namespace) -> tuple[Any, Any]:
    from plantdx.company.classes import load_company_dataset
    from plantdx.company.config import load_company_train_config

    cfg = load_company_train_config(args.config, overrides=_overrides(args))
    dataset = load_company_dataset(
        args.profile or cfg.data.profile, root_override=args.dataset_root
    )
    return cfg, dataset


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "run_name": "run_name",
        "seed": "seed",
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "lora_rank": "lora_rank",
        "lora_alpha": "lora_alpha",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "image_max_side": "image_max_side",
    }
    out: dict[str, Any] = {}
    for attr, key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            out[key] = value
    return out


def _audit(args: argparse.Namespace) -> int:
    from plantdx.company.dataset import scan_dataset

    _, dataset = _load(args)
    records, audit = scan_dataset(dataset, verify=True)
    if getattr(args, "json", False):
        print(json.dumps(audit.to_dict(), indent=2))
        return 0

    print(f"Company dataset : {dataset.name}")
    print(f"Root            : {dataset.root}")
    print(f"Files found     : {audit.total_files}")
    print(f"Usable images   : {audit.usable}")
    print(f"Corrupt         : {len(audit.corrupt)}")
    print(f"Duplicate groups: {audit.duplicate_groups} ({audit.duplicate_extra_files} extra files)")
    print(f"Dropped (dup)   : {audit.dropped_duplicates}")
    print(f"Dropped (label conflict): {audit.dropped_ambiguous}")
    print(f"Unmapped folders: {', '.join(audit.unmapped_dirs) or 'none'}")
    print(f"Extensions      : {audit.extensions}")
    print(f"Imbalance ratio : {audit.imbalance_ratio}x")
    print()
    print(f"{'class':24s} {'raw':>7s} {'usable':>7s}")
    for entry in dataset.classes:
        raw_count = audit.per_class_raw.get(entry.class_id, 0)
        print(f"{entry.class_id:24s} {raw_count:7d} {audit.per_class.get(entry.class_id, 0):7d}")
    del records
    return 0


def _prepare(args: argparse.Namespace) -> int:
    from plantdx.company.dataset import prepare_dataset

    cfg, dataset = _load(args)
    prepared = prepare_dataset(cfg, dataset)
    print(f"Dataset written : {prepared.dataset_dir}")
    print(f"Manifest        : {prepared.manifest_path}")
    for split, count in prepared.counts.items():
        print(f"  {split:11s} {count:6d}")
    print(f"Usable images   : {prepared.audit.usable} of {prepared.audit.total_files} files")
    return 0


def _preflight(args: argparse.Namespace) -> int:
    from plantdx.company.preflight import FAIL, OK, run_preflight

    report = run_preflight(
        config_path=args.config,
        profile_path=args.profile,
        deep=getattr(args, "deep", False),
    )
    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ready else 1

    for check in report.checks:
        mark = {OK: "[ok]  ", FAIL: "[fail]"}.get(check.status, "[warn]")
        print(f"{mark} {check.name:26s} {check.detail}")
    print()
    print("READY TO TRAIN" if report.ready else "NOT READY - fix the failing checks above")
    return 0 if report.ready else 1


def _smoke(args: argparse.Namespace) -> int:
    from plantdx.company.smoke import run_smoke_test

    cfg, dataset = _load(args)
    result = run_smoke_test(
        cfg,
        dataset,
        images_per_class=args.images_per_class,
        steps=args.steps,
        keep_artifacts=args.keep,
    )
    print(f"Smoke test: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  {result.detail}")
    if result.final_loss is not None:
        print(f"  final loss      : {result.final_loss:.4f}")
    if result.peak_memory_gb is not None:
        print(f"  peak GPU memory : {result.peak_memory_gb} GB")
    print(f"  elapsed         : {result.seconds:.1f}s")
    return 0 if result.passed else 1


def _train(args: argparse.Namespace) -> int:
    """Run the official training job.

    This is the only code path that starts a full training run, and it starts one
    only when a person invokes it (the Streamlit RUN TRAINING button, or this
    command by hand). Nothing else in the package calls it: preflight, dataset
    preparation, and the smoke test are all separate commands that stop on their
    own, so none of them can roll into a multi-hour run.
    """
    from plantdx.company.dataset import prepare_dataset
    from plantdx.company.preflight import run_preflight
    from plantdx.company.train import TrainProgress, run_training

    cfg, dataset = _load(args)

    if (
        args.skip_preflight
        or args.no_evaluate
        or args.max_steps is not None
        or args.max_train_samples is not None
    ):
        print(
            "Official training requires preflight and complete validation/test; use company smoke for tiny runs."
        )
        return 1
    print("Preparing dataset...")
    prepared = prepare_dataset(cfg, dataset)
    print(f"  {prepared.counts}")
    if not args.skip_preflight:
        report = run_preflight(
            config_path=args.config,
            profile_path=args.profile,
            overrides=_overrides(args),
            deep=True,
        )
        for check in report.blocking:
            print(f"[fail] {check.name}: {check.detail}")
        if not report.ready:
            print("Preflight failed; nothing was trained.")
            return 1

    def show(progress: TrainProgress) -> None:
        if progress.message and progress.message != "training":
            print(f"  {progress.message}")
            return
        loss = f"{progress.loss:.4f}" if progress.loss is not None else "n/a"
        lr = f"{progress.learning_rate:.2e}" if progress.learning_rate is not None else "n/a"
        memory = progress.memory.get("max_reserved_gb")
        print(
            f"  step {progress.step}/{progress.total_steps} "
            f"epoch {progress.epoch:.2f}/{progress.total_epochs:g} "
            f"loss {loss} val_loss {progress.validation_loss} lr {lr} mem {memory}GB "
            f"elapsed {progress.elapsed_seconds:.0f}s eta {progress.to_dict()['estimated_remaining_seconds']}s"
        )

    # A completed optimization phase can resume an interrupted final evaluation.
    from plantdx.company.train import TRAIN_SUMMARY

    if args.resume and (cfg.report_dir / TRAIN_SUMMARY).is_file():
        from plantdx.company.integrity import identity

        frozen = json.loads((cfg.checkpoint_dir / "run_identity.json").read_text(encoding="utf-8"))
        if frozen != json.loads(json.dumps(identity(cfg, dataset))):
            raise ValueError("Resume rejected: completed run identity differs")
        from plantdx.company.evaluate import finalize_run

        validation, test = finalize_run(cfg, dataset, cfg.checkpoint_dir)
        _print_eval(validation)
        _print_eval(test)
        print("COMPANY RUN COMPLETE: best adapter, validation and final test saved")
        return 0
    print(f"Training {cfg.run_name}...")
    result = run_training(
        cfg,
        dataset,
        max_train_samples=args.max_train_samples,
        max_steps=args.max_steps,
        on_progress=show,
        resume=args.resume,
    )
    print(f"Adapter written : {result.adapter_dir}")
    print(f"Steps           : {result.steps}")
    print(f"Final loss      : {result.final_loss}")
    print(f"Elapsed         : {result.train_seconds / 60:.1f} min")
    print(f"Peak GPU memory : {result.peak_memory_gb} GB")

    from plantdx.company.evaluate import finalize_run

    validation, test = finalize_run(cfg, dataset, result.adapter_dir)
    _print_eval(validation)
    _print_eval(test)
    print("COMPANY RUN COMPLETE: best adapter, validation and final test saved")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from plantdx.company.evaluate import run_evaluation

    cfg, dataset = _load(args)
    adapter = None if args.base else (args.adapter or cfg.checkpoint_dir)
    evaluation = run_evaluation(
        cfg,
        dataset,
        split=args.split,
        adapter_dir=adapter,
        limit=args.limit,
        on_progress=_ticker(),
    )
    _print_eval(evaluation)
    return 0


def _evaluate_external(args: argparse.Namespace) -> int:
    from plantdx.company.classes import load_company_dataset
    from plantdx.company.config import load_company_train_config
    from plantdx.company.external_evaluate import run_external_benchmark

    if not args.dataset_root:
        print("plantdx company: evaluate-external requires --dataset-root")
        return 2
    cfg = load_company_train_config(args.config)
    dataset = load_company_dataset(args.profile or cfg.data.profile)
    result = run_external_benchmark(
        cfg,
        dataset,
        args.dataset_root,
        adapter_dir=args.adapter,
        output_dir=args.output_dir,
        limit=args.limit,
        on_progress=_ticker(),
    )
    if result.ignored_folders:
        print(f"Ignored unmapped folders: {', '.join(result.ignored_folders)}")
    print()
    print("External Benchmark Results")
    print()
    print(f"Dataset: {result.dataset_root}")
    print(f"Model: {result.model} + {result.adapter}")
    print()
    print(f"Images: {result.total_images}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    print(f"Weighted F1: {result.weighted_f1:.4f}")
    print(f"Latency: {result.average_latency_seconds:.3f} sec/image")
    print(f"Reports: {result.output_dir}")
    return 0


def _ticker() -> Any:
    def tick(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}")

    return tick


def _print_eval(result: Any) -> None:
    print()
    print(f"Split           : {result.split} ({result.samples} images)")
    print(f"Model           : {result.adapter or 'base model (no adapter)'}")
    print(f"Accuracy        : {result.accuracy:.4f}")
    print(f"Macro F1        : {result.f1_macro:.4f}")
    print(f"Weighted F1     : {result.f1_weighted:.4f}")
    print(f"Macro precision : {result.precision_macro:.4f}")
    print(f"Macro recall    : {result.recall_macro:.4f}")
    print(f"Unparseable     : {result.unparseable}")
    print(f"Latency         : {result.seconds_per_sample:.2f} s/image")
    print()
    print(f"{'class':24s} {'support':>8s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
    for row in result.per_class:
        print(
            f"{row.class_id:24s} {row.support:8d} {row.precision:7.3f} "
            f"{row.recall:7.3f} {row.f1:7.3f}"
        )
    if result.confusions:
        print()
        print("Most confused pairs:")
        for pair in result.confusions[:5]:
            print(f"  {pair['true']} -> {pair['predicted']}: {pair['count']}")


def _predict(args: argparse.Namespace) -> int:
    from plantdx.company.modeling import ModelingError
    from plantdx.company.predict import load_predictor, predict_image

    cfg, dataset = _load(args)
    paths: list[Path] = []
    if args.image:
        paths.append(Path(args.image))
    if args.folder:
        folder = Path(args.folder)
        allowed = set(dataset.extensions)
        paths.extend(
            sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
        )
    if not paths:
        raise ModelingError("pass --image or --folder")

    loaded = load_predictor(cfg, adapter_dir=args.adapter)
    results = [predict_image(loaded, p, cfg=cfg, dataset=dataset) for p in paths]

    if getattr(args, "json", False):
        print(
            json.dumps(
                [{"image": str(p), **r.to_dict()} for p, r in zip(paths, results, strict=True)],
                indent=2,
            )
        )
        return 0
    for path, result in zip(paths, results, strict=True):
        confidence = f"{result.confidence:.3f}" if result.confidence is not None else "n/a"
        print(f"{path.name:44s} -> {result.label:20s} (conf {confidence}, {result.seconds:.2f}s)")
    return 0


def _activate(args: argparse.Namespace) -> int:
    from plantdx.company.evaluate import load_result
    from plantdx.company.modeling import ModelingError, adapter_is_valid
    from plantdx.company.registry import ActiveAdapter, write_active_adapter

    cfg, dataset = _load(args)
    adapter = Path(args.adapter).resolve() if args.adapter else cfg.checkpoint_dir
    ok, reason = adapter_is_valid(adapter)
    if not ok:
        raise ModelingError(reason)

    metrics: dict[str, float] = {}
    metrics_path = Path(args.metrics) if args.metrics else cfg.report_dir / "validation_finetuned"
    try:
        result = load_result(metrics_path)
        metrics = {
            "accuracy": result.accuracy,
            "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
        }
    except Exception:
        metrics = {}

    entry = ActiveAdapter(
        crop=dataset.crop,
        run_name=cfg.run_name,
        adapter_path=str(adapter),
        base_model=cfg.model.model_id,
        dataset=dataset.name,
        backend="transformers_cuda",
        classes=dataset.labels,
        metrics=metrics,
        activated_at=_datetime.datetime.now(_datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    path = write_active_adapter(entry, cfg.base_dir)
    print(f"Active adapter for {entry.crop}: {adapter}")
    print(f"Registry: {path}")
    return 0


def _status(args: argparse.Namespace) -> int:
    from plantdx.company.modeling import adapter_is_valid
    from plantdx.company.registry import read_registry

    cfg, dataset = _load(args)
    registry = read_registry(cfg.base_dir)
    if not registry:
        print("No active adapter registered.")
    for crop, entry in registry.items():
        ok, reason = adapter_is_valid(entry.resolved_path(cfg.base_dir))
        print(f"{crop}: {entry.run_name} -> {entry.adapter_path}")
        print(f"  base model : {entry.base_model}")
        print(f"  dataset    : {entry.dataset}")
        print(f"  status     : {'valid' if ok else reason}")
        if entry.metrics:
            print(f"  metrics    : {entry.metrics}")

    checkpoints = cfg.base_dir / "checkpoints"
    if checkpoints.is_dir():
        print()
        print("Checkpoint directories:")
        for path in sorted(p for p in checkpoints.iterdir() if p.is_dir()):
            ok, _ = adapter_is_valid(path)
            print(f"  {path.name:34s} {'valid adapter' if ok else 'no adapter files'}")
    del dataset
    return 0
