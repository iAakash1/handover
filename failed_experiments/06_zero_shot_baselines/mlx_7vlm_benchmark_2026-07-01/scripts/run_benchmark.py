"""
Main entry point: runs the zero-shot VLM benchmark across all configured
models and all images in the dataset, using an identical prompt for
every (model, image) pair, and writes standardized, richly-annotated
predictions to JSON and CSV.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --models mlx-community/Qwen2.5-VL-7B-Instruct-4bit,mlx-community/gemma-3-12b-it-4bit
    python scripts/run_benchmark.py --limit 1
    python scripts/run_benchmark.py --force

This phase only collects standardized predictions. It does not evaluate
correctness, does not compare anything against folder names, and does
not compute accuracy — see README.md.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aggregate
import config
import summary as summary_mod
import validate
from dataset_utils import DatasetItem, discover_images
from logging_utils import get_model_logger, get_run_logger
from mem_utils import set_seed
from metadata_utils import ModelMetadata, derive_model_metadata
from model_runner import generate_response, load_model, unload_model
from parser import FIELD_VARIANTS, parse_response


def slugify(model_id: str) -> str:
    """'mlx-community/Qwen2.5-VL-7B-Instruct-4bit' -> 'Qwen2.5-VL-7B-Instruct-4bit'."""
    return model_id.split("/")[-1]


def _empty_parsed_output() -> Dict[str, None]:
    return {key: None for key in FIELD_VARIANTS}


def _empty_metrics() -> Dict[str, None]:
    return {
        "prompt_tokens": None,
        "generated_tokens": None,
        "prompt_tokens_per_sec": None,
        "generation_tokens_per_sec": None,
        "peak_memory_mb": None,
        "inference_time_sec": None,
    }


def _build_load_failed_records(
    model_id: str,
    images: List[DatasetItem],
    error_message: str,
    attempt_start: datetime,
    attempt_finish: datetime,
) -> List[dict]:
    """
    One placeholder record per image when a model fails to load at all.
    Keeps the combined dataset complete — every image x every model has
    a record, whether successful or not — and gives validate.py and
    summary.json something honest to count, without ever inventing
    prediction content for a model that never actually ran.
    """
    model_metadata = derive_model_metadata(model_id)
    runtime = (attempt_finish - attempt_start).total_seconds()
    timestamps = {
        "start": attempt_start.isoformat(),
        "finish": attempt_finish.isoformat(),
        "runtime_sec": round(runtime, 3),
    }
    records = []
    for item in images:
        records.append(
            {
                "image": item["image_path"],
                "model": model_id,
                "prompt": config.PROMPT,
                "status": "model_load_failed",
                "error": error_message,
                "raw_output": None,
                "parsed_output": _empty_parsed_output(),
                "metrics": _empty_metrics(),
                "timestamps": timestamps,
                "image_metadata": item["image_metadata"],
                "model_metadata": model_metadata,
            }
        )
    return records


def _is_complete(records: List[dict], expected_count: int) -> bool:
    """
    A model's output file counts as 'complete' (skippable on resume)
    only if it has a record for every image AND the model actually
    loaded successfully at least once. If the model failed to load
    entirely, every record has status == 'model_load_failed', and the
    file is intentionally NOT treated as complete — the next run retries
    loading it, since a load failure is often transient/fixable (e.g.
    the model wasn't downloaded yet, or a dependency was missing).
    A completed run that has a few failed *images* (status
    'generation_failed' / 'parse_failed') is still treated as complete,
    since those failures were recorded, not silently skipped, and
    blindly re-running all images just because one failed would waste
    a lot of time on a large local model.
    """
    if len(records) < expected_count:
        return False
    return any(r.get("status") != "model_load_failed" for r in records)


def run_model(model_id: str, images: List[DatasetItem], force: bool) -> List[dict]:
    """
    Run one model over every image, writing results incrementally so a
    crash partway through never loses already-completed images. Returns
    the list of records for this model (loaded from disk if skipped).

    Failure handling: a failure at any single image (generation or
    parsing) is logged and recorded as its own failed record — it never
    stops the remaining images or the remaining models. A failure to
    load the model at all is likewise logged and recorded as a failed
    record for every image (see _build_load_failed_records) rather than
    silently producing no data for that model.
    """
    slug = slugify(model_id)
    out_path = config.OUTPUTS_DIR / f"predictions_{slug}.json"
    logger = get_model_logger(slug, config.LOGS_DIR)

    if not force and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if _is_complete(existing, len(images)):
                logger.info(f"[skip] {model_id} — already complete ({len(existing)} records)")
                return existing
        except json.JSONDecodeError:
            logger.warning(f"{out_path.name} exists but is not valid JSON — redoing this model.")

    logger.info(f"Loading {model_id} ...")
    load_start = datetime.now(timezone.utc)
    try:
        model, processor, model_config = load_model(model_id)
    except Exception as e:
        load_finish = datetime.now(timezone.utc)
        logger.error(f"Failed to load {model_id}: {e}")
        logger.error(
            "Recording a load-failure entry for every image and continuing with the rest of the benchmark."
        )
        records = _build_load_failed_records(model_id, images, str(e), load_start, load_finish)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2))
        return records

    load_elapsed = (datetime.now(timezone.utc) - load_start).total_seconds()
    logger.info(f"Loaded {model_id} in {load_elapsed:.1f}s")

    model_metadata: ModelMetadata = derive_model_metadata(model_id)
    records: List[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(images, 1):
        image_path = item["image_path"]
        logger.info(f"[{i}/{len(images)}] ({item['class_folder']}) {Path(image_path).name}")

        attempt_start = datetime.now(timezone.utc)
        attempt_start_perf = time.perf_counter()
        raw_text: Optional[str] = None
        metrics = _empty_metrics()
        timestamps: Dict[str, Any]
        generation_error: Optional[str] = None

        try:
            outcome = generate_response(
                model,
                processor,
                model_config,
                image_path,
                config.PROMPT,
                config.MAX_TOKENS,
                config.TEMPERATURE,
            )
            raw_text = outcome["raw_text"]
            metrics = outcome["metrics"]
            timestamps = outcome["timestamps"]
        except Exception as e:
            generation_error = str(e)
            attempt_finish = datetime.now(timezone.utc)
            elapsed = time.perf_counter() - attempt_start_perf
            metrics["inference_time_sec"] = round(elapsed, 3)
            timestamps = {
                "start": attempt_start.isoformat(),
                "finish": attempt_finish.isoformat(),
                "runtime_sec": round(elapsed, 3),
            }
            logger.error(f"    generation failed: {e}")

        parsed_output = _empty_parsed_output()
        parse_error: Optional[str] = None
        if raw_text is not None:
            try:
                parsed_output = parse_response(raw_text)
            except Exception as e:
                parse_error = str(e)
                logger.warning(f"    parsing failed (raw output preserved): {e}")

        if generation_error:
            status, error = "generation_failed", generation_error
        elif parse_error:
            status, error = "parse_failed", f"parsing raw output failed: {parse_error}"
        else:
            status, error = "success", None

        record = {
            "image": image_path,
            "model": model_id,
            "prompt": config.PROMPT,
            "status": status,
            "error": error,
            "raw_output": raw_text,
            "parsed_output": parsed_output,
            "metrics": metrics,
            "timestamps": timestamps,
            "image_metadata": item["image_metadata"],
            "model_metadata": model_metadata,
        }
        records.append(record)
        # Write after every image, not just at the end — partial
        # progress on a large model is never lost to a later crash.
        out_path.write_text(json.dumps(records, indent=2))

    unload_model(model, processor)
    n_ok = sum(1 for r in records if r["status"] == "success")
    logger.info(f"Done: {model_id} — {n_ok}/{len(records)} succeeded -> {out_path.name}")
    return records


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Zero-shot VLM benchmark for tomato leaf images.")
    arg_parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of model IDs to run (default: all configured models).",
    )
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per class folder (default: all). Useful for a quick smoke test.",
    )
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run models even if a completed predictions_<model>.json already exists.",
    )
    args = arg_parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else config.MODELS

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    run_logger = get_run_logger(config.LOGS_DIR)

    seeded = set_seed(config.SEED)
    run_logger.info("=" * 60)
    run_logger.info("  PlantDx — Zero-Shot VLM Benchmark (Tomato)")
    run_logger.info("=" * 60)
    run_logger.info(
        f"Seed        : {config.SEED} "
        f"({'applied' if seeded else 'no mx.random.seed API on this mlx version — skipped'})"
    )
    run_logger.info(f"Models      : {len(models)}")
    for m in models:
        run_logger.info(f"  - {m}")

    images = discover_images(config.DATASET_DIR, limit_per_class=args.limit)
    n_classes = len({i["class_folder"] for i in images})
    run_logger.info(f"Images      : {len(images)} across {n_classes} class folders")
    run_logger.info("Prompt      : identical for every (model, image) pair — see scripts/config.py")

    run_start = time.perf_counter()
    for model_id in models:
        run_model(model_id, images, force=args.force)
    total_elapsed = time.perf_counter() - run_start

    run_logger.info("=" * 60)
    run_logger.info(f"All requested models finished in {total_elapsed / 60:.1f} min.")

    run_logger.info("Loading combined predictions for validation + aggregation...")
    records, malformed_files = aggregate.load_all_predictions()

    run_logger.info("Validating outputs...")
    validation_report = validate.validate_records(records, malformed_files)
    validate.write_validation_report(validation_report)
    run_logger.info(
        f"Validation: {'clean' if validation_report['is_clean'] else 'issues found'} "
        f"-> {validate.VALIDATION_REPORT_PATH.name}"
    )

    run_logger.info("Aggregating per-model outputs into combined JSON + CSV...")
    aggregate.write_combined_json(records)
    aggregate.write_csv(records)
    run_logger.info(f"  Combined JSON -> {aggregate.COMBINED_JSON}")
    run_logger.info(f"  Combined CSV  -> {aggregate.COMBINED_CSV}")

    run_logger.info("Writing summary.json...")
    summary_mod.write_summary(records, total_elapsed)
    run_logger.info(f"  Summary -> {summary_mod.SUMMARY_PATH}")

    run_logger.info("")
    run_logger.info("This phase collects predictions only — no accuracy or")
    run_logger.info("correctness evaluation is performed. See README.md.")


if __name__ == "__main__":
    main()
