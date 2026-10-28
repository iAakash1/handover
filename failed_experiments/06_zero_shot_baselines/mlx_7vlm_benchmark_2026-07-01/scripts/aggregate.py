"""
Combine per-model prediction JSON files (outputs/predictions_<model>.json)
into one combined JSON file and one CSV, for easy downstream review.

Can be run standalone to rebuild the combined outputs at any time from
whatever per-model files currently exist in outputs/:

    python scripts/aggregate.py

It's also called automatically at the end of run_benchmark.py, after
validation (see validate.py).
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import OUTPUTS_DIR, RESULTS_DIR

COMBINED_JSON = OUTPUTS_DIR / "all_predictions.json"
COMBINED_CSV = RESULTS_DIR / "predictions.csv"


def load_all_predictions(outputs_dir: Path = OUTPUTS_DIR) -> Tuple[List[dict], List[str]]:
    """
    Load and concatenate every predictions_*.json file in outputs_dir,
    sorted by image then model.

    Returns (records, malformed_files). malformed_files lists the path
    of any file that failed to parse as JSON at all, so validate.py (and
    this module's own CLI) can report on it explicitly instead of the
    file silently vanishing from the combined output.
    """
    records: List[dict] = []
    malformed_files: List[str] = []
    for path in sorted(Path(outputs_dir).glob("predictions_*.json")):
        try:
            records.extend(json.loads(path.read_text()))
        except json.JSONDecodeError:
            malformed_files.append(str(path))
    records.sort(key=lambda r: (r.get("image") or "", r.get("model") or ""))
    return records, malformed_files


def _flatten_record(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten one nested prediction record into a single level of CSV
    columns. Missing nested sections (shouldn't normally happen, but
    handled defensively so a malformed record can't crash CSV export)
    contribute null columns rather than raising.
    """
    parsed: Dict[str, Any] = r.get("parsed_output") or {}
    metrics: Dict[str, Any] = r.get("metrics") or {}
    timestamps: Dict[str, Any] = r.get("timestamps") or {}
    image_meta: Dict[str, Any] = r.get("image_metadata") or {}
    model_meta: Dict[str, Any] = r.get("model_metadata") or {}
    return {
        "image": r.get("image"),
        "model": r.get("model"),
        "prompt": r.get("prompt"),
        "status": r.get("status"),
        "error": r.get("error"),
        "raw_output": r.get("raw_output"),
        "parsed_crop": parsed.get("crop"),
        "parsed_visible_symptoms": parsed.get("visible_symptoms"),
        "parsed_disease": parsed.get("disease"),
        "parsed_alternative_diagnoses": parsed.get("alternative_diagnoses"),
        "parsed_severity": parsed.get("severity"),
        "parsed_confidence": parsed.get("confidence"),
        "parsed_reasoning": parsed.get("reasoning"),
        "metric_prompt_tokens": metrics.get("prompt_tokens"),
        "metric_generated_tokens": metrics.get("generated_tokens"),
        "metric_prompt_tokens_per_sec": metrics.get("prompt_tokens_per_sec"),
        "metric_generation_tokens_per_sec": metrics.get("generation_tokens_per_sec"),
        "metric_peak_memory_mb": metrics.get("peak_memory_mb"),
        "metric_inference_time_sec": metrics.get("inference_time_sec"),
        "timestamp_start": timestamps.get("start"),
        "timestamp_finish": timestamps.get("finish"),
        "timestamp_runtime_sec": timestamps.get("runtime_sec"),
        "image_filename": image_meta.get("filename"),
        "image_relative_path": image_meta.get("relative_path"),
        "image_class_folder": image_meta.get("class_folder"),
        "image_width": image_meta.get("width"),
        "image_height": image_meta.get("height"),
        "image_file_size_bytes": image_meta.get("file_size_bytes"),
        "image_sha256": image_meta.get("sha256"),
        "model_id": model_meta.get("model_id"),
        "model_name": model_meta.get("model_name"),
        "model_family": model_meta.get("model_family"),
        "model_parameters": model_meta.get("parameters"),
        "model_quantization": model_meta.get("quantization"),
        "model_mlx": model_meta.get("mlx_model"),
    }


# Derived from _flatten_record's own output shape so the CSV header can
# never drift out of sync with the flattening logic above.
CSV_FIELDS: List[str] = list(_flatten_record({}).keys())


def write_combined_json(records: List[dict], path: Path = COMBINED_JSON) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))


def write_csv(records: List[dict], path: Path = COMBINED_CSV) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(_flatten_record(r))


def main() -> None:
    """Standalone entry point: rebuild the combined JSON/CSV from
    whatever per-model files currently exist in outputs/, without
    running any models or performing validation."""
    records, malformed_files = load_all_predictions()
    if malformed_files:
        print(f"  [warn] {len(malformed_files)} file(s) failed to parse as JSON and were skipped:")
        for f in malformed_files:
            print(f"           {f}")
    if not records:
        print(f"No usable predictions_*.json files found in {OUTPUTS_DIR}")
        return
    write_combined_json(records)
    write_csv(records)
    n_files = len(list(Path(OUTPUTS_DIR).glob("predictions_*.json")))
    print(f"Combined {len(records)} predictions from {n_files} model file(s)")
    print(f"  JSON -> {COMBINED_JSON}")
    print(f"  CSV  -> {COMBINED_CSV}")


if __name__ == "__main__":
    main()
