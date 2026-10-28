"""
Structural / integrity validation of collected prediction data.

This checks that the collected DATA is well-formed and complete: valid
JSON, all expected fields present on every record, no duplicate
(image, model) records, and internally consistent status/raw_output/
error combinations. It never inspects whether a prediction's CONTENT is
correct — that is expert evaluation, explicitly out of scope for this
phase (see README.md). Run automatically at the end of run_benchmark.py,
before aggregation, and available standalone:

    python scripts/validate.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import OUTPUTS_DIR
from parser import FIELD_VARIANTS

VALIDATION_REPORT_PATH = OUTPUTS_DIR / "validation_report.json"

REQUIRED_TOP_LEVEL_FIELDS = {
    "image",
    "model",
    "prompt",
    "status",
    "error",
    "raw_output",
    "parsed_output",
    "metrics",
    "timestamps",
    "image_metadata",
    "model_metadata",
}
REQUIRED_PARSED_OUTPUT_FIELDS = set(FIELD_VARIANTS.keys())
REQUIRED_METRICS_FIELDS = {
    "prompt_tokens",
    "generated_tokens",
    "prompt_tokens_per_sec",
    "generation_tokens_per_sec",
    "peak_memory_mb",
    "inference_time_sec",
}
REQUIRED_TIMESTAMP_FIELDS = {"start", "finish", "runtime_sec"}
REQUIRED_IMAGE_METADATA_FIELDS = {
    "filename",
    "relative_path",
    "class_folder",
    "width",
    "height",
    "file_size_bytes",
    "sha256",
}
REQUIRED_MODEL_METADATA_FIELDS = {
    "model_id",
    "model_name",
    "model_family",
    "parameters",
    "quantization",
    "mlx_model",
}

VALID_STATUSES = {"success", "generation_failed", "parse_failed", "model_load_failed"}


def _missing_keys(section: Any, required: set, prefix: str) -> List[str]:
    """Keys present in `required` but absent from `section` (a nested
    dict within a record). A section that isn't even a dict — which
    shouldn't happen, but is handled rather than raising — is reported
    as a single missing entry naming the section itself."""
    if not isinstance(section, dict):
        return [prefix.rstrip(".") or "<record>"]
    return [f"{prefix}{key}" for key in sorted(required) if key not in section]


def _record_missing_fields(record: Dict[str, Any]) -> List[str]:
    missing = _missing_keys(record, REQUIRED_TOP_LEVEL_FIELDS, "")
    missing += _missing_keys(record.get("parsed_output"), REQUIRED_PARSED_OUTPUT_FIELDS, "parsed_output.")
    missing += _missing_keys(record.get("metrics"), REQUIRED_METRICS_FIELDS, "metrics.")
    missing += _missing_keys(record.get("timestamps"), REQUIRED_TIMESTAMP_FIELDS, "timestamps.")
    missing += _missing_keys(record.get("image_metadata"), REQUIRED_IMAGE_METADATA_FIELDS, "image_metadata.")
    missing += _missing_keys(record.get("model_metadata"), REQUIRED_MODEL_METADATA_FIELDS, "model_metadata.")
    # A wholly-absent nested section (e.g. "image_metadata" missing
    # entirely) gets flagged both as a missing top-level key and as an
    # unreachable section by the calls above -- de-duplicate while
    # preserving order so each distinct issue is reported once.
    return list(dict.fromkeys(missing))


def _record_inconsistencies(record: Dict[str, Any]) -> List[str]:
    """Internal consistency checks only — e.g. a 'success' record should
    carry a non-empty raw_output and a null error; a failed one should
    carry a non-empty error. This never touches prediction correctness."""
    issues: List[str] = []
    status = record.get("status")
    if status not in VALID_STATUSES:
        issues.append(f"unrecognized status: {status!r}")
    if status == "success":
        if not record.get("raw_output"):
            issues.append("status is 'success' but raw_output is empty")
        if record.get("error") is not None:
            issues.append("status is 'success' but error is not null")
    elif status in ("generation_failed", "parse_failed", "model_load_failed"):
        if not record.get("error"):
            issues.append(f"status is {status!r} but error is empty")
    return issues


def validate_records(records: List[dict], malformed_files: List[str]) -> Dict[str, Any]:
    """
    Run all structural checks over the combined record set and return a
    JSON-serializable report (also written to disk by
    write_validation_report). Purely structural — see module docstring.
    """
    missing_field_records: List[Dict[str, Any]] = []
    inconsistent_records: List[Dict[str, Any]] = []
    seen: Dict[Tuple[Any, Any], int] = {}

    for r in records:
        key = (r.get("image"), r.get("model"))
        seen[key] = seen.get(key, 0) + 1

        missing = _record_missing_fields(r)
        if missing:
            missing_field_records.append(
                {
                    "image": r.get("image"),
                    "model": r.get("model"),
                    "missing_fields": missing,
                }
            )

        issues = _record_inconsistencies(r)
        if issues:
            inconsistent_records.append(
                {
                    "image": r.get("image"),
                    "model": r.get("model"),
                    "issues": issues,
                }
            )

    duplicate_records = [
        {"image": image, "model": model, "count": count}
        for (image, model), count in seen.items()
        if count > 1
    ]

    is_clean = not (malformed_files or missing_field_records or duplicate_records or inconsistent_records)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "malformed_json_files": malformed_files,
        "missing_field_records": missing_field_records,
        "duplicate_records": duplicate_records,
        "inconsistent_records": inconsistent_records,
        "is_clean": is_clean,
    }


def write_validation_report(report: Dict[str, Any], path: Path = VALIDATION_REPORT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def main() -> None:
    """Standalone entry point: validate whatever is currently in
    outputs/ without running any models or performing aggregation."""
    import aggregate  # deferred: keeps this module's own import graph minimal

    records, malformed_files = aggregate.load_all_predictions()
    report = validate_records(records, malformed_files)
    write_validation_report(report)
    print(
        f"Validated {report['total_records']} records — "
        f"{'clean' if report['is_clean'] else 'issues found'}"
    )
    print(f"  Report -> {VALIDATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
