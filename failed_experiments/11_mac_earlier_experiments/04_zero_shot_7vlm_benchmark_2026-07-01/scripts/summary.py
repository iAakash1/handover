"""
Run-level summary statistics for the current combined dataset.

Reflects the CURRENT complete state of outputs/predictions_*.json — i.e.
cumulative across however many run_benchmark.py invocations it took to
get there, the same records aggregate.py combines — not just the delta
from the most recent invocation. The one exception is
last_run_duration_sec; see its docstring in build_summary() below.
"""

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

SUMMARY_PATH = config.OUTPUTS_DIR / "summary.json"


def _package_version(name: str) -> Optional[str]:
    """Read an installed package's version from its metadata. None if
    the package isn't installed/found — never a guess."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_info() -> Dict[str, Any]:
    """Software/config snapshot relevant to reproducing a run. This is
    genuinely run-level (identical for every record in a given run), so
    it lives once here rather than being duplicated onto every record."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mlx_version": _package_version("mlx"),
        "mlx_vlm_version": _package_version("mlx-vlm"),
        "seed": config.SEED,
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
    }


def build_summary(records: List[dict], last_run_duration_sec: Optional[float] = None) -> Dict[str, Any]:
    """
    Compute summary statistics from the current combined record set.

    Args:
        records: the combined records, as produced by
            aggregate.load_all_predictions().
        last_run_duration_sec: wall-clock duration of the most recent
            run_benchmark.py invocation, if this is being called right
            after one. Unlike the other statistics here, this is
            inherently tied to a single invocation, not the cumulative
            dataset — across a resumed, multi-session benchmark it
            reflects only the final session's duration, not the sum of
            every session. Pass None when summarizing existing outputs
            without having just run anything (e.g. from this module's
            own standalone entry point).
    """
    completed = [r for r in records if r.get("status") == "success"]
    failed = [r for r in records if r.get("status") != "success"]
    runtimes = [
        r["metrics"]["inference_time_sec"]
        for r in completed
        if isinstance(r.get("metrics"), dict)
        and isinstance(r["metrics"].get("inference_time_sec"), (int, float))
    ]
    failed_by_status = {
        status: sum(1 for r in failed if r.get("status") == status)
        for status in sorted({r.get("status") for r in failed})
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_models": len({r.get("model") for r in records}),
        "num_images": len({r.get("image") for r in records}),
        "num_total_records": len(records),
        "num_completed_runs": len(completed),
        "num_failed_runs": len(failed),
        "failed_by_status": failed_by_status,
        "average_runtime_sec": round(sum(runtimes) / len(runtimes), 3) if runtimes else None,
        "last_run_duration_sec": (
            round(last_run_duration_sec, 2) if last_run_duration_sec is not None else None
        ),
        "environment": _environment_info(),
    }


def write_summary(
    records: List[dict],
    last_run_duration_sec: Optional[float] = None,
    path: Path = SUMMARY_PATH,
) -> Dict[str, Any]:
    summary = build_summary(records, last_run_duration_sec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Standalone entry point: rebuild summary.json from whatever is
    currently in outputs/, without running any models."""
    import aggregate  # deferred: keeps this module's own import graph minimal

    records, _ = aggregate.load_all_predictions()
    summary = write_summary(records)
    print(json.dumps(summary, indent=2))
    print(f"\nSummary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
