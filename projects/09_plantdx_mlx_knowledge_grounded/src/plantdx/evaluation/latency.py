"""Latency aggregation and system-info capture for the reproducibility manifest."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LatencyStats:
    """Aggregate latency/throughput/memory stats for one model over a run."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    tokens_per_sec: float
    images_per_sec: float
    mean_peak_memory_gb: float
    max_peak_memory_gb: float
    sample_count: int


def compute_latency_stats(
    runtimes_ms: list[float], generation_tokens: list[int], peak_memory_gb: list[float]
) -> LatencyStats:
    """Aggregate per-sample latency/token/memory telemetry into corpus stats."""
    n = len(runtimes_ms)
    if n == 0 or len(generation_tokens) != n or len(peak_memory_gb) != n:
        raise ValueError(
            "runtimes_ms, generation_tokens, and peak_memory_gb must be equal, non-empty"
        )

    ordered = sorted(runtimes_ms)
    total_seconds = sum(runtimes_ms) / 1000.0
    total_tokens = sum(generation_tokens)

    return LatencyStats(
        mean_ms=sum(runtimes_ms) / n,
        median_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        tokens_per_sec=(total_tokens / total_seconds) if total_seconds > 0 else 0.0,
        images_per_sec=(n / total_seconds) if total_seconds > 0 else 0.0,
        mean_peak_memory_gb=sum(peak_memory_gb) / n,
        max_peak_memory_gb=max(peak_memory_gb),
        sample_count=n,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower, upper = int(index), min(int(index) + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def capture_system_info(*, repo_root: str | Path = ".") -> dict[str, object]:
    """Capture hardware/software facts for exact reproduction (doc's manifest)."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "macos_version": _macos_version(),
        "cpu_count": _cpu_count(),
        "memory_gb": _total_memory_gb(),
        "git_commit": _git_commit(repo_root),
        "package_versions": _package_versions(),
    }


def _macos_version() -> str | None:
    try:
        release, _, _ = platform.mac_ver()
        return release or None
    except Exception:  # pragma: no cover - defensive, platform-dependent
        return None


def _cpu_count() -> int | None:
    import os

    return os.cpu_count()


def _total_memory_gb() -> float | None:
    try:
        import psutil

        return round(float(psutil.virtual_memory().total) / (1024**3), 2)
    except ImportError:
        return None


def _git_commit(repo_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _package_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    versions: dict[str, str] = {}
    for package in ("mlx", "mlx-vlm", "mlx-lm", "torch", "transformers", "plantdx"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions
