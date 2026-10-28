"""Hardware and package facts for the company training path.

Every probe degrades to a reported reason instead of raising, so preflight can
show what is wrong rather than crashing the caller.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors")


@dataclass(frozen=True)
class GpuFacts:
    """What the CUDA device reports, or why it could not be queried."""

    available: bool
    reason: str
    name: str | None = None
    total_vram_gb: float | None = None
    capability: str | None = None
    bf16: bool = False
    cuda_version: str | None = None
    device_count: int = 0


@dataclass(frozen=True)
class EnvironmentFacts:
    """A snapshot of the machine and the installed stack."""

    python: str
    platform: str
    gpu: GpuFacts
    packages: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        payload = asdict(self)
        payload["gpu"] = asdict(self.gpu)
        return payload


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str]:
    """Installed version per package name, or ``"not installed"``."""
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def gpu_facts() -> GpuFacts:
    """Probe the CUDA device without raising."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency here
        return GpuFacts(available=False, reason=f"torch import failed: {exc}")

    try:
        if not torch.cuda.is_available():
            return GpuFacts(
                available=False,
                reason="torch.cuda.is_available() is False (no CUDA GPU or CPU-only build)",
                cuda_version=torch.version.cuda,
            )
        props = torch.cuda.get_device_properties(0)
        return GpuFacts(
            available=True,
            reason="",
            name=torch.cuda.get_device_name(0),
            total_vram_gb=round(props.total_memory / 1024**3, 2),
            capability=f"{props.major}.{props.minor}",
            bf16=bool(torch.cuda.is_bf16_supported()),
            cuda_version=torch.version.cuda,
            device_count=torch.cuda.device_count(),
        )
    except Exception as exc:
        return GpuFacts(available=False, reason=f"CUDA probe failed: {exc}")


def environment_facts() -> EnvironmentFacts:
    """Collect the full environment snapshot."""
    return EnvironmentFacts(
        python=platform.python_version(),
        platform=platform.platform(),
        gpu=gpu_facts(),
        packages=package_versions(),
    )


def memory_snapshot() -> dict[str, float]:
    """Allocated / reserved / peak CUDA memory in GB. Empty when unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        }
    except Exception:
        return {}


def free_disk_gb(path: str | Path) -> float | None:
    """Free space in GB on the volume holding ``path`` (nearest existing parent)."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return round(shutil.disk_usage(probe).free / 1024**3, 2)
    except OSError:
        return None


@dataclass(frozen=True)
class GpuTelemetry:
    """A live nvidia-smi sample. Every field is ``None`` when it cannot be read."""

    available: bool
    name: str | None = None
    utilization_percent: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None

    @property
    def memory_used_gb(self) -> float | None:
        """VRAM in use, in GB."""
        return round(self.memory_used_mb / 1024, 2) if self.memory_used_mb is not None else None

    @property
    def memory_total_gb(self) -> float | None:
        """Total VRAM, in GB."""
        return round(self.memory_total_mb / 1024, 2) if self.memory_total_mb is not None else None

    @property
    def memory_percent(self) -> float | None:
        """VRAM in use as a percentage of total."""
        if self.memory_used_mb is None or not self.memory_total_mb:
            return None
        return round(100.0 * self.memory_used_mb / self.memory_total_mb, 1)


_SMI_FIELDS = (
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)


def gpu_telemetry(timeout: float = 5.0) -> GpuTelemetry:
    """Sample the GPU through ``nvidia-smi``. Display only; never raises."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(_SMI_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return GpuTelemetry(available=False)
    if result.returncode != 0 or not result.stdout.strip():
        return GpuTelemetry(available=False)

    parts = [item.strip() for item in result.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != len(_SMI_FIELDS):
        return GpuTelemetry(available=False)

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return GpuTelemetry(
        available=True,
        name=parts[0] or None,
        utilization_percent=number(parts[1]),
        memory_used_mb=number(parts[2]),
        memory_total_mb=number(parts[3]),
        temperature_c=number(parts[4]),
        power_w=number(parts[5]),
    )
