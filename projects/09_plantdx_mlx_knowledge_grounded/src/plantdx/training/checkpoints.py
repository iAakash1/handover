"""Checkpoint layout, resume detection, and retention (filesystem only).

mlx-vlm writes the adapter to ``<output_path>/adapters.safetensors`` and periodic
snapshots alongside it. This module owns the run's checkpoint directory: it
resolves the output path, finds an adapter to resume from, and prunes old
periodic snapshots to ``keep_last``. It never launches training and never touches
the frozen pipeline artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plantdx.training.config import CheckpointConfig

_ADAPTER_NAME = "adapters.safetensors"


@dataclass(frozen=True)
class CheckpointLayout:
    """Resolved paths for a run's checkpoints."""

    root: Path
    adapter_path: Path  # the final/latest adapter mlx-vlm writes
    resume_from: Path | None  # an existing adapter to resume from, if any


def resolve_layout(ckpt: CheckpointConfig) -> CheckpointLayout:
    """Resolve the checkpoint directory + adapter path, honoring ``resume``."""
    root = Path(ckpt.output_dir)
    adapter_path = root / _ADAPTER_NAME
    resume_from = None
    if ckpt.resume:
        resume_from = adapter_path if adapter_path.is_file() else _latest_snapshot(root)
    return CheckpointLayout(root=root, adapter_path=adapter_path, resume_from=resume_from)


def ensure_root(layout: CheckpointLayout) -> Path:
    """Create the checkpoint directory (idempotent) and return it."""
    layout.root.mkdir(parents=True, exist_ok=True)
    return layout.root


def list_snapshots(root: str | Path) -> list[Path]:
    """Return periodic snapshot files (e.g. ``*_000200_adapters.safetensors``) sorted."""
    root = Path(root)
    if not root.is_dir():
        return []
    snaps = [p for p in root.glob(f"*{_ADAPTER_NAME}") if p.name != _ADAPTER_NAME]
    return sorted(snaps, key=lambda p: p.name)


def prune(root: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the most recent ``keep_last`` periodic snapshots.

    Returns the list of removed paths. The final ``adapters.safetensors`` and any
    ``best*`` snapshot are never removed.
    """
    snaps = [p for p in list_snapshots(root) if not p.name.startswith("best")]
    removed: list[Path] = []
    if keep_last >= 0 and len(snaps) > keep_last:
        for path in snaps[: len(snaps) - keep_last]:
            path.unlink()
            removed.append(path)
    return removed


def _latest_snapshot(root: Path) -> Path | None:
    snaps = list_snapshots(root)
    return snaps[-1] if snaps else None
