"""Plain dataclasses describing the results of a dataset audit.

Everything here is a simple value object — no behavior beyond a couple of derived
properties. These serialize directly to JSON via ``dataclasses.asdict``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageRecord:
    """The result of inspecting one image file (metadata only, never decoded fully)."""

    dataset: str
    class_name: str
    relpath: str  # path relative to the dataset root
    ok: bool
    file_size: int
    sha256: str
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    format: str | None = None
    ahash: str | None = None  # average hash (only when near-duplicate detection is on)
    error: str | None = None

    @property
    def aspect_ratio(self) -> float | None:
        """Width / height, rounded, or ``None`` if dimensions are unknown."""
        if self.width and self.height:
            return round(self.width / self.height, 4)
        return None


@dataclass(frozen=True)
class Issue:
    """A single audit finding (structural problem or data-quality warning)."""

    dataset: str
    kind: str  # missing_root, empty_dataset, empty_folder, unexpected_folder,
    #            unsupported_file, corrupt_image, class_count_mismatch, class_imbalance
    detail: str
    path: str | None = None


@dataclass(frozen=True)
class Stats:
    """Min / max / mean of a numeric quantity (e.g., image widths)."""

    min: float
    max: float
    mean: float

    @classmethod
    def of(cls, values: Iterable[float]) -> Stats | None:
        """Build stats from ``values``; return ``None`` if there are none."""
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        return cls(min=min(vals), max=max(vals), mean=round(sum(vals) / len(vals), 4))


@dataclass
class DatasetReport:
    """The full audit summary for one dataset."""

    key: str  # tomato | mango (the config key)
    name: str  # PlantVillage | MangoLeafBD
    root: str
    exists: bool
    num_images: int
    num_ok: int
    num_corrupt: int
    num_unsupported: int
    num_classes: int
    configured_classes: int | None
    class_counts: dict[str, int]
    width: Stats | None
    height: Stats | None
    aspect_ratio: Stats | None
    imbalance_ratio: float | None
    num_exact_duplicate_groups: int
    num_near_duplicate_groups: int
    dataset_checksum: str
    issues: list[Issue] = field(default_factory=list)


@dataclass
class AuditManifest:
    """The top-level, reproducible audit manifest (dataset version manifest)."""

    generated_at: str  # UTC timestamp (informational; excluded from checksums)
    plantdx_version: str
    config_hash: str
    tool: str
    settings: dict[str, object]
    datasets: dict[str, str]  # dataset key -> deterministic checksum
    audit_checksum: str  # deterministic checksum over all dataset checksums
    totals: dict[str, int]
    splits: dict[str, str]  # splitting is deferred to a later milestone
