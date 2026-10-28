"""Plain dataclasses describing normalization results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NormalizedImage:
    """One image placed into the canonical dataset (a manifest row)."""

    dataset: str
    crop: str
    class_name: str  # canonical class
    split: str | None  # train / val / None (flat layout)
    source_path: str  # relative to the repo root
    normalized_path: str  # relative to the repo root
    checksum: str  # SHA-256 of the file bytes


@dataclass
class CropReport:
    """Summary of normalizing one crop's dataset."""

    crop: str
    dataset: str
    mode: str  # copy | link
    layout: str  # "flat (class)" | "nested (split/class)"
    processed_dir: str
    license: str
    citation: str
    url: str
    image_count: int
    class_count: int
    class_counts: dict[str, int]
    ignored_folders: list[str] = field(default_factory=list)
    disambiguated: list[str] = field(default_factory=list)
    duplicates_skipped: list[str] = field(default_factory=list)
    checksum_failures: list[str] = field(default_factory=list)
    dataset_checksum: str = ""
