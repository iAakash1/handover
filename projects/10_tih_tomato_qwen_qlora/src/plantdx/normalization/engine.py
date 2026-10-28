"""The dataset normalization engine.

Copies (or symlinks) the configured classes from the immutable raw datasets into
one canonical structure::

    datasets/<crop>/processed/<canonical_class>/<image files>

Layout detection is generic (no hardcoded split names): a *class directory* is a
directory that directly contains image files; its *split* is the parent directory
name when that parent is not the dataset root. This handles both ``root/class``
and ``root/train|val/class`` without assumptions.

Every copied file's SHA-256 is verified against the source. The raw datasets are
only read, never modified.
"""

from __future__ import annotations

import shutil
from logging import Logger
from pathlib import Path

from plantdx.config.schema import PlantDxConfig, SourceSpec
from plantdx.normalization import report as report_writer
from plantdx.normalization.models import CropReport, NormalizedImage
from plantdx.utils.hashing import sha256_bytes, sha256_hex
from plantdx.utils.logging import get_logger

# --------------------------------------------------------------------------- #
# Small filesystem helpers
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _relposix(path: Path, base: Path) -> str:
    """Path relative to ``base`` (POSIX), or absolute if it is outside ``base``."""
    resolved, base_resolved = Path(path).resolve(), Path(base).resolve()
    try:
        return resolved.relative_to(base_resolved).as_posix()
    except ValueError:
        return resolved.as_posix()


def _contains_images(directory: Path, extensions: set[str]) -> bool:
    """Whether ``directory`` directly contains at least one image file."""
    return any(p.is_file() and p.suffix.lower() in extensions for p in directory.iterdir())


def find_class_dirs(root: Path, extensions: set[str]) -> list[tuple[Path, str | None]]:
    """Return ``(class_dir, split)`` pairs for a dataset root (sorted, deterministic)."""
    if not root.exists():
        return []
    class_dirs: list[tuple[Path, str | None]] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if _contains_images(child, extensions):
            class_dirs.append((child, None))  # flat: child is a class
        else:
            for sub in sorted(
                p for p in child.iterdir() if p.is_dir() and not p.name.startswith(".")
            ):
                if _contains_images(sub, extensions):
                    class_dirs.append((sub, child.name))  # nested: split = child.name
    return class_dirs


def _place_file(
    src: Path, dst_dir: Path, checksum: str, mode: str, split: str | None, disambiguate: bool
) -> tuple[Path, str]:
    """Copy/link ``src`` into ``dst_dir``; return ``(dst, status)``.

    status is one of: ``placed``, ``disambiguated``, ``duplicate`` (identical file
    already present — skipped), ``collision_unresolved`` (name clash, different
    content, disambiguation disabled — skipped).
    """
    dst = dst_dir / src.name
    status = "placed"
    if dst.exists():
        if _sha256_file(dst) == checksum:
            return dst, "duplicate"
        if not disambiguate:
            return dst, "collision_unresolved"
        prefix = split or "dup"
        candidate = dst_dir / f"{prefix}__{src.name}"
        counter = 1
        while candidate.exists():
            candidate = dst_dir / f"{prefix}_{counter}__{src.name}"
            counter += 1
        dst, status = candidate, "disambiguated"

    if mode == "link":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
    return dst, status


def _verify(dst: Path, checksum: str, mode: str) -> bool:
    """Confirm the placed file's bytes hash to ``checksum``."""
    target = dst.resolve() if mode == "link" else dst
    return _sha256_file(target) == checksum


# --------------------------------------------------------------------------- #
# Per-crop normalization
# --------------------------------------------------------------------------- #


def normalize_crop(
    crop: str,
    source: SourceSpec,
    raw_root: Path,
    processed_base: Path,
    *,
    mode: str,
    extensions: set[str],
    disambiguate: bool,
    base_dir: Path,
) -> tuple[CropReport, list[NormalizedImage]]:
    """Normalize one crop; returns its report and the list of placed images.

    Copies only folders present in ``source.class_map``; every other folder is
    recorded in ``ignored_folders`` and left untouched.
    """
    exts = {e.lower() for e in extensions}
    base = Path(base_dir)
    processed_root = Path(processed_base) / crop / "processed"

    images: list[NormalizedImage] = []
    class_counts: dict[str, int] = {}
    ignored: list[str] = []
    disambiguated: list[str] = []
    duplicates: list[str] = []
    failures: list[str] = []

    class_dirs = find_class_dirs(Path(raw_root), exts)
    layout = "nested (split/class)" if any(split for _, split in class_dirs) else "flat (class)"

    for class_dir, split in class_dirs:
        canonical = source.class_map.get(class_dir.name)
        label = f"{split + '/' if split else ''}{class_dir.name}"
        if canonical is None:
            ignored.append(label)  # e.g. non-tomato PlantVillage crop — ignored, not copied
            continue

        dst_dir = processed_root / canonical
        dst_dir.mkdir(parents=True, exist_ok=True)
        source_files = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in exts
        )
        for src in source_files:
            checksum = _sha256_file(src)
            dst, status = _place_file(src, dst_dir, checksum, mode, split, disambiguate)
            if status == "duplicate":
                duplicates.append(_relposix(src, base))
                continue
            if status == "collision_unresolved":
                failures.append(_relposix(src, base))
                continue
            if status == "disambiguated":
                disambiguated.append(_relposix(dst, base))
            if not _verify(dst, checksum, mode):
                failures.append(_relposix(dst, base))
            images.append(
                NormalizedImage(
                    dataset=source.dataset,
                    crop=crop,
                    class_name=canonical,
                    split=split,
                    source_path=_relposix(src, base),
                    normalized_path=_relposix(dst, base),
                    checksum=checksum,
                )
            )
            class_counts[canonical] = class_counts.get(canonical, 0) + 1

    images.sort(key=lambda im: im.normalized_path)
    # Location-independent: identifies content + class organization, not the output path.
    dataset_checksum = sha256_hex(
        "\n".join(
            sorted(
                f"{im.class_name}/{Path(im.normalized_path).name}:{im.checksum}" for im in images
            )
        )
    )
    crop_report = CropReport(
        crop=crop,
        dataset=source.dataset,
        mode=mode,
        layout=layout,
        processed_dir=_relposix(processed_root, base),
        license=source.license,
        citation=source.citation,
        url=source.url,
        image_count=len(images),
        class_count=len(class_counts),
        class_counts=dict(sorted(class_counts.items())),
        ignored_folders=sorted(ignored),
        disambiguated=sorted(disambiguated),
        duplicates_skipped=sorted(duplicates),
        checksum_failures=sorted(failures),
        dataset_checksum=dataset_checksum,
    )
    return crop_report, images


# --------------------------------------------------------------------------- #
# Config-driven run
# --------------------------------------------------------------------------- #


def run_normalization(
    config: PlantDxConfig,
    *,
    base_dir: str | Path = ".",
    crops: list[str] | None = None,
    mode: str | None = None,
    plantdx_version: str = "0",
    config_hash: str = "",
    logger: Logger | None = None,
) -> dict[str, CropReport]:
    """Normalize the configured crops and write all output files."""
    log = logger or get_logger("normalize")
    base = Path(base_dir)
    norm = config.normalization
    extensions = set(config.audit.supported_extensions)
    processed_base = base / config.paths.processed_dir
    effective_mode = mode or norm.mode

    selected = crops or list(norm.sources)
    reports: dict[str, CropReport] = {}
    for crop in selected:
        source = norm.sources[crop]
        raw_root = base / config.paths.datasets[crop].root
        log.info("[%s] normalizing from %s (mode=%s) ...", crop, raw_root, effective_mode)
        crop_report, images = normalize_crop(
            crop,
            source,
            raw_root,
            processed_base,
            mode=effective_mode,
            extensions=extensions,
            disambiguate=norm.disambiguate_on_collision,
            base_dir=base,
        )
        report_writer.write_crop_outputs(processed_base / crop, source, crop_report, images)
        reports[crop] = crop_report
        log.info(
            "[%s] %d images -> %d classes (ignored %d folders, %d dup, %d failures) checksum=%s",
            crop,
            crop_report.image_count,
            crop_report.class_count,
            len(crop_report.ignored_folders),
            len(crop_report.duplicates_skipped),
            len(crop_report.checksum_failures),
            crop_report.dataset_checksum[:12],
        )

    report_writer.write_run_report(
        processed_base / "normalization_report.json",
        reports,
        plantdx_version=plantdx_version,
        config_hash=config_hash,
        mode=effective_mode,
    )
    return reports
