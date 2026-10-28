"""The dataset audit engine.

Runs entirely on CPU. For each configured dataset it discovers images, inspects
them (metadata + SHA-256), detects duplicates, computes summary statistics, and
writes a set of plain CSV/JSON/Markdown reports plus a deterministic manifest.

Determinism: images are processed in sorted order and all report rows are sorted,
so the reports and the checksums are identical across runs regardless of the
thread count. Only the manifest's ``generated_at`` timestamp varies, and it is
deliberately excluded from every checksum.
"""

from __future__ import annotations

import concurrent.futures as futures
from collections import Counter
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path

from plantdx.audit import discovery, duplicates, images, report
from plantdx.audit.discovery import DatasetSpec
from plantdx.audit.models import (
    AuditManifest,
    DatasetReport,
    ImageRecord,
    Issue,
    Stats,
)
from plantdx.config.schema import AuditConfig
from plantdx.utils.hashing import sha256_hex
from plantdx.utils.io import ensure_dir
from plantdx.utils.logging import configure_logging, get_logger

_SPLIT_STATUS = {
    "status": "not_performed",
    "note": "Dataset splitting is a later milestone; the audit only inventories images.",
}


def _inspect_all(
    image_items: list[tuple[Path, str]],
    dataset_key: str,
    root: Path,
    settings: AuditConfig,
) -> list[ImageRecord]:
    """Inspect every image, using a thread pool when it helps (I/O bound work).

    ``ThreadPoolExecutor.map`` preserves input order, so results stay sorted and
    the output is deterministic regardless of worker count.
    """

    def inspect(item: tuple[Path, str]) -> ImageRecord:
        path, class_name = item
        return images.inspect_image(
            path,
            dataset_key,
            class_name,
            root,
            compute_ahash=settings.near_duplicates,
            ahash_size=settings.ahash_size,
        )

    if settings.workers > 1 and len(image_items) > 1:
        with futures.ThreadPoolExecutor(max_workers=settings.workers) as pool:
            return list(pool.map(inspect, image_items))
    return [inspect(item) for item in image_items]


def _dataset_checksum(records: list[ImageRecord]) -> str:
    """Deterministic checksum over the file set (relpath + content hash)."""
    lines = sorted(f"{r.relpath}:{r.sha256}" for r in records)
    return sha256_hex("\n".join(lines))


def _audit_dataset(
    spec: DatasetSpec, settings: AuditConfig, log: Logger
) -> tuple[
    DatasetReport, list[ImageRecord], list[tuple[str, list[str]]], list[tuple[str, list[str]]]
]:
    """Audit one dataset; return its report, records, and duplicate groups."""
    issues: list[Issue] = []

    if not spec.root.exists():
        log.warning("[%s] dataset root missing: %s", spec.key, spec.root)
        issues.append(
            Issue(spec.key, "missing_root", f"root does not exist: {spec.root}", str(spec.root))
        )
        empty_report = DatasetReport(
            key=spec.key,
            name=spec.name,
            root=str(spec.root),
            exists=False,
            num_images=0,
            num_ok=0,
            num_corrupt=0,
            num_unsupported=0,
            num_classes=0,
            configured_classes=spec.configured_classes,
            class_counts={},
            width=None,
            height=None,
            aspect_ratio=None,
            imbalance_ratio=None,
            num_exact_duplicate_groups=0,
            num_near_duplicate_groups=0,
            dataset_checksum=sha256_hex(""),
            issues=issues,
        )
        return empty_report, [], [], []

    image_items, unsupported = discovery.discover_images(spec.root, settings.supported_extensions)
    empty_dirs, unexpected_dirs = discovery.find_dir_issues(spec.root)

    for path in unsupported:
        issues.append(
            Issue(
                spec.key,
                "unsupported_file",
                "unsupported file type",
                str(path.relative_to(spec.root)),
            )
        )
    for directory in empty_dirs:
        issues.append(
            Issue(
                spec.key,
                "empty_folder",
                "folder has no visible contents",
                str(directory.relative_to(spec.root)),
            )
        )
    for directory in unexpected_dirs:
        issues.append(
            Issue(
                spec.key,
                "unexpected_folder",
                "hidden/system folder",
                str(directory.relative_to(spec.root)),
            )
        )
    if not image_items:
        issues.append(Issue(spec.key, "empty_dataset", "no supported images found", str(spec.root)))

    log.info("[%s] inspecting %d images ...", spec.key, len(image_items))
    records = _inspect_all(image_items, spec.key, spec.root, settings)

    corrupt = [r for r in records if not r.ok]
    for record in corrupt:
        issues.append(
            Issue(spec.key, "corrupt_image", record.error or "unreadable", record.relpath)
        )

    class_counts = dict(sorted(Counter(r.class_name for r in records).items()))
    num_classes = len(class_counts)
    if spec.configured_classes is not None and num_classes != spec.configured_classes:
        issues.append(
            Issue(
                spec.key,
                "class_count_mismatch",
                f"discovered {num_classes} classes; config expects {spec.configured_classes}",
            )
        )

    readable = [r for r in records if r.ok]
    width = Stats.of(r.width for r in readable if r.width is not None)
    height = Stats.of(r.height for r in readable if r.height is not None)
    aspect = Stats.of(r.aspect_ratio for r in readable if r.aspect_ratio is not None)

    imbalance: float | None = None
    if class_counts:
        counts = list(class_counts.values())
        smallest = min(counts)
        imbalance = round(max(counts) / smallest, 2) if smallest else None
        if imbalance is not None and imbalance > settings.imbalance_warn_ratio:
            issues.append(Issue(spec.key, "class_imbalance", f"max/min class ratio = {imbalance}"))

    exact = duplicates.exact_duplicate_groups(records)
    near = duplicates.near_duplicate_groups(records) if settings.near_duplicates else []

    dataset_report = DatasetReport(
        key=spec.key,
        name=spec.name,
        root=str(spec.root),
        exists=True,
        num_images=len(records),
        num_ok=len(readable),
        num_corrupt=len(corrupt),
        num_unsupported=len(unsupported),
        num_classes=num_classes,
        configured_classes=spec.configured_classes,
        class_counts=class_counts,
        width=width,
        height=height,
        aspect_ratio=aspect,
        imbalance_ratio=imbalance,
        num_exact_duplicate_groups=len(exact),
        num_near_duplicate_groups=len(near),
        dataset_checksum=_dataset_checksum(records),
        issues=issues,
    )
    return dataset_report, records, exact, near


def run_audit(
    specs: list[DatasetSpec],
    settings: AuditConfig,
    reports_dir: str | Path,
    *,
    plantdx_version: str = "0",
    config_hash: str = "",
    logger: Logger | None = None,
) -> AuditManifest:
    """Audit all ``specs`` and write the reports; return the manifest.

    Args:
        specs: Datasets to audit (from :func:`plantdx.audit.discovery.build_specs`).
        settings: Audit configuration (from ``config.audit``).
        reports_dir: Directory to write reports into (created if missing).
        plantdx_version: Recorded in the manifest.
        config_hash: Recorded in the manifest.
        logger: Optional logger; if omitted, one is configured that also writes
            ``<reports_dir>/audit.log``.
    """
    reports = ensure_dir(reports_dir)
    if logger is None:
        configure_logging(log_file=reports / "audit.log")
        logger = get_logger("audit")

    logger.info("PlantDx dataset audit starting (%d dataset(s))", len(specs))

    all_records: list[ImageRecord] = []
    dataset_reports: list[DatasetReport] = []
    exact_per_dataset: list[tuple[str, list[tuple[str, list[str]]]]] = []
    near_per_dataset: list[tuple[str, list[tuple[str, list[str]]]]] = []
    checksums: dict[str, str] = {}

    for spec in specs:
        dataset_report, records, exact, near = _audit_dataset(spec, settings, logger)
        dataset_reports.append(dataset_report)
        all_records.extend(records)
        exact_per_dataset.append((spec.key, exact))
        near_per_dataset.append((spec.key, near))
        checksums[spec.key] = dataset_report.dataset_checksum
        report.write_dataset_summary_json(reports / f"{spec.key}_summary.json", dataset_report)
        logger.info(
            "[%s] %d images · %d classes · %d corrupt · %d exact-dup groups · checksum=%s",
            spec.key,
            dataset_report.num_images,
            dataset_report.num_classes,
            dataset_report.num_corrupt,
            dataset_report.num_exact_duplicate_groups,
            dataset_report.dataset_checksum[:12],
        )

    report.write_class_distribution_csv(reports / "class_distribution.csv", dataset_reports)
    report.write_image_statistics_csv(reports / "image_statistics.csv", all_records)
    report.write_duplicates_csv(reports / "duplicate_images.csv", exact_per_dataset)
    report.write_corrupt_csv(reports / "corrupt_images.csv", all_records)
    if settings.near_duplicates:
        report.write_near_duplicates_csv(reports / "near_duplicate_images.csv", near_per_dataset)

    audit_checksum = sha256_hex(*[checksums[key] for key in sorted(checksums)])
    totals = {
        "images": sum(r.num_images for r in dataset_reports),
        "ok": sum(r.num_ok for r in dataset_reports),
        "corrupt": sum(r.num_corrupt for r in dataset_reports),
        "unsupported": sum(r.num_unsupported for r in dataset_reports),
        "classes": sum(r.num_classes for r in dataset_reports),
        "exact_duplicate_groups": sum(r.num_exact_duplicate_groups for r in dataset_reports),
        "near_duplicate_groups": sum(r.num_near_duplicate_groups for r in dataset_reports),
    }
    manifest = AuditManifest(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        plantdx_version=plantdx_version,
        config_hash=config_hash,
        tool="plantdx.audit",
        settings=settings.model_dump(),
        datasets=checksums,
        audit_checksum=audit_checksum,
        totals=totals,
        splits=_SPLIT_STATUS,
    )
    report.write_audit_manifest_json(reports / "audit_manifest.json", manifest)
    report.write_dataset_card_md(reports / "dataset_card.md", dataset_reports, manifest)

    logger.info("Audit complete. Reports in %s (audit_checksum=%s)", reports, audit_checksum[:12])
    return manifest
