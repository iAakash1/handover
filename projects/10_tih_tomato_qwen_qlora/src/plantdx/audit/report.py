"""Writers for the audit report files (CSV, JSON, Markdown).

All output is plain CSV/JSON/Markdown (no custom formats). Rows are written in a
deterministic order so reports diff cleanly between runs.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from plantdx.audit.models import AuditManifest, DatasetReport, ImageRecord
from plantdx.utils.io import write_json

# A per-dataset group list: (dataset_key, [(hash, [relpaths]), ...]).
GroupsPerDataset = Sequence[tuple[str, Sequence[tuple[str, Sequence[str]]]]]

# Cap issues printed into the human-readable card; the CSVs hold the full list.
_MAX_ISSUES_IN_CARD = 50


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_image_statistics_csv(path: Path, records: Sequence[ImageRecord]) -> None:
    """One row per image: dimensions, mode, format, size, aspect ratio, hash."""
    header = [
        "dataset",
        "class",
        "relpath",
        "ok",
        "width",
        "height",
        "mode",
        "format",
        "file_size",
        "aspect_ratio",
        "sha256",
        "error",
    ]
    rows: list[list[object]] = []
    for r in sorted(records, key=lambda x: (x.dataset, x.relpath)):
        rows.append(
            [
                r.dataset,
                r.class_name,
                r.relpath,
                r.ok,
                r.width or "",
                r.height or "",
                r.mode or "",
                r.format or "",
                r.file_size,
                "" if r.aspect_ratio is None else r.aspect_ratio,
                r.sha256,
                r.error or "",
            ]
        )
    _write_csv(path, header, rows)


def write_class_distribution_csv(path: Path, reports: Sequence[DatasetReport]) -> None:
    """One row per (dataset, class) with its image count."""
    header = ["dataset", "class", "image_count"]
    rows: list[list[object]] = []
    for report in reports:
        for class_name in sorted(report.class_counts):
            rows.append([report.key, class_name, report.class_counts[class_name]])
    _write_csv(path, header, rows)


def write_corrupt_csv(path: Path, records: Sequence[ImageRecord]) -> None:
    """One row per corrupt/unreadable image."""
    header = ["dataset", "class", "relpath", "error", "sha256"]
    rows: list[list[object]] = [
        [r.dataset, r.class_name, r.relpath, r.error or "", r.sha256]
        for r in sorted(records, key=lambda x: (x.dataset, x.relpath))
        if not r.ok
    ]
    _write_csv(path, header, rows)


def _write_group_csv(path: Path, per_dataset: GroupsPerDataset, hash_column: str) -> None:
    header = ["dataset", "group", hash_column, "relpath"]
    rows: list[list[object]] = []
    group_id = 0
    for dataset_key, groups in per_dataset:
        for digest, paths in groups:
            group_id += 1
            for relpath in paths:
                rows.append([dataset_key, group_id, digest, relpath])
    _write_csv(path, header, rows)


def write_duplicates_csv(path: Path, per_dataset: GroupsPerDataset) -> None:
    """Exact-duplicate groups (grouped by SHA-256)."""
    _write_group_csv(path, per_dataset, "sha256")


def write_near_duplicates_csv(path: Path, per_dataset: GroupsPerDataset) -> None:
    """Near-duplicate groups (grouped by average hash)."""
    _write_group_csv(path, per_dataset, "ahash")


def write_dataset_summary_json(path: Path, report: DatasetReport) -> None:
    """Write one dataset's full summary as JSON."""
    write_json(path, asdict(report))


def write_audit_manifest_json(path: Path, manifest: AuditManifest) -> None:
    """Write the top-level audit manifest as JSON."""
    write_json(path, asdict(manifest))


def _stats_line(label: str, stats: object) -> str:
    if stats is None:
        return f"- **{label}:** n/a"
    return f"- **{label}:** min={stats.min}, max={stats.max}, mean={stats.mean}"  # type: ignore[attr-defined]


def write_dataset_card_md(
    path: Path, reports: Sequence[DatasetReport], manifest: AuditManifest
) -> None:
    """Write a human-readable dataset card summarizing the audit."""
    lines: list[str] = []
    lines.append("# PlantDx Dataset Audit — Dataset Card\n")
    lines.append(f"- Generated: `{manifest.generated_at}`")
    lines.append(f"- Tool: `{manifest.tool}` (plantdx {manifest.plantdx_version})")
    lines.append(f"- Config hash: `{manifest.config_hash}`")
    lines.append(f"- Audit checksum: `{manifest.audit_checksum}`")
    lines.append("")
    totals = manifest.totals
    lines.append("## Totals\n")
    lines.append(
        f"- Images: **{totals['images']}** "
        f"(ok: {totals['ok']}, corrupt: {totals['corrupt']}, "
        f"unsupported: {totals['unsupported']})"
    )
    lines.append(f"- Classes (across datasets): {totals['classes']}")
    lines.append(
        f"- Exact-duplicate groups: {totals['exact_duplicate_groups']} · "
        f"near-duplicate groups: {totals['near_duplicate_groups']}"
    )
    lines.append(f"- Split status: {manifest.splits['status']} — {manifest.splits['note']}")
    lines.append("")

    for report in reports:
        lines.append(f"## {report.key} — {report.name}\n")
        lines.append(f"- Root: `{report.root}` (exists: {report.exists})")
        lines.append(
            f"- Images: {report.num_images} (ok: {report.num_ok}, "
            f"corrupt: {report.num_corrupt}, unsupported: {report.num_unsupported})"
        )
        configured = "n/a" if report.configured_classes is None else report.configured_classes
        lines.append(f"- Classes discovered: {report.num_classes} (config expects: {configured})")
        lines.append(_stats_line("Width (px)", report.width))
        lines.append(_stats_line("Height (px)", report.height))
        lines.append(_stats_line("Aspect ratio", report.aspect_ratio))
        lines.append(f"- Class imbalance (max/min): {report.imbalance_ratio}")
        lines.append(f"- Dataset checksum: `{report.dataset_checksum}`")
        if report.class_counts:
            lines.append("\n| class | images |")
            lines.append("|-------|--------|")
            for class_name in sorted(report.class_counts):
                lines.append(f"| {class_name} | {report.class_counts[class_name]} |")
        if report.issues:
            lines.append(f"\n**Issues ({len(report.issues)}):** (full detail in the CSV reports)")
            for issue in report.issues[:_MAX_ISSUES_IN_CARD]:
                where = f" (`{issue.path}`)" if issue.path else ""
                lines.append(f"- `{issue.kind}`: {issue.detail}{where}")
            if len(report.issues) > _MAX_ISSUES_IN_CARD:
                lines.append(f"- … and {len(report.issues) - _MAX_ISSUES_IN_CARD} more")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
