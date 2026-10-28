"""Writers for normalization outputs (class mapping, manifest, card, run report)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from plantdx.config.schema import SourceSpec
from plantdx.normalization.models import CropReport, NormalizedImage
from plantdx.utils.io import write_json


def write_class_mapping_json(path: Path, source: SourceSpec, report: CropReport) -> None:
    """Write ``original folder -> normalized class`` plus the ignored folders."""
    write_json(
        path,
        {
            "crop": report.crop,
            "dataset": report.dataset,
            "mapping": dict(sorted(source.class_map.items())),
            "included_classes": sorted(report.class_counts),
            "ignored_folders": report.ignored_folders,
        },
    )


def write_manifest_json(path: Path, report: CropReport, images: Sequence[NormalizedImage]) -> None:
    """Write one row per image: source path, normalized path, checksum, class, split."""
    write_json(
        path,
        {
            "crop": report.crop,
            "dataset": report.dataset,
            "count": len(images),
            "dataset_checksum": report.dataset_checksum,
            "images": [asdict(image) for image in images],
        },
    )


def write_dataset_card_md(path: Path, report: CropReport) -> None:
    """Write the human-readable dataset card."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# {report.crop} — normalized dataset card\n",
        f"- **Source dataset:** {report.dataset}",
        f"- **License:** {report.license}",
        f"- **Original citation:** {report.citation}",
        f"- **Download URL:** {report.url}",
        f"- **Normalization timestamp:** {timestamp}",
        f"- **Mode:** {report.mode}",
        f"- **Image count:** {report.image_count}",
        f"- **Class count:** {report.class_count}",
        f"- **Dataset checksum (SHA-256):** `{report.dataset_checksum}`",
        f"- **Source layout detected:** {report.layout}",
        f"- **Processed directory:** `{report.processed_dir}`",
        "",
        "## Class distribution\n",
        "| class | images |",
        "|-------|--------|",
    ]
    for class_name in sorted(report.class_counts):
        lines.append(f"| {class_name} | {report.class_counts[class_name]} |")

    lines += [
        "",
        "## Known limitations\n",
        f"- Extracted from the full {report.dataset} dataset; only the mapped classes "
        "are included (other crops are ignored, never deleted).",
        "- Raw datasets are immutable; this is a copy/link into `processed/`.",
        "- Any `train`/`val` folders are merged into single class directories; the split "
        "of each image is preserved in `manifest.json`.",
        "- Filenames are preserved except on a genuine collision, where the split is "
        f"prefixed ({len(report.disambiguated)} file(s) affected).",
        f"- Duplicate files skipped: {len(report.duplicates_skipped)}; "
        f"checksum verification failures: {len(report.checksum_failures)}.",
        f"- Ignored (non-target) folders: {len(report.ignored_folders)}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_crop_outputs(
    crop_dir: Path, source: SourceSpec, report: CropReport, images: Sequence[NormalizedImage]
) -> None:
    """Write class_mapping.json, manifest.json and dataset_card.md for one crop."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    write_class_mapping_json(crop_dir / "class_mapping.json", source, report)
    write_manifest_json(crop_dir / "manifest.json", report, images)
    write_dataset_card_md(crop_dir / "dataset_card.md", report)


def write_run_report(
    path: Path, reports: dict[str, CropReport], *, plantdx_version: str, config_hash: str, mode: str
) -> None:
    """Write the combined normalization run report."""
    write_json(
        path,
        {
            "tool": "plantdx.normalize",
            "plantdx_version": plantdx_version,
            "config_hash": config_hash,
            "normalized_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "totals": {
                "images": sum(r.image_count for r in reports.values()),
                "classes": sum(r.class_count for r in reports.values()),
            },
            "crops": {crop: asdict(report) for crop, report in reports.items()},
        },
    )
