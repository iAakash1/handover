"""Dataset Audit Engine (Milestone 2).

CPU-only. Inspects every image in the configured datasets and produces a
reproducibility report (counts, class distribution, dimensions, duplicates,
corrupt images, checksums, and a manifest). Entry point: :func:`run_audit`.
"""

from __future__ import annotations

from plantdx.audit.discovery import DatasetSpec, build_specs
from plantdx.audit.engine import run_audit
from plantdx.audit.models import AuditManifest, DatasetReport, ImageRecord

__all__ = [
    "AuditManifest",
    "DatasetReport",
    "DatasetSpec",
    "ImageRecord",
    "build_specs",
    "run_audit",
]
