"""Dataset Normalization Engine (Milestone 2.1).

Copies the tomato and mango classes from the immutable raw datasets into one
canonical structure under ``datasets/<crop>/processed/<class>/`` and writes the
mapping, manifest, dataset card, and run report. Entry point: :func:`run_normalization`.
"""

from __future__ import annotations

from plantdx.normalization.engine import normalize_crop, run_normalization
from plantdx.normalization.models import CropReport, NormalizedImage

__all__ = ["CropReport", "NormalizedImage", "normalize_crop", "run_normalization"]
