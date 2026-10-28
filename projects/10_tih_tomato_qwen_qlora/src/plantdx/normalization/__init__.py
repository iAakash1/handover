"""Dataset Normalization Engine (Milestone 2.1).

Copies the configured tomato classes from the immutable raw dataset into the canonical datasets/ structure.
"""

from __future__ import annotations

from plantdx.normalization.engine import normalize_crop, run_normalization
from plantdx.normalization.models import CropReport, NormalizedImage

__all__ = ["CropReport", "NormalizedImage", "normalize_crop", "run_normalization"]
