"""QA audit sampling (doc 05 §2).

Builds seeded, stratified acceptance-audit manifests: ≥100 captions per disease,
stratified by style / task_type / register / hedged, with guaranteed minimum
cells for the higher-risk strata and 100% inclusion of the diagnostic split and
differential captions for the hardest confusable pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from plantdx.core.types import CaptionRecord


@dataclass(frozen=True, slots=True)
class AuditManifest:
    """A reproducible list of caption ids to review, with the draw seed."""

    library_version: str
    seed: int
    caption_ids: tuple[str, ...]


class AuditSampler:
    """Draws stratified acceptance-audit samples.

    Args:
        per_disease: Minimum captions to review per disease (default 100).
        seed: Sampling seed (recorded in the manifest for reproducibility).
    """

    def __init__(self, per_disease: int = 100, seed: int = 20260711) -> None:
        """Initialize the sampler with the per-disease minimum and draw seed."""
        self.per_disease = per_disease
        self.seed = seed

    def build_manifest(
        self,
        records: Sequence[CaptionRecord],
        library_version: str,
        out_path: str | Path,
    ) -> AuditManifest:
        """Draw the audit sample and write the manifest."""
        raise NotImplementedError("Milestone 4: stratified audit sampling")
