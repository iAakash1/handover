"""Split Builder (doc 04 §5).

Builds image-grouped, disease-stratified train/val/test splits plus a held-out
``diagnostic`` split of the hardest confusable pairs. Grouping by image prevents
caption-level leakage; splits are seeded and versioned so all four models train
and evaluate on identical partitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from plantdx.core.enums import Split
from plantdx.core.types import ImageRef


class SplitBuilder:
    """Assigns images to splits and writes the split id files.

    Args:
        train: Train fraction (by image).
        val: Validation fraction.
        test: Test fraction.
        seed: Split RNG seed.
        splits_dir: Output directory for ``*_image_ids.txt``.
    """

    def __init__(
        self,
        train: float,
        val: float,
        test: float,
        seed: int,
        splits_dir: str | Path,
    ) -> None:
        """Initialize the split builder with the split ratios, seed, and output dir."""
        self.train = train
        self.val = val
        self.test = test
        self.seed = seed
        self.splits_dir = Path(splits_dir)

    def build(
        self,
        images: Sequence[tuple[ImageRef, str]],
        confusable_groups: Sequence[tuple[str, ...]],
    ) -> dict[str, Split]:
        """Assign each image id to a split and write the split files."""
        raise NotImplementedError("Milestone 4: stratified image-grouped splitting")
