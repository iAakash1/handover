"""Deterministic, image-grouped, disease-stratified train/val/test assignment.

Splitting is at the *image* level: every training row derived from one image
lands in the same split, so no image leaks across splits. Assignment is a pure
function of ``(split_seed, image_id)`` via a SHA-256 hash mapped to [0, 1), then
bucketed per disease by the configured ratios — no RNG state, no shuffling, no
dependence on discovery order.
"""

from __future__ import annotations

from plantdx.training.data.discovery import ImageItem
from plantdx.utils.hashing import sha256_hex

_DENOM = float(1 << 32)


def _unit_interval(seed: int, image_id: str) -> float:
    """Map (seed, image_id) deterministically into [0, 1)."""
    digest = sha256_hex(str(seed), image_id)
    return int(digest[:8], 16) / _DENOM


def assign_splits(
    images: list[ImageItem],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    """Return ``{image_id: split}`` with split in {"train","validation","test"}.

    Stratified by disease: each disease's images are ordered by their hashed
    position and cut at the ratio boundaries, so class proportions are preserved
    in every split even for small classes.
    """
    _ = train_ratio + val_ratio + test_ratio  # validated upstream to sum to 1.0
    by_disease: dict[str, list[ImageItem]] = {}
    for item in images:
        by_disease.setdefault(item.disease_id, []).append(item)

    assignment: dict[str, str] = {}
    for disease in sorted(by_disease):
        ordered = sorted(
            by_disease[disease],
            key=lambda it: (_unit_interval(seed, it.image_id), it.image_id),
        )
        n = len(ordered)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        for idx, item in enumerate(ordered):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "validation"
            else:
                split = "test"
            assignment[item.image_id] = split
    return assignment
