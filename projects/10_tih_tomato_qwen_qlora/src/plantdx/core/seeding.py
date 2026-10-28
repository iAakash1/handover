"""Deterministic seed derivation.

Implements the reproducibility model of ``caption_framework/00_methodology_overview.md``
§6: one ``global_seed`` fans out to per-image, per-caption, and per-attempt seeds
by hashing, so the entire corpus is regenerable bit-for-bit.

These are pure module-level functions (no state); Milestone 3 fills in the SHA-256
bodies. Behavior is deterministic: identical inputs always yield identical digests.
"""

from __future__ import annotations


def image_seed(global_seed: int, image_id: str) -> str:
    """Per-image seed ``SHA256(global_seed ‖ image_id)`` as a hex digest."""
    raise NotImplementedError("Milestone 3: deterministic seed derivation")


def caption_seed(base_seed: str, index: int) -> str:
    """Per-caption seed ``SHA256(base_seed ‖ index)`` as a hex digest."""
    raise NotImplementedError("Milestone 3: deterministic seed derivation")


def attempt_seed(parent_seed: str, attempt: int) -> str:
    """Per-attempt seed folding the regeneration attempt into ``parent_seed``."""
    raise NotImplementedError("Milestone 3: deterministic seed derivation")
