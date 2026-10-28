"""De-duplicator — part of component (H) (doc 00 §7.5).

Two-level duplicate suppression:
    * exact: normalized-string hash set (per disease);
    * near-duplicate: MinHash over token shingles with a Jaccard threshold,
      rejecting candidates too similar to an accepted caption of the SAME image.
"""

from __future__ import annotations


class Deduplicator:
    """Tracks emitted captions and rejects (near-)duplicates.

    Args:
        jaccard_threshold: Near-duplicate rejection threshold (within an image).
        num_perm: MinHash permutations.
        shingle_size: Token-shingle size.
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.90,
        num_perm: int = 128,
        shingle_size: int = 5,
    ) -> None:
        """Initialize the deduplicator with the near-duplicate detection thresholds."""
        self.jaccard_threshold = jaccard_threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size

    def is_duplicate(self, caption: str, disease_id: str, image_id: str) -> bool:
        """Return True if ``caption`` duplicates an accepted one (exact or near)."""
        raise NotImplementedError("Milestone 3: exact + MinHash de-duplication")

    def register(self, caption: str, disease_id: str, image_id: str) -> None:
        """Record an accepted caption so later candidates can be compared."""
        raise NotImplementedError("Milestone 3: dedup index registration")
