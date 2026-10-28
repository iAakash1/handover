"""Diversity Controller — part of component (H) (doc 00 §7.3).

Enforces anti-domination caps (max template share, max skeleton share, max
opening-trigram share) during generation and exposes the running usage counts
the template selector consults.
"""

from __future__ import annotations

from plantdx.config.schema import AntiDomination


class DiversityController:
    """Tracks usage and enforces anti-domination caps per disease.

    Args:
        caps: Anti-domination thresholds from config.
    """

    def __init__(self, caps: AntiDomination) -> None:
        """Initialize the controller with the anti-domination thresholds."""
        self.caps = caps

    def template_share_ok(self, disease_id: str, template_id: str) -> bool:
        """Return True if using ``template_id`` keeps it under the share cap."""
        raise NotImplementedError("Milestone 3: template-share tracking")

    def skeleton_share_ok(self, disease_id: str, skeleton_hash: str) -> bool:
        """Return True if the caption skeleton is under its share cap."""
        raise NotImplementedError("Milestone 3: skeleton-share tracking")

    def opening_trigram_ok(self, disease_id: str, trigram: str) -> bool:
        """Return True if the caption-initial trigram is under its share cap."""
        raise NotImplementedError("Milestone 3: opening-trigram balancing")

    def record(self, disease_id: str, template_id: str, skeleton_hash: str, trigram: str) -> None:
        """Update running counts after a caption is accepted."""
        raise NotImplementedError("Milestone 3: usage bookkeeping")
