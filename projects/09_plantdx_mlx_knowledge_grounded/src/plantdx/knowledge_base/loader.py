"""Disease Knowledge Base loader and integrity checker (interface).

Responsibilities (Milestone 2):
    * parse ``knowledge_base/dkb.json`` into :class:`KnowledgeBase`;
    * compute and record ``dkb_sha256`` (pins the source of truth in provenance);
    * run the DKB self-checks that the caption framework depends on
      (e.g., no ``never_appear`` term also appears in a ``recommended_*`` list;
      every non-healthy disease has diagnostic terminology) — see doc 01 §8.

The loader never mutates the DKB; it is a read-only projection.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.knowledge_base.models import KnowledgeBase


class DKBLoader:
    """Loads and validates the FINAL Disease Knowledge Base.

    Args:
        dkb_path: Path to ``dkb.json``.
    """

    def __init__(self, dkb_path: str | Path) -> None:
        """Initialize the loader with the path to dkb.json."""
        self.dkb_path = Path(dkb_path)

    def load(self) -> KnowledgeBase:
        """Parse and return the knowledge base.

        Raises:
            plantdx.core.exceptions.KnowledgeBaseError: If missing or malformed.
        """
        raise NotImplementedError("Milestone 2: DKB parsing")

    def verify(self, kb: KnowledgeBase) -> None:
        """Run DKB integrity self-checks (doc 01 §8).

        Raises:
            plantdx.core.exceptions.KnowledgeBaseError: On any consistency failure.
        """
        raise NotImplementedError("Milestone 2: DKB integrity checks")
