"""QA review records and loader (doc 05 §3, §7).

Models the per-caption reviewer checklist verdicts and loads independent
per-reviewer result files. The review UI itself is a separate local tool; this
module owns the data contract it reads/writes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from plantdx.core.enums import DefectClass


@dataclass(frozen=True, slots=True)
class ChecklistVerdict:
    """One reviewer's verdict on one caption (doc 05 §3)."""

    caption_id: str
    reviewer_id: str
    accepted: bool
    critical_items: dict[str, bool]  # A1..A6, C10
    major_items: dict[str, bool]  # B7..B9, C11
    minor_items: dict[str, bool]  # D12..D13
    notes: str | None = None

    def worst_defect(self) -> DefectClass | None:
        """Return the most severe defect class present, or None if clean."""
        raise NotImplementedError("Milestone 4: defect triage")


class ReviewStore:
    """Loads and aggregates per-reviewer verdict files."""

    def __init__(self, results_dir: str | Path) -> None:
        """Initialize the store with the reviewer results directory."""
        self.results_dir = Path(results_dir)

    def load(self) -> Iterable[ChecklistVerdict]:
        """Yield all reviewer verdicts across all reviewers."""
        raise NotImplementedError("Milestone 4: review result loading")
