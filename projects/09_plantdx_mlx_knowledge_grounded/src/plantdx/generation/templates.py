"""Template Library — component (E) (doc 02).

Loads the 52 caption templates and the instruction bank, indexes them by
style/length/register/required-slots, and selects a compatible template for a
caption request with anti-domination sampling (doc 02 §5).
"""

from __future__ import annotations

from pathlib import Path

from plantdx.core.types import SelectedConcepts
from plantdx.generation.models import CaptionRequest, InstructionTemplate, Template


class TemplateLibrary:
    """Holds and selects caption + instruction templates.

    Args:
        templates_path: Path to ``templates.json`` (the 52 caption templates).
        instructions_path: Path to ``instructions.json`` (the instruction bank).
    """

    def __init__(self, templates_path: str | Path, instructions_path: str | Path) -> None:
        """Initialize the library with the caption template and instruction bank paths."""
        self.templates_path = Path(templates_path)
        self.instructions_path = Path(instructions_path)

    def load(self) -> None:
        """Parse and index the template and instruction files."""
        raise NotImplementedError("Milestone 3: template loading and indexing")

    def choose_caption_template(
        self,
        request: CaptionRequest,
        concepts: SelectedConcepts,
        allowed_sign_types: frozenset[str],
        usage_counts: dict[str, int],
        seed: str,
    ) -> Template:
        """Select a compatible caption template (doc 02 §5).

        Guarantees ``required_slots(template) ⊆ concepts`` and applies the
        anti-domination cap on per-disease template share.
        """
        raise NotImplementedError("Milestone 3: caption template selection")

    def choose_instruction(
        self,
        request: CaptionRequest,
        seed: str,
    ) -> InstructionTemplate:
        """Select an instruction (user-turn) template for the request's task type."""
        raise NotImplementedError("Milestone 3: instruction selection")
