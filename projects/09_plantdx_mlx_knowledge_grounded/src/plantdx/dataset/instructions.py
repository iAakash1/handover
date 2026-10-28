"""Instruction bank + task/response pairing constraints (doc 04 §4).

Loads the instruction templates, exposes valid ``(task_type, style)`` pairings
per disease, and provides the response-constraint mask a task imposes on concept
selection (e.g., ``color_qa`` requires only the ``lesion_color`` concept).
"""

from __future__ import annotations

from pathlib import Path

from plantdx.core.enums import ConceptId, TaskType
from plantdx.generation.models import InstructionTemplate
from plantdx.ontology.models import DiseaseOntology


class InstructionBank:
    """Holds instruction templates and their response constraints.

    Args:
        instructions_path: Path to ``instructions.json``.
    """

    def __init__(self, instructions_path: str | Path) -> None:
        """Initialize the instruction bank with the path to instructions.json."""
        self.instructions_path = Path(instructions_path)

    def load(self) -> None:
        """Parse the instruction bank."""
        raise NotImplementedError("Milestone 4: instruction bank loading")

    def valid_tasks_for(self, ontology: DiseaseOntology) -> tuple[TaskType, ...]:
        """Return task types whose required concepts the disease supports (doc 04 §4.2)."""
        raise NotImplementedError("Milestone 4: task validity pruning")

    def response_mask(
        self, task_type: TaskType
    ) -> tuple[frozenset[ConceptId], frozenset[ConceptId]]:
        """Return ``(required_concepts, allowed_concepts)`` for a task (doc 04 §4.2)."""
        raise NotImplementedError("Milestone 4: response constraint masks")

    def get(self, task_type: TaskType, seed: str) -> InstructionTemplate:
        """Select an instruction paraphrase for a task type (anti-domination)."""
        raise NotImplementedError("Milestone 4: instruction selection")
