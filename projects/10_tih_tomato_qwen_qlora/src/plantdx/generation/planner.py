"""Caption Budget Planner (doc 00 §7.4, §7.6).

Decides, per image, how many captions to generate (``K``) and their
``(style, length, register, task_type, information_level)`` mix, drawn from the
global target distributions and jittered per image (seeded). Applies the
per-class automatic adjustments (doc 02 §6): low-concept classes skew short;
confusable classes get more educational captions.
"""

from __future__ import annotations

from plantdx.config.schema import GenerationConfig
from plantdx.generation.models import BudgetPlan
from plantdx.ontology.models import DiseaseOntology


class CaptionBudgetPlanner:
    """Plans the per-image caption budget and style mix.

    Args:
        config: The generation configuration (distributions, K, balance mode).
    """

    def __init__(self, config: GenerationConfig) -> None:
        """Initialize the planner with the generation configuration."""
        self.config = config

    def plan(self, image_id: str, ontology: DiseaseOntology, base_seed: str) -> BudgetPlan:
        """Return the budget plan for one image."""
        raise NotImplementedError("Milestone 3: per-image budget planning")
