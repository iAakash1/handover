"""Generation package: components D (selector), E (templates), F (realizer), engine.

Model/record types live in :mod:`plantdx.generation.models`.
"""

from __future__ import annotations

from plantdx.generation.engine import CaptionEngine
from plantdx.generation.planner import CaptionBudgetPlanner
from plantdx.generation.realizer import SlotRealizer
from plantdx.generation.selector import ConceptSelector
from plantdx.generation.templates import TemplateLibrary

__all__ = [
    "CaptionBudgetPlanner",
    "CaptionEngine",
    "ConceptSelector",
    "SlotRealizer",
    "TemplateLibrary",
]
