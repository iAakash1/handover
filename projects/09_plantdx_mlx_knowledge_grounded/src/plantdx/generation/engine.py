"""Caption Engine - orchestrates components D-I (doc 00 §3).

Implements the per-image generation algorithm: plan the budget, then for each
requested caption select concepts, choose a template, realize + expand, validate
(with reseeded regeneration and a guaranteed-valid fallback), de-duplicate, and
emit a canonical record. The engine calls no neural model and never reads pixels.
"""

from __future__ import annotations

from collections.abc import Iterator

from plantdx.core.types import CaptionRecord, DiseaseLabel, ImageRef
from plantdx.dataset.emitter import Emitter
from plantdx.diversity.controller import DiversityController
from plantdx.diversity.deduplicator import Deduplicator
from plantdx.generation.planner import CaptionBudgetPlanner
from plantdx.generation.realizer import SlotRealizer
from plantdx.generation.selector import ConceptSelector
from plantdx.generation.templates import TemplateLibrary
from plantdx.ontology.models import Ontology
from plantdx.validation.battery import ValidatorBattery


class CaptionEngine:
    """Deterministic, image-blind caption generator (doc 00 §2-§3).

    Args:
        ontology: The derived caption ontology.
        selector: Concept selector (D).
        templates: Template library (E).
        realizer: Slot realizer + expander (F).
        validator: The 12-stage validator battery (G).
        deduplicator: Exact + near-duplicate suppressor (H).
        diversity: Anti-domination / coverage controller (H).
        emitter: Canonical record emitter (I).
        planner: Per-image budget planner.
        global_seed: Root seed for deterministic seed derivation (doc 00 §6);
            fanned out via :mod:`plantdx.core.seeding`.
        max_attempts: Regeneration budget before minimal fallback.
    """

    def __init__(
        self,
        ontology: Ontology,
        selector: ConceptSelector,
        templates: TemplateLibrary,
        realizer: SlotRealizer,
        validator: ValidatorBattery,
        deduplicator: Deduplicator,
        diversity: DiversityController,
        emitter: Emitter,
        planner: CaptionBudgetPlanner,
        global_seed: int,
        max_attempts: int = 8,
    ) -> None:
        """Initialize the engine with its D-I components, the seed, and retry budget."""
        self.ontology = ontology
        self.selector = selector
        self.templates = templates
        self.realizer = realizer
        self.validator = validator
        self.deduplicator = deduplicator
        self.diversity = diversity
        self.emitter = emitter
        self.planner = planner
        self.global_seed = global_seed
        self.max_attempts = max_attempts

    def generate_for_image(
        self,
        image: ImageRef,
        label: DiseaseLabel,
    ) -> Iterator[CaptionRecord]:
        """Yield the validated caption records for a single image (doc 00 §3)."""
        raise NotImplementedError("Milestone 3: per-image caption generation loop")

    def generate_corpus(
        self,
        images: Iterator[tuple[ImageRef, DiseaseLabel]],
    ) -> Iterator[CaptionRecord]:
        """Yield validated records for a stream of labeled images."""
        raise NotImplementedError("Milestone 3: corpus generation")
