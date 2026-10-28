"""Diversity package (component H): de-duplication, anti-domination, metrics."""

from __future__ import annotations

from plantdx.diversity.controller import DiversityController
from plantdx.diversity.deduplicator import Deduplicator
from plantdx.diversity.metrics import DiversityEvaluator

__all__ = ["Deduplicator", "DiversityController", "DiversityEvaluator"]
