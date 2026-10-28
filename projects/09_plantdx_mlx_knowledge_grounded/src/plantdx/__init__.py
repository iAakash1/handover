"""PlantDx: a knowledge-grounded framework for agricultural VLM instruction-tuning datasets.

Captions are generated from a curated, cited Disease Knowledge Base and a
controlled vocabulary via ontology-driven templates and multi-stage validation —
never from a VLM/LLM prediction and never from image analysis. See
``caption_framework/`` for the design specification (the source of truth).

Public API (top level): the version and the config loader. Component classes are
imported from their subpackages, e.g. ``from plantdx.ontology import OntologyBuilder``.
"""

from __future__ import annotations

from plantdx.__about__ import __version__
from plantdx.config import PlantDxConfig, load_config

__all__ = ["PlantDxConfig", "__version__", "load_config"]
