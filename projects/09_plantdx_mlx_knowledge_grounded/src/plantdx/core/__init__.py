"""Core value objects, enums, exceptions, and seed helpers shared pipeline-wide.

``core`` is a leaf package: it depends only on the standard library. Everything
here is concrete API surface; no pipeline algorithms live in this package.

Public entry points re-exported here are the enums, the exception hierarchy, and
the deterministic seed functions. Value objects (``CaptionRecord``, ``Provenance``,
…) are imported from :mod:`plantdx.core.types` where they are defined.
"""

from __future__ import annotations

from plantdx.core.enums import (
    AgentCategory,
    ConceptId,
    Crop,
    DefectClass,
    ExpansionEdgeType,
    InformationLevel,
    LengthBand,
    Register,
    SignType,
    Split,
    Style,
    TargetModel,
    TaskType,
    Verdict,
)
from plantdx.core.exceptions import (
    ConfigError,
    ConversionError,
    DerivationError,
    DiversityGateError,
    GenerationError,
    InvariantViolation,
    KnowledgeBaseError,
    PlantDxError,
    ReproducibilityError,
    ValidationHardError,
)
from plantdx.core.seeding import attempt_seed, caption_seed, image_seed

__all__ = [  # noqa: RUF022 - grouped by source (enums / exceptions / seed helpers), not flat-sorted
    # enums
    "AgentCategory",
    "ConceptId",
    "Crop",
    "DefectClass",
    "ExpansionEdgeType",
    "InformationLevel",
    "LengthBand",
    "Register",
    "SignType",
    "Split",
    "Style",
    "TargetModel",
    "TaskType",
    "Verdict",
    # exceptions
    "ConfigError",
    "ConversionError",
    "DerivationError",
    "DiversityGateError",
    "GenerationError",
    "InvariantViolation",
    "KnowledgeBaseError",
    "PlantDxError",
    "ReproducibilityError",
    "ValidationHardError",
    # seed helpers
    "attempt_seed",
    "caption_seed",
    "image_seed",
]
