"""Exception hierarchy for PlantDx.

All PlantDx errors derive from :class:`PlantDxError` so callers can catch the
whole family. Specific subclasses map to the failure modes named in the
specification (invariant breaches, derivation faults, validation hard-errors).
"""

from __future__ import annotations


class PlantDxError(Exception):
    """Base class for all PlantDx errors."""


class ConfigError(PlantDxError):
    """Invalid or inconsistent configuration."""


class KnowledgeBaseError(PlantDxError):
    """The Disease Knowledge Base is missing, malformed, or internally inconsistent."""


class DerivationError(PlantDxError):
    """A DKB→ontology/vocabulary derivation rule failed (doc 01 §8).

    Indicates a bug in derivation or a contradiction in the DKB — not a data issue.
    """


class InvariantViolation(PlantDxError):  # noqa: N818 - established public name; not renamed
    """A design invariant was violated (see ``caption_framework/README.md``).

    Raised when, e.g., a term not traceable to the DKB would enter a caption, or a
    non-observable concept is asserted in a visual register.
    """


class GenerationError(PlantDxError):
    """The caption generation engine could not produce a candidate."""


class ValidationHardError(PlantDxError):
    """Even the minimal fallback caption failed validation (doc 03 §4).

    Attributed to an ontology/lexicon fault; the correct response is to fix the
    knowledge base or derivation, not to relax validation.
    """

    def __init__(self, disease_id: str, validator_id: str, detail: str) -> None:
        """Record which disease/validator/span triggered the hard error."""
        self.disease_id = disease_id
        self.validator_id = validator_id
        self.detail = detail
        super().__init__(f"Hard validation error for {disease_id!r} at {validator_id}: {detail}")


class DiversityGateError(PlantDxError):
    """A generated corpus failed a hard diversity acceptance gate (doc 00 §7.7)."""


class ConversionError(PlantDxError):
    """A per-model dataset converter produced an invalid training line (doc 04 §6.5)."""


class ReproducibilityError(PlantDxError):
    """Regenerating a record from its provenance did not reproduce it (doc 00 §6)."""
