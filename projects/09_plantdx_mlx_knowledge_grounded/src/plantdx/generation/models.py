"""Data models for the caption generation engine (doc 00 §3, doc 02)."""

from __future__ import annotations

from dataclasses import dataclass

from plantdx.core.enums import (
    ConceptId,
    InformationLevel,
    LengthBand,
    Register,
    SignType,
    Style,
    TaskType,
)


@dataclass(frozen=True, slots=True)
class Template:
    """A caption template record (doc 02 §2). Carries syntax only, no domain content."""

    id: str
    style: Style
    length_band: LengthBand
    target_tokens: tuple[int, int]
    register: Register
    hedged: bool
    required_slots: tuple[str, ...]
    optional_slots: tuple[str, ...]
    min_concepts: int
    max_concepts: int
    sign_type_allow: tuple[SignType, ...]
    pattern: str
    use_when: str


@dataclass(frozen=True, slots=True)
class InstructionTemplate:
    """An instruction (user-turn) template with its response constraint (doc 04 §4)."""

    id: str
    task_type: TaskType
    text: str
    required_concepts: tuple[ConceptId, ...]
    allowed_concepts: tuple[ConceptId, ...] | None  # None = standard budget


@dataclass(frozen=True, slots=True)
class CaptionRequest:
    """One planned caption slot for an image (produced by the budget planner)."""

    style: Style
    length_band: LengthBand
    register: Register
    task_type: TaskType
    information_level: InformationLevel
    hedged: bool


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """The per-image plan: how many captions and their specs (doc 00 §7.4)."""

    image_id: str
    disease_id: str
    requests: tuple[CaptionRequest, ...]


@dataclass(frozen=True, slots=True)
class CaptionDraft:
    """A realized-but-not-yet-validated caption plus the metadata to trace it."""

    text: str
    request: CaptionRequest
    template_id: str
    instruction_template_id: str
    concept_ids: tuple[ConceptId, ...]
