"""Core value objects for PlantDx.

These frozen dataclasses are the data contract exchanged between pipeline
components. They are concrete (a dataclass definition *is* the interface). No
behavior/algorithms live here — only structure and typing.

The :class:`CaptionRecord` mirrors the canonical schema in
``caption_framework/04_dataset_schema_spec.md`` §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plantdx.core.enums import (
    AgentCategory,
    ConceptId,
    Crop,
    ExpansionEdgeType,
    LengthBand,
    Register,
    SignType,
    Split,
    Style,
    TaskType,
    Verdict,
)

# --------------------------------------------------------------------------- #
# Image / label
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ImageRef:
    """A reference to a source image (never the pixels — grounding is label-only)."""

    id: str
    path: str
    dataset: str
    crop: Crop


@dataclass(frozen=True, slots=True)
class DiseaseLabel:
    """Ground-truth label for an image, resolved from its dataset folder."""

    disease_id: str
    class_label: str
    is_pathogen_disease: bool
    agent_category: AgentCategory


# --------------------------------------------------------------------------- #
# Concepts / realizations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConceptRealization:
    """A caption-ready phrase realizing a concept, with DKB provenance."""

    concept_id: ConceptId
    phrase: str
    source_field: str
    sign_type: SignType | None = None


@dataclass(frozen=True, slots=True)
class SelectedConcepts:
    """The concept set chosen for one caption, with their realizations."""

    concept_ids: tuple[ConceptId, ...]
    realizations: tuple[ConceptRealization, ...]


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Per-caption request: what style/register/task and how much information."""

    style: Style
    length_band: LengthBand
    register: Register
    task_type: TaskType
    hedged: bool = False


# --------------------------------------------------------------------------- #
# Provenance (reproducibility; doc 00 §6)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ExpansionEdge:
    """One meaning-preserving vocabulary-expansion step (doc 01 §7.5)."""

    type: ExpansionEdgeType
    value: str
    source_field: str


@dataclass(frozen=True, slots=True)
class VocabChoice:
    """A slot's chosen surface form and its DKB source field."""

    slot: str
    surface: str
    source_field: str


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to regenerate a caption bit-for-bit and audit it."""

    global_seed: int
    base_seed: str
    caption_seed: str
    template_id: str
    instruction_template_id: str
    expansion_edges: tuple[ExpansionEdge, ...]
    vocab_choices: tuple[VocabChoice, ...]
    fallback: bool
    dkb_sha256: str
    ontology_build_id: str
    template_set_version: str
    vocabulary_version: str
    config_hash: str
    generator_version: str
    created_utc: str


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ValidatorResult:
    """Outcome of a single validator (V1..V12 or a soft check)."""

    validator_id: str
    passed: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Aggregate validation outcome for one caption (doc 03 §5)."""

    verdict: Verdict
    attempts: int
    results: tuple[ValidatorResult, ...]
    soft: tuple[ValidatorResult, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Instruction / response / record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Instruction:
    """The user-turn of a training example (doc 04 §4)."""

    task_type: TaskType
    template_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Response:
    """The assistant-turn (the caption) of a training example."""

    text: str
    style: Style
    length_band: LengthBand
    register: Register
    hedged: bool
    token_count: int


@dataclass(frozen=True, slots=True)
class QAAnnotation:
    """Human-review annotation attached to a record (filled in QA; doc 05)."""

    reviewed: bool = False
    verdict: str | None = None
    reviewer_id: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class CaptionRecord:
    """The canonical, model-agnostic caption record (doc 04 §1).

    One image yields many records (captions x instruction pairings); all share
    ``image.id`` and therefore the same :attr:`split`.
    """

    schema_version: str
    caption_id: str
    image: ImageRef
    label: DiseaseLabel
    instruction: Instruction
    response: Response
    concepts: tuple[ConceptId, ...]
    provenance: Provenance
    split: Split
    qa: QAAnnotation = field(default_factory=QAAnnotation)
