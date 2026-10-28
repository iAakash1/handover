"""Enumerations shared across the PlantDx pipeline.

These enums are part of the public API and mirror the controlled taxonomies in
the caption framework specification (``caption_framework/``). They are concrete
(not stubs): an enum *is* an interface contract.
"""

from __future__ import annotations

from enum import Enum


class Crop(str, Enum):
    """Supported crops."""

    TOMATO = "tomato"
    MANGO = "mango"


class AgentCategory(str, Enum):
    """Causal-agent category. Mirrors ``dkb.json:agent_category``.

    The ``*_PEST`` and :attr:`SAPROPHYTIC_FUNGUS` members denote classes that are
    **not** tissue infections; captions for them use pest/mechanical/surface
    language (spec invariant #5).
    """

    NONE = "none"
    BACTERIUM = "bacterium"
    FUNGUS = "fungus"
    OOMYCETE = "oomycete"
    VIRUS = "virus"
    ARTHROPOD_PEST = "arthropod_pest"
    INSECT_PEST = "insect_pest"
    SAPROPHYTIC_FUNGUS = "saprophytic_fungus"

    @property
    def is_pathogen(self) -> bool:
        """Whether this category is a tissue-infecting pathogen."""
        return self in {
            AgentCategory.BACTERIUM,
            AgentCategory.FUNGUS,
            AgentCategory.OOMYCETE,
            AgentCategory.VIRUS,
        }


class SignType(str, Enum):
    """Visual sign type of a primary symptom (doc 01 §3.3).

    Drives which descriptive concepts (color/shape/size/texture) are eligible.
    """

    LESION = "lesion"
    COATING = "coating"
    GALL = "gall"
    STIPPLING = "stippling"
    CUT = "cut"
    DEFORMATION = "deformation"
    MOTTLE = "mottle"
    HEALTHY = "healthy"


class ConceptId(str, Enum):
    """The 20 caption concept types (doc 01 §2.1)."""

    DISEASE_IDENTITY = "disease_identity"
    HOST = "host"
    AGENT_REFERENCE = "agent_reference"
    AGENT_CATEGORY_DESCRIPTOR = "agent_category_descriptor"
    PRIMARY_SIGN = "primary_sign"
    LESION_COLOR = "lesion_color"
    LESION_SHAPE = "lesion_shape"
    LESION_SIZE = "lesion_size"
    LESION_DISTRIBUTION = "lesion_distribution"
    LEAF_LOCATION = "leaf_location"
    TEXTURE = "texture"
    CHLOROSIS = "chlorosis"
    NECROSIS = "necrosis"
    LEAF_DEFORMATION = "leaf_deformation"
    SECONDARY_SIGN = "secondary_sign"
    EXTENT = "extent"
    SEVERITY_STAGE = "severity_stage"
    DIFFERENTIAL = "differential"
    HEALTHY_STATE = "healthy_state"
    MANAGEMENT = "management"


class Style(str, Enum):
    """Caption stylistic families (doc 02 §3)."""

    SHORT = "short"
    SINGLE_SENTENCE = "single_sentence"
    TWO_SENTENCE = "two_sentence"
    CLINICAL = "clinical"
    DESCRIPTIVE = "descriptive"
    EDUCATIONAL = "educational"
    DENSE = "dense"
    LONG = "long"


class LengthBand(str, Enum):
    """Target caption length band."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class Register(str, Enum):
    """Caption register (doc 01 §2.1 / doc 02 §2)."""

    VISUAL = "visual"
    CLINICAL = "clinical"
    DESCRIPTIVE = "descriptive"
    EDUCATIONAL = "educational"


class InformationLevel(str, Enum):
    """Concept-budget level controlling caption richness (doc 00 §4)."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DENSE = "dense"


class TaskType(str, Enum):
    """Instruction task types (doc 04 §4)."""

    DESCRIBE = "describe"
    IDENTIFY = "identify"
    SIGNS = "signs"
    COLOR_QA = "color_qa"
    LOCATION_QA = "location_qa"
    CROP_QA = "crop_qa"
    DIFFERENTIAL = "differential"
    HEALTHY_CHECK = "healthy_check"


class ExpansionEdgeType(str, Enum):
    """Typed, meaning-preserving vocabulary-expansion operations (doc 01 §7.3)."""

    SUBST_SYN = "SUBST_syn"
    ADD_COLOR = "ADD_color"
    ADD_SHAPE = "ADD_shape"
    ADD_SIZE = "ADD_size"
    ADD_TEXTURE = "ADD_texture"
    ADD_EXTENT = "ADD_extent"
    ADD_LOCATION = "ADD_location"
    REORDER_LIST = "REORDER_list"


class Verdict(str, Enum):
    """Terminal verdict of the validation loop for a caption (doc 03 §5)."""

    ACCEPT = "accept"
    FALLBACK = "fallback"
    HARD_ERROR = "hard_error"


class DefectClass(str, Enum):
    """QA defect severity (doc 05 §6)."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class Split(str, Enum):
    """Dataset partition (doc 04 §5)."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    DIAGNOSTIC = "diagnostic"


class TargetModel(str, Enum):
    """Fine-tuning target models / converters (doc 04 §6)."""

    QWEN2_5_VL = "qwen2_5_vl"
    QWEN3_VL = "qwen3_vl"
    INTERNVL3 = "internvl3"
    GEMMA3 = "gemma3"
    MLX_VLM = "mlx_vlm"
