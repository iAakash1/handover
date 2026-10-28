"""Global concept-model policies: deterministic classification tables.

The crop-independent half of ``ConceptModels = f(DKB, Ontology, Vocabulary)``,
mirroring the ontology/vocabulary policy modules: hand-authored, versioned
constants, nothing learned, nothing guessed. See ``docs/CONCEPTS.md`` for the
concept taxonomy and the two documented omissions (``lesion_size``, ``management``)
noted here.
"""

from __future__ import annotations

CONCEPTS_VERSION = "C1"
SCHEMA_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# The 20 caption concepts (plantdx.core.enums.ConceptId), in canonical
# sentence-planning order. The Sentence Planner emits concepts in this order;
# the ordering is a legality constraint (relationship legality), not a
# suggestion. Concepts absent from a disease are simply skipped.
# --------------------------------------------------------------------------- #

CONCEPT_ORDER: tuple[str, ...] = (
    "host",
    "disease_identity",
    "agent_category_descriptor",
    "agent_reference",
    "primary_sign",
    "lesion_color",
    "lesion_shape",
    "texture",
    "lesion_size",
    "extent",
    "lesion_distribution",
    "leaf_location",
    "chlorosis",
    "necrosis",
    "leaf_deformation",
    "secondary_sign",
    "differential",
    "healthy_state",
    "severity_stage",
    "management",
)

# Concepts that describe something not visible on a single leaf (or not a visual
# fact at all). Captions in a `visual` register may not realize these; the
# Caption Validator enforces it (observability legality, doc 03 V7).
NON_OBSERVABLE_CONCEPTS = frozenset(
    {
        "agent_reference",
        "agent_category_descriptor",
        "severity_stage",
        "management",
    }
)

# Concepts always forbidden in this milestone's disease-level corpus:
#  - severity_stage: gated behind a per-image severity label that does not exist
#    (doc 00 §5 severity-honesty policy); forbidden unless severity_conditioned.
#  - management: not a leaf-observable visual fact; retained in the DKB for
#    completeness only.
#  - lesion_size: the DKB has no controlled size vocabulary (only free-text
#    prose like "~3-12 mm"); synthesizing a size axis would violate no-invention,
#    so the concept is unavailable (never mandatory/optional, always forbidden).
ALWAYS_FORBIDDEN = frozenset({"severity_stage", "management", "lesion_size"})

# Controlled descriptor phrase per DKB agent_category. These restate the DKB's
# `agent_category` field in caption-ready English (no new facts) — the pest
# categories deliberately avoid infection/pathogen language (spec invariant #5).
AGENT_CATEGORY_DESCRIPTORS: dict[str, str] = {
    "fungus": "a fungal disease",
    "bacterium": "a bacterial disease",
    "oomycete": "an oomycete disease",
    "virus": "a viral disease",
    "arthropod_pest": "a mite infestation",
    "insect_pest": "insect feeding damage",
    "saprophytic_fungus": "a surface fungal growth",
}

# Sign types whose primary sign may take color/shape/texture/extent modifiers
# (Caption Framework 01 §2.4 co-selection rule) — same set the symptom lexicon
# uses. Modifier legality: these concepts are only *available* when the disease's
# primary sign has a modifiable sign type.
MODIFIABLE_SIGN_TYPES = frozenset({"lesion", "coating", "gall", "stippling"})

# Quality concepts that are only offered when the primary sign is modifiable.
MODIFIER_CONCEPTS = frozenset({"lesion_color", "lesion_shape", "texture"})

# --------------------------------------------------------------------------- #
# DKB fields consulted directly by the concept builder for concepts the domain
# ontology did not decompose into structured nodes (chlorosis/necrosis/
# deformation/distribution). Each maps DKB field -> concept_id. Every one of
# these is a per-disease list that reads "none" (or ["none"]) when absent.
# --------------------------------------------------------------------------- #

DKB_FIELD_CONCEPTS: tuple[tuple[str, str], ...] = (
    ("chlorosis", "chlorosis"),
    ("necrosis", "necrosis"),
    ("leaf_curling", "leaf_deformation"),
    ("lesion_distribution", "lesion_distribution"),
)

# Sentinel values in DKB fields meaning "not present" (case-insensitive).
NA_VALUES = frozenset({"none", "n/a", "na", "not applicable", ""})

# First-word tokens that denote absence/negation; a DKB-field phrase beginning
# with one of these describes a feature that is *not* present for the disease and
# is dropped so no caption asserts a negated feature (e.g. "none characteristic",
# "minimal; ...", "no visible ...").
NEGATION_TOKENS = frozenset(
    {"none", "no", "not", "minimal", "n/a", "na", "nil", "absent", "negligible"}
)

# Severity-stage surface tokens that must never appear in an unconditioned
# caption (doc 00 §5, doc 03 V9). Added to every non-severity-conditioned
# disease's ``never_appear`` set as defense in depth.
STAGE_TOKENS: tuple[str, ...] = (
    "mild",
    "moderate",
    "severe",
    "early-stage",
    "early stage",
    "advanced",
    "late-stage",
    "late stage",
)
