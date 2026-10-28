"""Global vocabulary policies: deterministic classification tables.

The crop-independent half of ``Vocabulary = f(Ontology, Policies)``, mirroring
``plantdx.ontology.domain.policies``'s role: hand-authored, versioned constants,
nothing learned, nothing guessed. See ``docs/VOCABULARY.md`` for the category
design and the two documented exceptions (Confidence, Size) noted here.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "1.0.0"
VOCABULARY_VERSION = "V1"

# --------------------------------------------------------------------------- #
# Vocabulary categories (component B): one ontology concept type -> one
# category, reached through one relation type, with a fixed part-of-speech.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CategorySpec:
    """One vocabulary category: which ontology concept type(s) populate it."""

    category: str  # human-readable, used as LexicalItem.concept
    concept_types: tuple[str, ...]  # plantdx.ontology.domain.policies.ConceptType.id values
    relation: str  # the relation type this item is reached through ("identity" = the node itself)
    part_of_speech: str
    carries_evidence: bool  # whether the grounding relation carries an evidence attribute


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec("color", ("Color",), "has_color", "adjective", carries_evidence=True),
    CategorySpec("shape", ("Shape",), "has_shape", "adjective", carries_evidence=True),
    CategorySpec("texture", ("Texture",), "has_texture", "adjective", carries_evidence=True),
    CategorySpec("extent", ("Extent",), "has_extent", "quantifier", carries_evidence=True),
    CategorySpec("leaf_region", ("LeafRegion",), "appears_on", "noun", carries_evidence=False),
    CategorySpec("sign_type", ("SignType",), "has_sign_type", "noun", carries_evidence=False),
    CategorySpec(
        "agent_name",
        ("Bacterium", "Fungus", "Oomycete", "Virus", "ArthropodPest", "InsectPest", "Saprophyte"),
        "caused_by",
        "phrase",
        carries_evidence=True,
    ),
    CategorySpec(
        "disease_name",
        ("Disease", "PestDamage", "SurfaceColonization", "HealthyState"),
        "identity",
        "phrase",
        carries_evidence=True,  # evidence aggregated from the condition's own outgoing edges
    ),
    CategorySpec(
        "environment", ("EnvironmentalCondition",), "favored_by", "phrase", carries_evidence=True
    ),
    CategorySpec(
        "observability_modifier",
        ("Observability",),
        "has_observability",
        "modifier",
        carries_evidence=False,
    ),
)

# Confidence modifiers are NOT ontology nodes: `Confidence` is an edge-attribute
# enum (plantdx.ontology.domain.models.Confidence), never instantiated as graph
# nodes. This is a deliberate, documented exception (see docs/VOCABULARY.md) —
# the three values are still part of the ontology's schema (every evidence-
# carrying edge's `confidence` attribute), just not graph individuals.
CONFIDENCE_VALUES: tuple[str, ...] = ("asserted", "typical", "hedged")
CONFIDENCE_CATEGORY = CategorySpec(
    "confidence_modifier", (), "identity_enum", "modifier", carries_evidence=False
)

# --------------------------------------------------------------------------- #
# Symptom Lexicon (component C): bounded, non-combinatorial realization rules.
# --------------------------------------------------------------------------- #

# Sign types that may take a color/shape/texture/extent modifier (Caption
# Framework 01_caption_ontology_spec.md §2.4 co-selection rule). Modifiers are
# never attached to stippling/cut/deformation/mottle/healthy_surface symptoms.
MODIFIABLE_SIGN_TYPES = frozenset({"lesion", "coating", "gall", "stippling"})

# Which quality relations attach single-word modifiers to a lexicon head noun.
MODIFIER_RELATIONS: tuple[str, ...] = ("has_color", "has_shape", "has_texture", "has_extent")
