"""Global ontology policies: the fixed T-Box and deterministic classification maps.

This is the crop-independent half of ``Ontology = f(DKB, Policies)``. It is
authored once and versioned with the code. Nothing here is learned; all maps are
static and applied by exact, order-independent rules. See ontology_design/ docs
02 (schema), 03 (hierarchy), 04 (relations), 05 (rules), 10 (build algorithm).
"""

from __future__ import annotations

import re

from plantdx.ontology.domain.models import ConceptType, RelationType

SCHEMA_VERSION = "1.0.0"
ONTOLOGY_VERSION = "O1"

# --------------------------------------------------------------------------- #
# Deterministic string helpers
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    """Lowercase, collapse non-alphanumerics to '_', strip. Deterministic."""
    return _SLUG_RE.sub("_", text.strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# T-Box: concept types (the complete is_a taxonomy actually instantiated)
# --------------------------------------------------------------------------- #

CONCEPT_TYPES: tuple[ConceptType, ...] = (
    ConceptType("Entity", None, abstract=True),
    ConceptType("BiologicalTaxon", "Entity", abstract=True),
    ConceptType("Crop", "BiologicalTaxon"),
    ConceptType("CausalAgent", "BiologicalTaxon", abstract=True),
    ConceptType("Pathogen", "CausalAgent", abstract=True),
    ConceptType("Bacterium", "Pathogen"),
    ConceptType("Fungus", "Pathogen"),
    ConceptType("Oomycete", "Pathogen"),
    ConceptType("Virus", "Pathogen"),
    ConceptType("Pest", "CausalAgent", abstract=True),
    ConceptType("ArthropodPest", "Pest"),
    ConceptType("InsectPest", "Pest"),
    ConceptType("Saprophyte", "CausalAgent"),
    ConceptType("NoAgent", "CausalAgent", closed=True),
    ConceptType("PathogenFamily", "Entity"),
    ConceptType("Condition", "Entity", abstract=True),
    ConceptType("Disease", "Condition"),
    ConceptType("PestDamage", "Condition"),
    ConceptType("SurfaceColonization", "Condition"),
    ConceptType("HealthyState", "Condition"),
    ConceptType("Observation", "Entity", abstract=True),
    ConceptType("Symptom", "Observation"),
    ConceptType("Quality", "Entity", abstract=True),
    ConceptType("Color", "Quality"),
    ConceptType("Shape", "Quality"),
    ConceptType("Texture", "Quality"),
    ConceptType("Anatomy", "Entity", abstract=True),
    ConceptType("PlantPart", "Anatomy", closed=True),
    ConceptType("LeafRegion", "Anatomy", closed=True),
    ConceptType("SignType", "Entity", closed=True),
    ConceptType("Epistemic", "Entity", abstract=True),
    ConceptType("Severity", "Epistemic", closed=True),
    ConceptType("Extent", "Epistemic"),
    ConceptType("Observability", "Epistemic", closed=True),
    ConceptType("AgentCategory", "Entity", closed=True),
    ConceptType("EnvironmentalCondition", "Entity"),
    ConceptType("Evidence", "Entity", abstract=True),
    ConceptType("PeerReviewed", "Evidence"),
    ConceptType("ExtensionService", "Evidence"),
    ConceptType("Textbook", "Evidence"),
)

CONCEPT_TYPE_BY_ID = {c.id: c for c in CONCEPT_TYPES}


def ancestors(type_id: str) -> list[str]:
    """Return ``[type_id, parent, …, Entity]`` (deterministic).

    Tolerates an unknown ``type_id`` (returns just ``[type_id]``) so the validator
    can report it as a V-ONT-1 error rather than crashing on it.
    """
    chain: list[str] = []
    current: str | None = type_id
    while current is not None:
        chain.append(current)
        concept = CONCEPT_TYPE_BY_ID.get(current)
        if concept is None:
            break
        current = concept.is_a
    return chain


def is_subtype(type_id: str, of: str) -> bool:
    """True if ``type_id`` is ``of`` or a descendant (single-inheritance)."""
    return of in ancestors(type_id)


# --------------------------------------------------------------------------- #
# T-Box: relation types
# --------------------------------------------------------------------------- #

RELATION_TYPES: tuple[RelationType, ...] = (
    RelationType(
        "affects",
        ("Condition",),
        ("Crop",),
        "1..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
    ),
    RelationType(
        "caused_by",
        ("Condition",),
        ("CausalAgent",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
        carries_flags=("disputed",),
    ),
    RelationType("agent_in_category", ("CausalAgent",), ("AgentCategory",), "1", "0..n"),
    RelationType(
        "member_of_family",
        ("Pathogen",),
        ("PathogenFamily",),
        "0..1",
        "0..n",
        carries_evidence=True,
    ),
    RelationType(
        "has_symptom",
        ("Condition",),
        ("Symptom",),
        "1..n",
        "1",
        carries_confidence=True,
        carries_evidence=True,
        carries_flags=("primary",),
    ),
    RelationType("has_sign_type", ("Symptom",), ("SignType",), "1", "0..n"),
    RelationType("has_observability", ("Symptom",), ("Observability",), "1", "0..n"),
    RelationType("appears_on", ("Symptom",), ("LeafRegion", "PlantPart"), "0..n", "0..n"),
    RelationType(
        "has_color",
        ("Condition",),
        ("Color",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
    ),
    RelationType(
        "has_shape",
        ("Condition",),
        ("Shape",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
    ),
    RelationType(
        "has_texture",
        ("Condition",),
        ("Texture",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
    ),
    RelationType(
        "has_extent",
        ("Condition",),
        ("Extent",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
        carries_flags=("image_licensed",),
    ),
    RelationType(
        "typical_at_severity",
        ("Condition",),
        ("Severity",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
        carries_flags=("image_licensed",),
    ),
    RelationType(
        "differentiated_from",
        ("Condition",),
        ("Condition",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
    ),
    RelationType(
        "favored_by",
        ("Condition",),
        ("EnvironmentalCondition",),
        "0..n",
        "0..n",
        carries_confidence=True,
        carries_evidence=True,
    ),
)

RELATION_TYPE_BY_NAME = {r.name: r for r in RELATION_TYPES}

# --------------------------------------------------------------------------- #
# Closed vocabularies (fixed individuals, crop-independent)
# --------------------------------------------------------------------------- #

SIGN_TYPES: tuple[str, ...] = (
    "lesion",
    "coating",
    "gall",
    "stippling",
    "cut",
    "deformation",
    "mottle",
    "healthy_surface",
)
LEAF_REGIONS: tuple[str, ...] = (
    "lamina",
    "margin",
    "tip",
    "midrib",
    "vein",
    "interveinal",
    "adaxial_surface",
    "abaxial_surface",
)
NON_LEAF_PARTS: tuple[str, ...] = ("fruit", "stem", "twig", "flower", "root", "whole_plant")
SEVERITY_STAGES: tuple[str, ...] = ("mild", "moderate", "severe")
OBSERVABILITY: tuple[str, ...] = ("observable", "non_observable")
AGENT_CATEGORIES: tuple[str, ...] = (
    "none",
    "bacterium",
    "fungus",
    "oomycete",
    "virus",
    "arthropod_pest",
    "insect_pest",
    "saprophytic_fungus",
)

# --------------------------------------------------------------------------- #
# DKB → ontology mapping policies
# --------------------------------------------------------------------------- #


# Condition subtype from (is_pathogen_disease, agent_category).
def condition_subtype(is_pathogen: bool, agent_category: str) -> str:
    """Return the Condition subtype for a DKB disease's pathogen flag and category."""
    if agent_category == "none":
        return "HealthyState"
    if agent_category == "saprophytic_fungus":
        return "SurfaceColonization"
    if agent_category in ("arthropod_pest", "insect_pest"):
        return "PestDamage"
    return "Disease"


# CausalAgent concept type from agent_category.
AGENT_TYPE_BY_CATEGORY = {
    "bacterium": "Bacterium",
    "fungus": "Fungus",
    "oomycete": "Oomycete",
    "virus": "Virus",
    "arthropod_pest": "ArthropodPest",
    "insect_pest": "InsectPest",
    "saprophytic_fungus": "Saprophyte",
    "none": "NoAgent",
}

# Symptom-bearing DKB fields, in fixed processing order, with abbreviation + confidence.
SYMPTOM_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("diagnostic_visual_features", "dvf", "asserted"),
    ("key_differentiating_features", "kdf", "asserted"),
    ("primary_symptoms", "prm", "asserted"),
    ("secondary_symptoms", "sec", "hedged"),
    ("forbidden_symptoms_not_leaf_observable", "fbd", "hedged"),
)
FORBIDDEN_FIELD = "forbidden_symptoms_not_leaf_observable"
PRIMARY_FIELDS = frozenset({"diagnostic_visual_features", "primary_symptoms"})

# Quality axis -> DKB controlled-vocabulary field. (These are the axes the DKB
# provides as controlled lists; attached at Condition level, matching DKB granularity.)
QUALITY_AXES: tuple[tuple[str, str], ...] = (
    ("color", "color_vocabulary"),
    ("shape", "shape_vocabulary"),
    ("texture", "texture_vocabulary"),
)

# Sign-type keyword classification, in priority order (first match wins).
SIGN_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gall", ("gall", "wart", "pimple", "nodul", "swelling")),
    ("cut", ("cut", "notch", "windowpane", "trimmed", "scissor", "chewed")),
    ("stippling", ("stippl", "webbing", "bronz", "speckl", "fleck")),
    (
        "coating",
        ("coating", "mould", "mold", "sooty", "powdery", "mildew", "film", "velvety", "downy"),
    ),
    ("mottle", ("mosaic", "mottl")),
    (
        "deformation",
        (
            "curl",
            "cupping",
            "distort",
            "fern",
            "malform",
            "pucker",
            "rugose",
            "crinkl",
            "scorch",
            "wilt",
            "dieback",
            "die-back",
            "dried",
        ),
    ),
    (
        "lesion",
        (
            "lesion",
            "spot",
            "blotch",
            "necro",
            "blight",
            "canker",
            "chloros",
            "shot-hole",
            "ring",
            "target",
            "concentric",
            "water-soaked",
            "greasy",
            "halo",
        ),
    ),
)

# Sign types permitted per condition subtype (rule C2); default when no keyword fits.
SUBTYPE_ALLOWED_SIGNS = {
    "Disease": frozenset({"lesion", "mottle", "coating", "deformation"}),
    "PestDamage": frozenset({"stippling", "cut", "gall", "deformation"}),
    "SurfaceColonization": frozenset({"coating"}),
    "HealthyState": frozenset({"healthy_surface"}),
}


def default_sign(subtype: str, agent_category: str) -> str:
    """Return the fallback sign type for a condition subtype and agent category."""
    if subtype == "HealthyState":
        return "healthy_surface"
    if subtype == "SurfaceColonization":
        return "coating"
    if subtype == "PestDamage":
        return "stippling" if agent_category == "arthropod_pest" else "deformation"
    return "mottle" if agent_category == "virus" else "lesion"  # Disease


def classify_sign_type(phrase: str, subtype: str, agent_category: str) -> str:
    """Deterministically pick a sign type for a symptom phrase (rule-constrained)."""
    if subtype == "HealthyState":
        return "healthy_surface"
    text = phrase.lower()
    for sign, keywords in SIGN_TYPE_KEYWORDS:
        if any(k in text for k in keywords):
            if sign in SUBTYPE_ALLOWED_SIGNS[subtype]:
                return sign
            break  # matched but incompatible with subtype -> fall through to default
    return default_sign(subtype, agent_category)


# Anatomy keyword maps for best-effort `appears_on` (leaf regions for any symptom;
# non-leaf parts only for forbidden/non-observable symptoms — keeps rule C4 clean).
LEAF_REGION_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("interveinal", "interveinal"),
    ("midrib", "midrib"),
    ("underside", "abaxial_surface"),
    ("abaxial", "abaxial_surface"),
    ("lower surface", "abaxial_surface"),
    ("upper surface", "adaxial_surface"),
    ("adaxial", "adaxial_surface"),
    ("margin", "margin"),
    ("tip", "tip"),
    ("vein", "vein"),
    ("lamina", "lamina"),
)
NON_LEAF_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("fruit", "fruit"),
    ("twig", "twig"),
    ("branch", "twig"),
    ("stem", "stem"),
    ("blossom", "flower"),
    ("flower", "flower"),
    ("root", "root"),
    ("whole plant", "whole_plant"),
    ("tree", "whole_plant"),
    ("vascular", "stem"),
    ("gummosis", "stem"),
)

# --------------------------------------------------------------------------- #
# DKB field coverage (rule V-ONT-11)
# --------------------------------------------------------------------------- #

CONSUMED_FIELDS = frozenset(
    {
        "id",
        "crop",
        "class_label",
        "is_pathogen_disease",
        "agent_category",
        "disease",
        "common_name",
        "scientific_name",
        "scientific_name_synonyms",
        "taxonomy_note",
        "pathogen_type",
        "pathogen_family",
        "environmental_conditions",
        "primary_symptoms",
        "secondary_symptoms",
        "diagnostic_visual_features",
        "key_differentiating_features",
        "forbidden_symptoms_not_leaf_observable",
        "color_vocabulary",
        "shape_vocabulary",
        "texture_vocabulary",
        "severity_vocabulary",
        "severity",
        "confused_with",
        "references",
    }
)
# Retained on the condition or intentionally not decomposed (prose / caption-layer / agronomic).
ALLOWLIST_FIELDS = frozenset(
    {
        "dataset",
        "host_plant",
        "disease_progression",
        "leaf_color",
        "lesion_morphology",
        "lesion_shape",
        "lesion_size",
        "lesion_distribution",
        "leaf_margin_changes",
        "leaf_curling",
        "necrosis",
        "chlorosis",
        "texture_changes",
        "severity_indicators",
        "recommended_controlled_vocabulary",
        "recommended_synonyms",
        "recommended_adjectives",
        "forbidden_adjectives",
        "recommended_caption_vocabulary",
        "forbidden_terms",
        "management_practices",
        "treatment_recommendations",
        "prevention_recommendations",
    }
)
