"""In-memory models of the Disease Knowledge Base (Stage 1).

Mirrors the 46-field per-disease schema of ``knowledge_base/dkb.json`` and its
metadata block. These models are read-only projections of the FINAL DKB — the
single source of truth. Nothing here re-authors disease facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plantdx.core.enums import AgentCategory, Crop


@dataclass(frozen=True, slots=True)
class SeverityBuckets:
    """The DKB ``severity`` sub-object."""

    mild: tuple[str, ...]
    moderate: tuple[str, ...]
    severe: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiseaseReferences:
    """The DKB ``references`` sub-object (citation keys)."""

    recent_research: tuple[str, ...]
    extension_service: tuple[str, ...]
    textbook: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiseaseEntry:
    """One disease/condition entry from ``dkb.json`` (all 46 fields)."""

    id: str
    crop: Crop
    dataset: str
    class_label: str
    is_pathogen_disease: bool
    agent_category: AgentCategory
    disease: str
    common_name: str
    scientific_name: str
    scientific_name_synonyms: tuple[str, ...]
    taxonomy_note: str
    pathogen_type: str
    pathogen_family: str
    host_plant: str
    environmental_conditions: tuple[str, ...]
    disease_progression: str
    primary_symptoms: tuple[str, ...]
    secondary_symptoms: tuple[str, ...]
    leaf_color: tuple[str, ...]
    lesion_morphology: tuple[str, ...]
    lesion_shape: tuple[str, ...]
    lesion_size: str
    lesion_distribution: tuple[str, ...]
    leaf_margin_changes: tuple[str, ...]
    leaf_curling: tuple[str, ...]
    necrosis: tuple[str, ...]
    chlorosis: tuple[str, ...]
    texture_changes: tuple[str, ...]
    severity_indicators: tuple[str, ...]
    severity: SeverityBuckets
    confused_with: tuple[str, ...]
    key_differentiating_features: tuple[str, ...]
    diagnostic_visual_features: tuple[str, ...]
    forbidden_symptoms_not_leaf_observable: tuple[str, ...]
    recommended_controlled_vocabulary: tuple[str, ...]
    recommended_synonyms: tuple[str, ...]
    recommended_adjectives: tuple[str, ...]
    forbidden_adjectives: tuple[str, ...]
    recommended_caption_vocabulary: tuple[str, ...]
    severity_vocabulary: tuple[str, ...]
    color_vocabulary: tuple[str, ...]
    shape_vocabulary: tuple[str, ...]
    texture_vocabulary: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    management_practices: tuple[str, ...]
    treatment_recommendations: tuple[str, ...]
    prevention_recommendations: tuple[str, ...]
    references: DiseaseReferences


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    """A citation-registry entry (``metadata.reference_registry``)."""

    key: str
    citation: str
    url: str


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    """The whole DKB: metadata + 18 disease entries, indexed by ``disease_id``."""

    version: str
    dkb_sha256: str
    reference_registry: dict[str, ReferenceEntry]
    diseases: dict[str, DiseaseEntry] = field(default_factory=dict)

    def get(self, disease_id: str) -> DiseaseEntry:
        """Return the entry for ``disease_id`` (raises KeyError if absent)."""
        return self.diseases[disease_id]
