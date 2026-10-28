"""Data models for the derived caption ontology (doc 01 §2-§3).

The ontology is a deterministic *projection* of the DKB. These models describe
the projection's output; the derivation rules live in
:class:`plantdx.ontology.builder.OntologyBuilder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from plantdx.core.enums import ConceptId, Register, SignType


@dataclass(frozen=True, slots=True)
class ConceptSpec:
    """Global concept-registry entry (doc 01 §2.1)."""

    concept_id: ConceptId
    backing_fields: tuple[str, ...]
    observable: bool
    registers: tuple[Register, ...]
    min_cardinality: int
    max_cardinality: int
    default_salience: float


@dataclass(frozen=True, slots=True)
class ConceptSchema:
    """The global concept registry + co-selection constraints (doc 01 §2)."""

    concepts: dict[ConceptId, ConceptSpec]
    mutex_groups: tuple[tuple[str, ...], ...]
    co_selection_requires: dict[ConceptId, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class VocabAxes:
    """Per-disease controlled vocabulary axes (doc 01 §3.1)."""

    color: tuple[str, ...]
    shape: tuple[str, ...]
    size: tuple[str, ...]
    texture: tuple[str, ...]
    location: tuple[str, ...]
    extent: tuple[str, ...]
    severity_stage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegisterPolicy:
    """Which registers are enabled for a disease (doc 01 §3.1)."""

    visual: bool = True
    clinical: bool = True
    educational: bool = True
    severity_conditioned: bool = False


@dataclass(frozen=True, slots=True)
class DiseaseOntology:
    """The derived per-disease caption ontology record (doc 01 §3.1)."""

    disease_id: str
    is_pathogen_disease: bool
    register_policy: RegisterPolicy
    required_concepts: tuple[ConceptId, ...]
    optional_concepts: tuple[ConceptId, ...]
    forbidden_concepts: tuple[str, ...]
    observable_concepts: tuple[ConceptId, ...]
    non_observable_concepts: tuple[ConceptId, ...]
    min_information: int
    max_information: int
    required_medical_terminology: tuple[str, ...]
    optional_descriptive_terminology: tuple[str, ...]
    vocab_axes: VocabAxes
    never_appear: tuple[str, ...]
    concept_realizations: dict[ConceptId, tuple[str, ...]]
    concept_sign_types: dict[str, SignType]
    salience: dict[ConceptId, float]
    provenance_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ontology:
    """The whole derived ontology: schema + per-disease records + build id."""

    ontology_build_id: str
    dkb_sha256: str
    schema: ConceptSchema
    diseases: dict[str, DiseaseOntology] = field(default_factory=dict)
