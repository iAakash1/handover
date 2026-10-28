"""Disease Knowledge Base loader and integrity checker (Milestone 2).

Responsibilities:
* parse ``knowledge_base/dkb.json`` into :class:`KnowledgeBase`;
* compute and record ``dkb_sha256`` to pin the source of truth;
* run DKB integrity checks required by the caption framework.

The loader never mutates the DKB; it is a read-only projection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from plantdx.core.enums import AgentCategory, Crop
from plantdx.core.exceptions import KnowledgeBaseError
from plantdx.knowledge_base.models import (
    DiseaseEntry,
    DiseaseReferences,
    KnowledgeBase,
    ReferenceEntry,
    SeverityBuckets,
)


class DKBLoader:
    """Loads and validates the final Disease Knowledge Base."""

    def __init__(self, dkb_path: str | Path) -> None:
        """Initialize the loader with the path to dkb.json."""
        self.dkb_path = Path(dkb_path)

    def load(self) -> KnowledgeBase:
        """Parse, validate, and return the knowledge base.

        Raises:
            KnowledgeBaseError: If the file is missing, malformed, or
                internally inconsistent.
        """
        if not self.dkb_path.is_file():
            raise KnowledgeBaseError(
                f"DKB file not found: {self.dkb_path}"
            )

        try:
            raw_bytes = self.dkb_path.read_bytes()
        except OSError as exc:
            raise KnowledgeBaseError(
                f"Unable to read DKB file {self.dkb_path}: {exc}"
            ) from exc

        dkb_sha256 = hashlib.sha256(raw_bytes).hexdigest()

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise KnowledgeBaseError(
                f"DKB is not valid UTF-8: {self.dkb_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"DKB is not valid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise KnowledgeBaseError(
                "DKB root must be a JSON object."
            )

        metadata = data.get("metadata")
        diseases_raw = data.get("diseases")

        if not isinstance(metadata, dict):
            raise KnowledgeBaseError(
                "DKB must contain a 'metadata' object."
            )

        if not isinstance(diseases_raw, list):
            raise KnowledgeBaseError(
                "DKB must contain a 'diseases' array."
            )

        version = metadata.get("version")
        if not isinstance(version, str) or not version.strip():
            raise KnowledgeBaseError(
                "DKB metadata.version must be a non-empty string."
            )

        reference_registry_raw = metadata.get("reference_registry", {})
        if not isinstance(reference_registry_raw, dict):
            raise KnowledgeBaseError(
                "DKB metadata.reference_registry must be an object."
            )

        reference_registry = self._parse_reference_registry(
            reference_registry_raw
        )

        diseases: dict[str, DiseaseEntry] = {}

        for index, raw_entry in enumerate(diseases_raw):
            if not isinstance(raw_entry, dict):
                raise KnowledgeBaseError(
                    f"DKB disease entry at index {index} must be an object."
                )

            entry = self._parse_disease(raw_entry, index)

            if entry.id in diseases:
                raise KnowledgeBaseError(
                    f"Duplicate disease ID in DKB: {entry.id!r}"
                )

            diseases[entry.id] = entry

        kb = KnowledgeBase(
            version=version,
            dkb_sha256=dkb_sha256,
            reference_registry=reference_registry,
            diseases=diseases,
        )

        self.verify(kb)

        return kb

    def verify(self, kb: KnowledgeBase) -> None:
        """Run DKB integrity self-checks.

        Raises:
            KnowledgeBaseError: If an integrity invariant is violated.
        """
        if not kb.diseases:
            raise KnowledgeBaseError(
                "DKB contains no disease/condition entries."
            )

        reference_keys = set(kb.reference_registry)

        for disease_id, entry in kb.diseases.items():
            if disease_id != entry.id:
                raise KnowledgeBaseError(
                    f"DKB disease index mismatch for {disease_id!r}."
                )

            # Every referenced citation key must exist in the registry.
            for ref_group_name, ref_group in (
                ("recent_research", entry.references.recent_research),
                ("extension_service", entry.references.extension_service),
                ("textbook", entry.references.textbook),
            ):
                for ref_key in ref_group:
                    if ref_key not in reference_keys:
                        raise KnowledgeBaseError(
                            f"{entry.class_label}: reference key "
                            f"{ref_key!r} in {ref_group_name} is not present "
                            "in metadata.reference_registry."
                        )

            # A forbidden term must never also be recommended vocabulary.
            recommended_groups = (
                entry.recommended_controlled_vocabulary,
                entry.recommended_synonyms,
                entry.recommended_adjectives,
                entry.recommended_caption_vocabulary,
                entry.severity_vocabulary,
                entry.color_vocabulary,
                entry.shape_vocabulary,
                entry.texture_vocabulary,
            )

            recommended_terms = {
                term.strip().casefold()
                for group in recommended_groups
                for term in group
                if term.strip()
            }

            forbidden_terms = {
                term.strip().casefold()
                for term in entry.forbidden_terms
                if term.strip()
            }

            forbidden_adjectives = {
                term.strip().casefold()
                for term in entry.forbidden_adjectives
                if term.strip()
            }

            overlap = recommended_terms & forbidden_terms
            if overlap:
                raise KnowledgeBaseError(
                    f"{entry.class_label}: terms appear in both forbidden_terms "
                    f"and recommended vocabulary: {sorted(overlap)}"
                )

            adjective_overlap = (
                {
                    term.strip().casefold()
                    for term in entry.recommended_adjectives
                    if term.strip()
                }
                & forbidden_adjectives
            )

            if adjective_overlap:
                raise KnowledgeBaseError(
                    f"{entry.class_label}: adjectives appear in both "
                    f"recommended_adjectives and forbidden_adjectives: "
                    f"{sorted(adjective_overlap)}"
                )

            # Healthy classes do not require disease terminology.
            is_healthy = entry.class_label.lower().startswith("healthy_")

            if not is_healthy:
                if not entry.diagnostic_visual_features:
                    raise KnowledgeBaseError(
                        f"{entry.class_label}: non-healthy entry has no "
                        "diagnostic_visual_features."
                    )

                if not entry.key_differentiating_features:
                    raise KnowledgeBaseError(
                        f"{entry.class_label}: non-healthy entry has no "
                        "key_differentiating_features."
                    )

            # Pathogen consistency is checked against the enum contract.
            expected_pathogen = entry.agent_category.is_pathogen

            if entry.is_pathogen_disease != expected_pathogen:
                raise KnowledgeBaseError(
                    f"{entry.class_label}: is_pathogen_disease="
                    f"{entry.is_pathogen_disease!r} does not match "
                    f"agent_category={entry.agent_category.value!r}."
                )

            # Pest categories must not be marked as pathogen diseases.
            if entry.agent_category in {
                AgentCategory.ARTHROPOD_PEST,
                AgentCategory.INSECT_PEST,
            } and entry.is_pathogen_disease:
                raise KnowledgeBaseError(
                    f"{entry.class_label}: pest entry cannot be marked "
                    "is_pathogen_disease=true."
                )

            # The current project is tomato-only.
            if entry.crop is not Crop.TOMATO:
                raise KnowledgeBaseError(
                    f"{entry.class_label}: unsupported crop "
                    f"{entry.crop.value!r}; expected 'tomato'."
                )

    @staticmethod
    def _parse_reference_registry(
        raw_registry: dict[str, Any],
    ) -> dict[str, ReferenceEntry]:
        """Convert metadata.reference_registry into typed entries."""
        registry: dict[str, ReferenceEntry] = {}

        for key, raw_ref in raw_registry.items():
            if not isinstance(key, str) or not key.strip():
                raise KnowledgeBaseError(
                    "Reference registry contains an invalid key."
                )

            if not isinstance(raw_ref, dict):
                raise KnowledgeBaseError(
                    f"Reference {key!r} must be an object."
                )

            citation = raw_ref.get("citation")
            url = raw_ref.get("url", "")

            if not isinstance(citation, str) or not citation.strip():
                raise KnowledgeBaseError(
                    f"Reference {key!r} must have a non-empty citation."
                )

            if not isinstance(url, str):
                raise KnowledgeBaseError(
                    f"Reference {key!r}: url must be a string."
                )

            registry[key] = ReferenceEntry(
                key=key,
                citation=citation,
                url=url,
            )

        return registry

    @staticmethod
    def _parse_disease(
        raw: dict[str, Any],
        index: int,
    ) -> DiseaseEntry:
        """Convert one raw DKB disease object into DiseaseEntry."""
        required = {
            "id",
            "crop",
            "dataset",
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
            "host_plant",
            "environmental_conditions",
            "disease_progression",
            "primary_symptoms",
            "secondary_symptoms",
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
            "severity",
            "confused_with",
            "key_differentiating_features",
            "diagnostic_visual_features",
            "forbidden_symptoms_not_leaf_observable",
            "recommended_controlled_vocabulary",
            "recommended_synonyms",
            "recommended_adjectives",
            "forbidden_adjectives",
            "recommended_caption_vocabulary",
            "severity_vocabulary",
            "color_vocabulary",
            "shape_vocabulary",
            "texture_vocabulary",
            "forbidden_terms",
            "management_practices",
            "treatment_recommendations",
            "prevention_recommendations",
            "references",
        }

        missing = sorted(required - raw.keys())
        if missing:
            raise KnowledgeBaseError(
                f"DKB disease entry at index {index} is missing fields: "
                f"{missing}"
            )

        disease_id = _require_string(raw["id"], "id", index)
        crop_raw = _require_string(raw["crop"], "crop", index)
        agent_raw = _require_string(
            raw["agent_category"],
            "agent_category",
            index,
        )

        try:
            crop = Crop(crop_raw)
        except ValueError as exc:
            raise KnowledgeBaseError(
                f"{disease_id}: unsupported crop {crop_raw!r}."
            ) from exc

        try:
            agent_category = AgentCategory(agent_raw)
        except ValueError as exc:
            raise KnowledgeBaseError(
                f"{disease_id}: unsupported agent_category "
                f"{agent_raw!r}."
            ) from exc

        is_pathogen_disease = raw["is_pathogen_disease"]
        if not isinstance(is_pathogen_disease, bool):
            raise KnowledgeBaseError(
                f"{disease_id}: is_pathogen_disease must be boolean."
            )

        severity_raw = raw["severity"]
        if not isinstance(severity_raw, dict):
            raise KnowledgeBaseError(
                f"{disease_id}: severity must be an object."
            )

        severity = SeverityBuckets(
            mild=_string_tuple(
                severity_raw.get("mild"),
                f"{disease_id}.severity.mild",
            ),
            moderate=_string_tuple(
                severity_raw.get("moderate"),
                f"{disease_id}.severity.moderate",
            ),
            severe=_string_tuple(
                severity_raw.get("severe"),
                f"{disease_id}.severity.severe",
            ),
        )

        references_raw = raw["references"]
        if not isinstance(references_raw, dict):
            raise KnowledgeBaseError(
                f"{disease_id}: references must be an object."
            )

        references = DiseaseReferences(
            recent_research=_string_tuple(
                references_raw.get("recent_research"),
                f"{disease_id}.references.recent_research",
            ),
            extension_service=_string_tuple(
                references_raw.get("extension_service"),
                f"{disease_id}.references.extension_service",
            ),
            textbook=_string_tuple(
                references_raw.get("textbook"),
                f"{disease_id}.references.textbook",
            ),
        )

        return DiseaseEntry(
            id=disease_id,
            crop=crop,
            dataset=_require_string(raw["dataset"], "dataset", index),
            class_label=_require_string(
                raw["class_label"],
                "class_label",
                index,
            ),
            is_pathogen_disease=is_pathogen_disease,
            agent_category=agent_category,
            disease=_require_string(raw["disease"], "disease", index),
            common_name=_require_string(
                raw["common_name"],
                "common_name",
                index,
            ),
            scientific_name=_require_string(
                raw["scientific_name"],
                "scientific_name",
                index,
            ),
            scientific_name_synonyms=_string_tuple(
                raw["scientific_name_synonyms"],
                f"{disease_id}.scientific_name_synonyms",
            ),
            taxonomy_note=_require_string(
                raw["taxonomy_note"],
                "taxonomy_note",
                index,
            ),
            pathogen_type=_require_string(
                raw["pathogen_type"],
                "pathogen_type",
                index,
            ),
            pathogen_family=_require_string(
                raw["pathogen_family"],
                "pathogen_family",
                index,
            ),
            host_plant=_require_string(
                raw["host_plant"],
                "host_plant",
                index,
            ),
            environmental_conditions=_string_tuple(
                raw["environmental_conditions"],
                f"{disease_id}.environmental_conditions",
            ),
            disease_progression=_require_string(
                raw["disease_progression"],
                "disease_progression",
                index,
            ),
            primary_symptoms=_string_tuple(
                raw["primary_symptoms"],
                f"{disease_id}.primary_symptoms",
            ),
            secondary_symptoms=_string_tuple(
                raw["secondary_symptoms"],
                f"{disease_id}.secondary_symptoms",
            ),
            leaf_color=_string_tuple(
                raw["leaf_color"],
                f"{disease_id}.leaf_color",
            ),
            lesion_morphology=_string_tuple(
                raw["lesion_morphology"],
                f"{disease_id}.lesion_morphology",
            ),
            lesion_shape=_string_tuple(
                raw["lesion_shape"],
                f"{disease_id}.lesion_shape",
            ),
            lesion_size=_require_string(
                raw["lesion_size"],
                "lesion_size",
                index,
            ),
            lesion_distribution=_string_tuple(
                raw["lesion_distribution"],
                f"{disease_id}.lesion_distribution",
            ),
            leaf_margin_changes=_string_tuple(
                raw["leaf_margin_changes"],
                f"{disease_id}.leaf_margin_changes",
            ),
            leaf_curling=_string_tuple(
                raw["leaf_curling"],
                f"{disease_id}.leaf_curling",
            ),
            necrosis=_string_tuple(
                raw["necrosis"],
                f"{disease_id}.necrosis",
            ),
            chlorosis=_string_tuple(
                raw["chlorosis"],
                f"{disease_id}.chlorosis",
            ),
            texture_changes=_string_tuple(
                raw["texture_changes"],
                f"{disease_id}.texture_changes",
            ),
            severity_indicators=_string_tuple(
                raw["severity_indicators"],
                f"{disease_id}.severity_indicators",
            ),
            severity=severity,
            confused_with=_string_tuple(
                raw["confused_with"],
                f"{disease_id}.confused_with",
            ),
            key_differentiating_features=_string_tuple(
                raw["key_differentiating_features"],
                f"{disease_id}.key_differentiating_features",
            ),
            diagnostic_visual_features=_string_tuple(
                raw["diagnostic_visual_features"],
                f"{disease_id}.diagnostic_visual_features",
            ),
            forbidden_symptoms_not_leaf_observable=_string_tuple(
                raw["forbidden_symptoms_not_leaf_observable"],
                f"{disease_id}.forbidden_symptoms_not_leaf_observable",
            ),
            recommended_controlled_vocabulary=_string_tuple(
                raw["recommended_controlled_vocabulary"],
                f"{disease_id}.recommended_controlled_vocabulary",
            ),
            recommended_synonyms=_string_tuple(
                raw["recommended_synonyms"],
                f"{disease_id}.recommended_synonyms",
            ),
            recommended_adjectives=_string_tuple(
                raw["recommended_adjectives"],
                f"{disease_id}.recommended_adjectives",
            ),
            forbidden_adjectives=_string_tuple(
                raw["forbidden_adjectives"],
                f"{disease_id}.forbidden_adjectives",
            ),
            recommended_caption_vocabulary=_string_tuple(
                raw["recommended_caption_vocabulary"],
                f"{disease_id}.recommended_caption_vocabulary",
            ),
            severity_vocabulary=_string_tuple(
                raw["severity_vocabulary"],
                f"{disease_id}.severity_vocabulary",
            ),
            color_vocabulary=_string_tuple(
                raw["color_vocabulary"],
                f"{disease_id}.color_vocabulary",
            ),
            shape_vocabulary=_string_tuple(
                raw["shape_vocabulary"],
                f"{disease_id}.shape_vocabulary",
            ),
            texture_vocabulary=_string_tuple(
                raw["texture_vocabulary"],
                f"{disease_id}.texture_vocabulary",
            ),
            forbidden_terms=_string_tuple(
                raw["forbidden_terms"],
                f"{disease_id}.forbidden_terms",
            ),
            management_practices=_string_tuple(
                raw["management_practices"],
                f"{disease_id}.management_practices",
            ),
            treatment_recommendations=_string_tuple(
                raw["treatment_recommendations"],
                f"{disease_id}.treatment_recommendations",
            ),
            prevention_recommendations=_string_tuple(
                raw["prevention_recommendations"],
                f"{disease_id}.prevention_recommendations",
            ),
            references=references,
        )

    @staticmethod
    def sha256(path: str | Path) -> str:
        """Return the SHA-256 digest of a DKB file."""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_string(value: Any, field_name: str, index: int) -> str:
    """Require a non-empty string field."""
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeBaseError(
            f"DKB disease entry at index {index}: "
            f"{field_name} must be a non-empty string."
        )
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    """Validate and convert a JSON string array to a tuple."""
    if not isinstance(value, list):
        raise KnowledgeBaseError(
            f"{field_name} must be a JSON array."
        )

    result: list[str] = []

    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise KnowledgeBaseError(
                f"{field_name}[{index}] must be a string."
            )
        if not item.strip():
            raise KnowledgeBaseError(
                f"{field_name}[{index}] must not be empty."
            )
        result.append(item)

    return tuple(result)