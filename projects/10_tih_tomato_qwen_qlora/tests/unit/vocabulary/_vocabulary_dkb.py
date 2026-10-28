"""A minimal, valid synthetic DKB for vocabulary + symptom lexicon tests.

Uses the real DKB field names (same shape as ``tests/unit/ontology/_dkb.py``)
but is purpose-built for vocabulary concerns: a primary symptom with a
modifiable sign type, a secondary symptom that must get no modifiers, a
deliberate cross-axis word collision ("raised" as both a shape and a texture
value, on ``tomato_test_blight``) to exercise the lexicon's dedup rule, and a
quality value ("brown") shared by two diseases to exercise dkb_reference
grouping.
"""

from __future__ import annotations

import copy
from typing import Any


def minimal_dkb() -> dict[str, Any]:
    return {
        "metadata": {
            "reference_registry": {
                "REF1": {"citation": "Author (2024). A.", "url": "https://example.org/a"},
                "REF2": {"citation": "Extension (2023). B.", "url": ""},
            }
        },
        "diseases": [
            {
                "id": "tomato_test_blight",
                "crop": "tomato",
                "class_label": "Test Blight",
                "is_pathogen_disease": True,
                "agent_category": "fungus",
                "disease": "Test blight",
                "common_name": "test blight",
                "scientific_name": "Testus fungus",
                "scientific_name_synonyms": ["Oldus fungus"],
                "taxonomy_note": "",
                "pathogen_type": "Fungus",
                "pathogen_family": "Testaceae",
                "environmental_conditions": ["warm humid conditions"],
                "primary_symptoms": ["brown concentric lesions on the lamina"],
                "secondary_symptoms": ["leaflet yellowing"],
                "diagnostic_visual_features": ["target-like concentric rings with a halo"],
                "key_differentiating_features": ["coarser rings than test spot"],
                "forbidden_symptoms_not_leaf_observable": ["stem-end rot on the fruit"],
                "color_vocabulary": ["brown", "yellow"],
                "shape_vocabulary": ["circular", "raised"],
                "texture_vocabulary": ["dry", "raised"],
                "severity_vocabulary": ["few", "numerous"],
                "severity": {
                    "mild": ["a few spots"],
                    "moderate": ["many spots"],
                    "severe": ["extensive blighting"],
                },
                "confused_with": ["test spot (finer rings)"],
                "references": {
                    "recent_research": ["REF1"],
                    "extension_service": ["REF2"],
                    "textbook": [],
                },
            },
            {
                "id": "tomato_test_spot",
                "crop": "tomato",
                "class_label": "Test Spot",
                "is_pathogen_disease": True,
                "agent_category": "fungus",
                "disease": "Test spot",
                "common_name": "test spot",
                "scientific_name": "Spotus fungus",
                "scientific_name_synonyms": [],
                "taxonomy_note": "",
                "pathogen_type": "Fungus",
                "pathogen_family": "Testaceae",
                "environmental_conditions": ["warm"],
                "primary_symptoms": ["fine concentric ring spots"],
                "secondary_symptoms": [],
                "diagnostic_visual_features": ["fine rings"],
                "key_differentiating_features": [],
                "forbidden_symptoms_not_leaf_observable": [],
                "color_vocabulary": ["brown"],
                "shape_vocabulary": ["circular"],
                "texture_vocabulary": ["dry"],
                "severity_vocabulary": ["scattered"],
                "severity": {"mild": [], "moderate": ["many"], "severe": []},
                "confused_with": ["test blight (coarser rings)"],
                "references": {
                    "recent_research": ["REF1"],
                    "extension_service": [],
                    "textbook": [],
                },
            },
            {
                "id": "tomato_healthy",
                "crop": "tomato",
                "class_label": "Healthy",
                "is_pathogen_disease": False,
                "agent_category": "none",
                "disease": "Healthy",
                "common_name": "healthy",
                "scientific_name": "None",
                "scientific_name_synonyms": [],
                "taxonomy_note": "",
                "pathogen_type": "N/A",
                "pathogen_family": "N/A",
                "environmental_conditions": [],
                "primary_symptoms": ["uniform green healthy leaf surface"],
                "secondary_symptoms": [],
                "diagnostic_visual_features": [],
                "key_differentiating_features": [],
                "forbidden_symptoms_not_leaf_observable": ["any spot or lesion"],
                "color_vocabulary": ["green"],
                "shape_vocabulary": [],
                "texture_vocabulary": ["smooth"],
                "severity_vocabulary": ["not applicable"],
                "severity": {"mild": ["not applicable"], "moderate": [], "severe": []},
                "confused_with": [],
                "references": {
                    "recent_research": [],
                    "extension_service": ["REF2"],
                    "textbook": [],
                },
            },
        ],
    }


def clone(dkb: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dkb)
