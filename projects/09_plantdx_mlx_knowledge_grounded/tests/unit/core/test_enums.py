"""Tests for core enumerations."""

from __future__ import annotations

import pytest

from plantdx.core.enums import AgentCategory, ConceptId, Split, TargetModel, TaskType


@pytest.mark.unit
def test_agent_category_is_pathogen() -> None:
    assert AgentCategory.FUNGUS.is_pathogen
    assert AgentCategory.BACTERIUM.is_pathogen
    assert AgentCategory.OOMYCETE.is_pathogen
    assert AgentCategory.VIRUS.is_pathogen
    # Pest / saprophyte / none are NOT tissue pathogens (spec invariant #5).
    assert not AgentCategory.ARTHROPOD_PEST.is_pathogen
    assert not AgentCategory.INSECT_PEST.is_pathogen
    assert not AgentCategory.SAPROPHYTIC_FUNGUS.is_pathogen
    assert not AgentCategory.NONE.is_pathogen


@pytest.mark.unit
def test_concept_registry_has_twenty_concepts() -> None:
    # doc 01 §2.1 defines exactly 20 concept types.
    assert len(list(ConceptId)) == 20


@pytest.mark.unit
def test_string_enums_roundtrip() -> None:
    assert TaskType("describe") is TaskType.DESCRIBE
    assert Split("diagnostic") is Split.DIAGNOSTIC
    assert {m.value for m in TargetModel} == {
        "qwen2_5_vl",
        "qwen3_vl",
        "internvl3",
        "gemma3",
        "mlx_vlm",
    }
