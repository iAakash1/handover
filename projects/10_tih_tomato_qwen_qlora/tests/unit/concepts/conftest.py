"""Shared real-pipeline fixture for the concept-model tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_REAL_DKB = Path(__file__).resolve().parents[3] / "knowledge_base" / "dkb.json"


@pytest.fixture(scope="session")
def pipeline() -> tuple[Any, Any, Any]:
    """Compile ontology + vocabulary + concept models from the real DKB (once)."""
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")
    from plantdx.concepts import build_concept_models
    from plantdx.ontology.domain import compile_ontology, validate_ontology
    from plantdx.vocabulary.domain import build_vocabulary_result

    onto = compile_ontology(_REAL_DKB)
    validate_ontology(onto)
    vocab = build_vocabulary_result(onto.ontology)
    models = build_concept_models(onto.dkb, onto.ontology, vocab)
    return onto, vocab, models
