"""Shared fixture: a real caption corpus for exporter tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_REAL_DKB = _ROOT / "knowledge_base" / "dkb.json"
_TEMPLATES = _ROOT / "assets" / "templates" / "templates.json"


@pytest.fixture(scope="session")
def corpus() -> Any:
    """Build the real caption corpus once for exporter tests."""
    if not (_REAL_DKB.is_file() and _TEMPLATES.is_file()):
        pytest.skip("real DKB or templates not present")
    from plantdx.concepts import build_concept_models
    from plantdx.corpus import build_corpus
    from plantdx.ontology.domain import compile_ontology, validate_ontology
    from plantdx.templates import load_library
    from plantdx.vocabulary.domain import build_vocabulary_result

    onto = compile_ontology(_REAL_DKB)
    validate_ontology(onto)
    vocab = build_vocabulary_result(onto.ontology)
    models = build_concept_models(onto.dkb, onto.ontology, vocab)
    library = load_library(_TEMPLATES)
    built, _ = build_corpus(models, library)
    return built
