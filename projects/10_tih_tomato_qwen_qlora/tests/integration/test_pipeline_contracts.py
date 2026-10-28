"""Integration placeholders for the end-to-end pipeline.

These encode the intended behaviour of the assembled pipeline and are skipped
until the relevant milestone implements the components. Keeping them here makes
the acceptance criteria explicit and version-controlled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_REAL_DKB = Path(__file__).resolve().parents[2] / "knowledge_base" / "dkb.json"
_TEMPLATES = Path(__file__).resolve().parents[2] / "assets" / "templates" / "templates.json"


def _build_corpus() -> tuple[Any, Any, Any]:
    if not (_REAL_DKB.is_file() and _TEMPLATES.is_file()):
        pytest.skip("real DKB or templates not present")
    from plantdx.concepts import build_concept_models
    from plantdx.corpus import build_corpus
    from plantdx.ontology.domain import compile_ontology, validate_ontology
    from plantdx.templates import load_library, validate_library
    from plantdx.vocabulary.domain import build_vocabulary_result

    onto = compile_ontology(_REAL_DKB)
    validate_ontology(onto)
    vocab = build_vocabulary_result(onto.ontology)
    models = build_concept_models(onto.dkb, onto.ontology, vocab)
    library = load_library(_TEMPLATES)
    validate_library(library)
    corpus, _report = build_corpus(models, library)
    return models, library, corpus


@pytest.mark.integration
@pytest.mark.skip(reason="Milestone 2: DKB -> ontology derivation")
def test_ontology_is_pure_projection_of_dkb() -> None:
    """Every ontology vocab value must trace to a DKB field (invariant #2)."""


@pytest.mark.integration
@pytest.mark.requires_dkb
def test_generation_never_emits_forbidden_terms() -> None:
    """No accepted caption contains a term from its disease `never_appear` set."""
    import re

    models, _library, corpus = _build_corpus()
    by_model = {m.disease_id: m for m in models.disease_models}
    for c in corpus.captions:
        low = c.text.lower()
        for term in by_model[c.disease_id].never_appear:
            assert not re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", low), (c.text, term)


@pytest.mark.integration
@pytest.mark.requires_dkb
def test_generation_is_bit_for_bit_reproducible() -> None:
    """Same DKB + templates => identical caption corpus (doc 00 §6)."""
    from plantdx.corpus import build_corpus

    models, library, corpus = _build_corpus()
    again, _ = build_corpus(models, library)
    assert again.provenance["content_hash"] == corpus.provenance["content_hash"]


@pytest.mark.integration
@pytest.mark.skip(reason="Milestone 4: splits")
def test_splits_are_grouped_by_image() -> None:
    """All captions of an image share one split (no caption-level leakage)."""


@pytest.mark.integration
@pytest.mark.skip(reason="Milestone 4: converters")
def test_all_converters_preserve_response_text() -> None:
    """Converters change serialization only, never the caption/instruction text."""
