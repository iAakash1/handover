"""Full corpus build: coverage, validity, and the severity-honesty invariant."""

from __future__ import annotations

import re
from typing import Any

import pytest

from plantdx.concepts.policies import STAGE_TOKENS
from plantdx.corpus.validator import validate_caption


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_every_disease_has_captions(bundle: tuple[Any, Any, Any]) -> None:
    _, _, (corpus, report) = bundle
    assert report["status"] == "valid"
    assert len(corpus.captions) > 0
    diseases = {c.disease_id for c in corpus.captions}
    assert len(diseases) == 18


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_every_caption_revalidates(bundle: tuple[Any, Any, Any]) -> None:
    models, library, (corpus, _report) = bundle
    by_model = {m.disease_id: m for m in models.disease_models}
    by_template = {t.id: t for t in library.templates}
    for c in corpus.captions:
        assert validate_caption(c, by_model[c.disease_id], by_template[c.template_id]) == []


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_no_caption_asserts_a_severity_stage(bundle: tuple[Any, Any, Any]) -> None:
    _, _, (corpus, _report) = bundle
    for c in corpus.captions:
        low = c.text.lower()
        for token in STAGE_TOKENS:
            assert not re.search(rf"(?<!\w){re.escape(token)}(?!\w)", low), c.text


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_no_exact_duplicate_captions(bundle: tuple[Any, Any, Any]) -> None:
    _, _, (corpus, _report) = bundle
    norms = [" ".join(c.text.lower().split()) for c in corpus.captions]
    assert len(norms) == len(set(norms))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_captions_start_capital_end_period(bundle: tuple[Any, Any, Any]) -> None:
    _, _, (corpus, _report) = bundle
    for c in corpus.captions:
        assert c.text[0].isupper()
        assert c.text[-1] in ".!?"


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_no_ontology_parenthetical_leakage(bundle: tuple[Any, Any, Any]) -> None:
    """RC1 W1: DKB disambiguation parentheticals must not reach captions."""
    _, _, (corpus, _report) = bundle
    for c in corpus.captions:
        for leak in ("(halo)", "(early)", "(necrotic galls)", "(smooth)", "(sooty mold)"):
            assert leak not in c.text, c.text


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_no_adjacent_duplicate_words(bundle: tuple[Any, Any, Any]) -> None:
    """RC1 W3: no caption repeats a word back-to-back."""
    _, _, (corpus, _report) = bundle
    for c in corpus.captions:
        assert not re.search(r"\b(\w+)\s+\1\b", c.text, re.IGNORECASE), c.text


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_disease_balance_and_healthy_coverage(bundle: tuple[Any, Any, Any]) -> None:
    """RC1 W5/W6: every disease (incl. healthy) has meaningful coverage."""
    _, _, (corpus, _report) = bundle
    counts: dict[str, int] = {}
    for c in corpus.captions:
        counts[c.disease_id] = counts.get(c.disease_id, 0) + 1
    assert min(counts.values()) >= 12  # sparse diseases still well covered
    healthy = [c for c in corpus.captions if c.condition_type == "HealthyState"]
    assert len(healthy) >= 24


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_condition_and_crop_filters(bundle: tuple[Any, Any, Any]) -> None:
    models, library, _ = bundle
    from plantdx.corpus import build_corpus

    one, _ = build_corpus(models, library, condition="tomato_early_blight")
    assert {c.disease_id for c in one.captions} == {"tomato_early_blight"}
    mango, _ = build_corpus(models, library, crop="mango")
    assert {c.crop for c in mango.captions} == {"mango"}
