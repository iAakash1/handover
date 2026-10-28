"""Fail-closed tests: the Caption Validator detects every injected defect."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from plantdx.corpus.models import Caption
from plantdx.corpus.validator import validate_caption


def _real_caption(bundle: tuple[Any, Any, Any]) -> tuple[Caption, Any, Any]:
    models, library, (corpus, _report) = bundle
    caption = corpus.captions[0]
    model = next(m for m in models.disease_models if m.disease_id == caption.disease_id)
    template = next(t for t in library.templates if t.id == caption.template_id)
    return caption, model, template


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_real_caption_passes(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    assert validate_caption(caption, model, template) == []


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_non_available_concept_detected(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    bad = dataclasses.replace(caption, concepts=(*caption.concepts, "management"))
    assert any(v.startswith("V-CAP-1") for v in validate_caption(bad, model, template))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_template_mismatch_detected(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    bad = dataclasses.replace(caption, template_id="T-NOPE")
    assert any(v.startswith("V-CAP-2") for v in validate_caption(bad, model, template))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_forbidden_concept_detected(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    bad = dataclasses.replace(caption, concepts=(*caption.concepts, "severity_stage"))
    assert any(v.startswith("V-CAP-4") for v in validate_caption(bad, model, template))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_never_appear_detected(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    bad = dataclasses.replace(caption, text=caption.text[:-1] + " but not severe.")
    checks = {v.split(":")[0] for v in validate_caption(bad, model, template)}
    assert "V-CAP-5" in checks or "V-CAP-11" in checks


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_grammar_defect_detected(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    bad = dataclasses.replace(caption, text="lowercase start with  double space")
    checks = {v.split(":")[0] for v in validate_caption(bad, model, template)}
    assert "V-CAP-7" in checks


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_confidence_mismatch_detected(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    other = "hedged" if caption.confidence != "hedged" else "asserted"
    bad = dataclasses.replace(caption, confidence=other)
    assert any(v.startswith("V-CAP-10") for v in validate_caption(bad, model, template))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_evidence_tamper_detected(bundle: tuple[Any, Any, Any]) -> None:
    caption, model, template = _real_caption(bundle)
    bad = dataclasses.replace(caption, evidence=("evidence:FAKE",))
    assert any(v.startswith("V-CAP-12") for v in validate_caption(bad, model, template))
