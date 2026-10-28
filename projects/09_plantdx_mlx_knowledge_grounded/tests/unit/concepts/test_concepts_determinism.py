"""Determinism, checksum, and golden-hash regression for the concept models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from plantdx.concepts import build_concept_models, compute_statistics, serialization

# Golden content hash of the concept models derived from the real 18-condition
# DKB. A change here is a semantic event (DKB / ontology / vocabulary / concept
# policy change) and must be a reviewed version bump.
GOLDEN_HASH = "sha256:e6f4870b777d63864d5e33078c9b8183cb49bcb89ab68ea40f471a4791288cb3"


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_repeated_build_is_byte_identical(pipeline: tuple[Any, Any, Any]) -> None:
    onto, vocab, models = pipeline
    again = build_concept_models(onto.dkb, onto.ontology, vocab)
    j1 = serialization.canonical_json(serialization.concept_models_document(models))
    j2 = serialization.canonical_json(serialization.concept_models_document(again))
    assert j1 == j2


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_statistics_are_deterministic(pipeline: tuple[Any, Any, Any]) -> None:
    _, _, models = pipeline
    assert compute_statistics(models, "valid") == compute_statistics(models, "valid")


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_regression_real_dkb_matches_golden(pipeline: tuple[Any, Any, Any]) -> None:
    if not (Path.cwd() / "knowledge_base" / "dkb.json").is_file():
        pytest.skip("must run from repo root for the golden hash")
    _, _, models = pipeline
    assert models.provenance["content_hash"] == GOLDEN_HASH
