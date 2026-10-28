"""Determinism, checksum, and golden-hash regression for the caption corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from plantdx.corpus import build_corpus, serialization

# Golden content hash of the corpus built from the real DKB + template library.
# A change is a semantic event (any upstream change or template edit) requiring a
# reviewed version bump.
GOLDEN_HASH = "sha256:1225eb8605edb71839f89618bfdcd4a1abba414869f64c1fdc5951ee865ba089"


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_repeated_build_is_byte_identical(bundle: tuple[Any, Any, Any]) -> None:
    models, library, (corpus, _report) = bundle
    again, _ = build_corpus(models, library)
    j1 = serialization.canonical_json(serialization.captions_document(corpus))
    j2 = serialization.canonical_json(serialization.captions_document(again))
    assert j1 == j2


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_checksum_is_content_only(bundle: tuple[Any, Any, Any]) -> None:
    models, library, (corpus, _report) = bundle
    again, _ = build_corpus(models, library)
    again.provenance["ontology_content_hash"] = "sha256:different"
    from plantdx.corpus.checksum import content_hash

    assert content_hash(corpus) == content_hash(again)


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_regression_real_dkb_matches_golden(bundle: tuple[Any, Any, Any]) -> None:
    if not (Path.cwd() / "knowledge_base" / "dkb.json").is_file():
        pytest.skip("must run from repo root for the golden hash")
    _, _, (corpus, _report) = bundle
    assert corpus.provenance["content_hash"] == GOLDEN_HASH


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_jsonl_and_csv_row_counts_match(bundle: tuple[Any, Any, Any]) -> None:
    _, _, (corpus, _report) = bundle
    jsonl = serialization.jsonl_text(corpus).strip().splitlines()
    csv_rows = serialization.csv_text(corpus).strip().splitlines()
    assert len(jsonl) == len(corpus.captions)
    assert len(csv_rows) == len(corpus.captions) + 1  # + header
