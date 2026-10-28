"""Determinism, checksum, statistics, and regression tests for the compiler."""

from __future__ import annotations

from pathlib import Path

import pytest
from _vocabulary_dkb import minimal_dkb

from plantdx.ontology.domain import builder as ontology_builder
from plantdx.vocabulary.domain import (
    build_vocabulary_result,
    checksum,
    compute_statistics,
    serialization,
    statistics,
    validate_vocabulary_result,
)

# Golden content hash of the vocabulary+lexicon derived from the real 18-condition
# DKB's compiled ontology. A change here is a semantic event (DKB / ontology
# policy / vocabulary builder change) and must be a reviewed version bump.
GOLDEN_REAL_HASH = "sha256:99d5833cf6405eea8edf01b227411c5db572393e6dfe27323ce6519b29249628"
_REAL_DKB = Path(__file__).resolve().parents[3] / "knowledge_base" / "dkb.json"


def _compile_ontology():
    dkb = minimal_dkb()
    ontology_builder.validate_dkb(dkb)
    ontology, _log = ontology_builder.build_ontology(dkb, "testsha")
    return ontology


@pytest.mark.unit
def test_repeated_build_is_byte_identical() -> None:
    ontology = _compile_ontology()
    r1 = build_vocabulary_result(ontology)
    r2 = build_vocabulary_result(ontology)
    j1 = serialization.canonical_json(serialization.vocabulary_document(r1))
    j2 = serialization.canonical_json(serialization.vocabulary_document(r2))
    assert j1 == j2
    l1 = serialization.canonical_json(serialization.lexicon_document(r1))
    l2 = serialization.canonical_json(serialization.lexicon_document(r2))
    assert l1 == l2


@pytest.mark.unit
def test_checksum_is_content_only() -> None:
    ontology = _compile_ontology()
    r1 = build_vocabulary_result(ontology)
    # A different (fabricated) ontology provenance must not change the vocabulary
    # content hash, as long as the derived items are identical.
    r2 = build_vocabulary_result(ontology)
    r2.provenance["ontology_content_hash"] = "sha256:completely_different"
    assert checksum.content_hash(r1) == checksum.content_hash(r2)


@pytest.mark.unit
def test_statistics_are_deterministic() -> None:
    ontology = _compile_ontology()
    result = build_vocabulary_result(ontology)
    s1 = statistics.compute(result, "valid")
    s2 = statistics.compute(result, "valid")
    assert s1 == s2
    assert s1["vocabulary_item_count"] == len(result.vocabulary_items)
    assert s1["lexicon_item_count"] == len(result.lexicon_items)
    assert s1["build_checksum"] == result.provenance["content_hash"]


@pytest.mark.unit
def test_compute_statistics_wrapper_matches_module() -> None:
    ontology = _compile_ontology()
    result = build_vocabulary_result(ontology)
    assert compute_statistics(result, "valid") == statistics.compute(result, "valid")


@pytest.mark.unit
def test_validate_vocabulary_result_report_shape() -> None:
    ontology = _compile_ontology()
    result = build_vocabulary_result(ontology)
    report = validate_vocabulary_result(result, ontology)
    assert report["status"] == "valid"
    assert report["violation_count"] == 0
    assert report["vocabulary_item_count"] == len(result.vocabulary_items)
    assert report["lexicon_item_count"] == len(result.lexicon_items)


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_regression_real_dkb_builds_validates_and_matches_golden() -> None:
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")
    dkb = ontology_builder.load_dkb(_REAL_DKB)
    ontology_builder.validate_dkb(dkb)
    ontology, _ = ontology_builder.build_ontology(dkb, ontology_builder.dkb_file_sha256(_REAL_DKB))
    result = build_vocabulary_result(ontology)
    validate_vocabulary_result(result, ontology)  # 18 conditions, must pass
    assert result.provenance["content_hash"] == GOLDEN_REAL_HASH
