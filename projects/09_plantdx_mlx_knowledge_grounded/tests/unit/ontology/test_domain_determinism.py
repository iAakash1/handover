"""Determinism, checksum, and regression tests for the compiler."""

from __future__ import annotations

from pathlib import Path

import pytest
from _dkb import minimal_dkb

from plantdx.ontology.domain import builder, serialization, statistics, validator

# Golden content hash of the real 18-condition DKB. A change here is a semantic
# event (DKB / policy / builder change) and must be a reviewed version bump.
GOLDEN_REAL_DKB_HASH = "sha256:25ae0f6d9692a6d00a8968dc916a3665001bba29dd45616afe5e9b3c49bf2ca4"
_REAL_DKB = Path(__file__).resolve().parents[3] / "knowledge_base" / "dkb.json"


@pytest.mark.unit
def test_repeated_build_is_byte_identical() -> None:
    dkb = minimal_dkb()
    o1, _ = builder.build_ontology(dkb, "sha")
    o2, _ = builder.build_ontology(dkb, "sha")
    j1 = serialization.canonical_json(serialization.ontology_document(o1))
    j2 = serialization.canonical_json(serialization.ontology_document(o2))
    assert j1 == j2


@pytest.mark.unit
def test_checksum_is_content_only() -> None:
    dkb = minimal_dkb()
    # Different DKB file hashes (provenance) must not change the content hash.
    o_a, _ = builder.build_ontology(dkb, "dkb_hash_A")
    o_b, _ = builder.build_ontology(dkb, "dkb_hash_B")
    assert o_a.provenance["dkb_sha256"] != o_b.provenance["dkb_sha256"]
    assert o_a.provenance["content_hash"] == o_b.provenance["content_hash"]


@pytest.mark.unit
def test_statistics_are_deterministic() -> None:
    dkb = minimal_dkb()
    ontology, _ = builder.build_ontology(dkb, "sha")
    s1 = statistics.compute(ontology, dkb, "valid")
    s2 = statistics.compute(ontology, dkb, "valid")
    assert s1 == s2
    assert s1["condition_concepts"] == 3
    assert s1["build_checksum"] == ontology.provenance["content_hash"]


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_regression_real_dkb_builds_validates_and_matches_golden() -> None:
    if not _REAL_DKB.is_file():
        pytest.skip("real DKB not present")
    dkb = builder.load_dkb(_REAL_DKB)
    builder.validate_dkb(dkb)
    ontology, _ = builder.build_ontology(dkb, builder.dkb_file_sha256(_REAL_DKB))
    validator.validate(ontology, dkb)  # 18 conditions, must pass
    conditions = [
        n
        for n in ontology.nodes
        if n.type in ("Disease", "PestDamage", "SurfaceColonization", "HealthyState")
    ]
    assert len(conditions) == 18
    assert ontology.provenance["content_hash"] == GOLDEN_REAL_DKB_HASH
