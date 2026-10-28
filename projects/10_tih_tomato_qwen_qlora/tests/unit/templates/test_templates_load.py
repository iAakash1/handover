"""Load-side + legality tests for the Template Engine."""

from __future__ import annotations

from typing import Any

import pytest

from plantdx.core.enums import Style
from plantdx.templates import validate_library


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_library_loads_and_validates(library: Any) -> None:
    report = validate_library(library)
    assert report["status"] == "valid"
    assert len(library.templates) >= 24


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_every_family_is_present(library: Any) -> None:
    families = {t.family for t in library.templates}
    assert families == {s.value for s in Style}


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_unique_ids(library: Any) -> None:
    ids = [t.id for t in library.templates]
    assert len(ids) == len(set(ids))


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_healthy_templates_isolated(library: Any) -> None:
    for t in library.templates:
        if set(t.sign_type_allow) == {"healthy"}:
            assert "healthy_state" in t.required
            assert "primary_sign" not in t.required
        elif "healthy" in t.sign_type_allow:
            raise AssertionError(f"{t.id} mixes healthy with disease signs")
