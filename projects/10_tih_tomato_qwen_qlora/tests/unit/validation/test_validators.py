"""Tests for the validator battery structure (doc 03 §2)."""

from __future__ import annotations

import pytest

from plantdx.validation.validators import ORDERED_VALIDATORS


@pytest.mark.unit
def test_twelve_validators_in_order() -> None:
    assert len(ORDERED_VALIDATORS) == 12
    ids = [v.validator_id for v in ORDERED_VALIDATORS]
    assert ids == [f"V{i}" for i in range(1, 13)]


@pytest.mark.unit
def test_validator_ids_unique() -> None:
    ids = [v.validator_id for v in ORDERED_VALIDATORS]
    assert len(set(ids)) == len(ids)


@pytest.mark.unit
def test_validate_is_stubbed() -> None:
    validator = ORDERED_VALIDATORS[0]()
    with pytest.raises(NotImplementedError):
        validator.validate(record=None, context={})  # type: ignore[arg-type]
