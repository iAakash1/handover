"""Tests for core value objects (dataclasses)."""

from __future__ import annotations

import dataclasses

import pytest

from plantdx.core.enums import Crop
from plantdx.core.types import ImageRef


@pytest.mark.unit
def test_imageref_is_frozen() -> None:
    ref = ImageRef(
        id="x", path="tomato/raw/PlantVillage/a/1.JPG", dataset="PlantVillage", crop=Crop.TOMATO
    )
    assert ref.crop is Crop.TOMATO
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.id = "y"  # type: ignore[misc]


@pytest.mark.unit
def test_frozen_dataclasses_are_hashable() -> None:
    a = ImageRef(id="x", path="p", dataset="PlantVillage", crop=Crop.TOMATO)
    b = ImageRef(id="x", path="p", dataset="PlantVillage", crop=Crop.TOMATO)
    assert a == b
    assert len({a, b}) == 1
