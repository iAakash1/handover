"""Tests for the per-model converter registry (doc 04 §6)."""

from __future__ import annotations

import pytest

from plantdx.core.enums import TargetModel
from plantdx.dataset.converters import CONVERTER_REGISTRY, BaseConverter


@pytest.mark.unit
def test_registry_covers_all_target_models() -> None:
    assert set(CONVERTER_REGISTRY) == set(TargetModel)


@pytest.mark.unit
def test_converters_subclass_base_and_expose_model_key() -> None:
    for model, cls in CONVERTER_REGISTRY.items():
        assert issubclass(cls, BaseConverter)
        assert cls.model_key == model.value


@pytest.mark.unit
def test_convert_is_stubbed() -> None:
    converter = CONVERTER_REGISTRY[TargetModel.QWEN3_VL]()
    with pytest.raises(NotImplementedError):
        converter.convert(record=None)  # type: ignore[arg-type]
