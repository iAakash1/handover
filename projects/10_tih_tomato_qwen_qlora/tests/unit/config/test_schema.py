"""Tests for the typed configuration schema (defaults + validation)."""

from __future__ import annotations

import pydantic
import pytest

from plantdx.config.schema import GenerationConfig, PlantDxConfig


@pytest.mark.unit
def test_generation_defaults_match_spec() -> None:
    gen = GenerationConfig()
    # doc 06 §4 defaults.
    assert gen.global_seed == 20260711
    assert gen.captions_per_image == 3
    assert gen.max_attempts == 8
    assert gen.severity_conditioned is False  # invariant #6: gated OFF by default
    assert gen.hedging_probability == pytest.approx(0.9)


@pytest.mark.unit
def test_config_requires_paths() -> None:
    # `paths` is mandatory (no sensible default); constructing without it must fail.
    with pytest.raises(pydantic.ValidationError):
        PlantDxConfig()  # type: ignore[call-arg]


@pytest.mark.unit
def test_config_is_frozen() -> None:
    gen = GenerationConfig()
    with pytest.raises(pydantic.ValidationError):
        gen.global_seed = 1  # type: ignore[misc]
