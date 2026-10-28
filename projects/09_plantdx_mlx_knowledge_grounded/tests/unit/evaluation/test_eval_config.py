"""Eval config resolution: defaults + fail-closed validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.core.exceptions import ConfigError
from plantdx.evaluation.config import resolve_crop, resolve_eval_config


@pytest.mark.unit
def test_defaults_match_the_frozen_tomato_run() -> None:
    cfg = resolve_eval_config()
    assert cfg.stage == "all"
    assert cfg.split == "test"
    assert "Qwen2.5-VL" in cfg.model_path
    # The checkpoint DIRECTORY, not the weights file inside it -- mlx-vlm's
    # apply_lora_layers expects a directory containing adapter_config.json +
    # adapters.safetensors (see evaluation/checkpoint.py).
    assert not cfg.adapter_path.endswith("adapters.safetensors")
    assert "qwen25vl_tomato_qlora" in cfg.adapter_path
    assert cfg.predictions_path.endswith("raw/predictions.jsonl")
    assert cfg.output_dir == "reports/qwen25vl_tomato_qlora/evaluation"


@pytest.mark.unit
def test_output_dir_defaults_from_the_adapters_own_run_name_not_a_hardcoded_crop() -> None:
    """The report directory must track whichever run is actually being
    evaluated -- never a literal tomato (or any other) crop name. Regression
    test for the bug where mango evaluation output silently landed in
    reports/qwen25vl_tomato_qlora/ because the default was a hardcoded string
    independent of --adapter."""
    cfg = resolve_eval_config(adapter="checkpoints/qwen25vl_mango_qlora")
    assert cfg.output_dir == "reports/qwen25vl_mango_qlora/evaluation"


@pytest.mark.unit
def test_output_dir_default_resolves_the_weights_file_form_too() -> None:
    cfg = resolve_eval_config(adapter="checkpoints/qwen25vl_mango_qlora/adapters.safetensors")
    assert cfg.output_dir == "reports/qwen25vl_mango_qlora/evaluation"


@pytest.mark.unit
def test_explicit_output_dir_still_wins() -> None:
    cfg = resolve_eval_config(adapter="checkpoints/qwen25vl_mango_qlora", output_dir="/custom")
    assert cfg.output_dir == "/custom"


@pytest.mark.unit
def test_resolve_crop_reads_the_dataset_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"crop": "mango"}), encoding="utf-8")
    assert resolve_crop(tmp_path) == "mango"


@pytest.mark.unit
def test_resolve_crop_fails_closed_when_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="manifest not found"):
        resolve_crop(tmp_path)


@pytest.mark.unit
def test_resolve_crop_fails_closed_when_crop_field_missing(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"classes": []}), encoding="utf-8")
    with pytest.raises(ConfigError, match="no 'crop' field"):
        resolve_crop(tmp_path)


@pytest.mark.unit
def test_overrides_are_applied() -> None:
    cfg = resolve_eval_config(stage="analyze", split="validation", max_samples=10, batch_size=4)
    assert cfg.stage == "analyze"
    assert cfg.split == "validation"
    assert cfg.max_samples == 10
    assert cfg.batch_size == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": "bogus"},
        {"split": "bogus"},
        {"device": "bogus"},
        {"batch_size": 0},
        {"max_samples": 0},
    ],
)
def test_invalid_values_fail_closed(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        resolve_eval_config(**kwargs)
