"""Command builder: exact flags, toggles, resume, and DoRA fail-closed."""

from __future__ import annotations

import dataclasses

import pytest

from plantdx.core.exceptions import ConfigError
from plantdx.training.command import build_command, render_command
from plantdx.training.config import TrainingConfig


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.unit
def test_core_flags_present(training_config: TrainingConfig) -> None:
    argv = build_command(
        training_config, dataset_dir="/ds", output_path="/out/adapters.safetensors"
    )
    assert argv[1:3] == ["-m", "mlx_vlm.lora"]
    assert _flag_value(argv, "--model-path") == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    assert _flag_value(argv, "--dataset") == "/ds"
    assert _flag_value(argv, "--split") == "train"
    assert _flag_value(argv, "--train-mode") == "sft"
    assert _flag_value(argv, "--lora-rank") == "16"
    assert _flag_value(argv, "--learning-rate") == "0.0001"
    assert _flag_value(argv, "--batch-size") == "1"
    assert _flag_value(argv, "--assistant-id") == "77091"


@pytest.mark.unit
def test_toggles_and_resize(training_config: TrainingConfig) -> None:
    argv = build_command(training_config, dataset_dir="/ds", output_path="/o")
    assert "--grad-checkpoint" in argv
    assert "--train-on-completions" in argv
    assert "--train-vision" not in argv  # train_vision False
    idx = argv.index("--image-resize-shape")
    assert argv[idx + 1 : idx + 3] == ["448", "448"]


@pytest.mark.unit
def test_resume_adds_adapter_path(training_config: TrainingConfig) -> None:
    argv = build_command(
        training_config,
        dataset_dir="/ds",
        output_path="/o",
        resume_adapter="/o/adapters.safetensors",
    )
    assert _flag_value(argv, "--adapter-path") == "/o/adapters.safetensors"


@pytest.mark.unit
def test_dora_fails_closed(training_config: TrainingConfig) -> None:
    cfg = dataclasses.replace(
        training_config, lora=dataclasses.replace(training_config.lora, method="dora")
    )
    with pytest.raises(ConfigError, match="not supported"):
        build_command(cfg, dataset_dir="/ds", output_path="/o")


@pytest.mark.unit
def test_render_is_shell_safe(training_config: TrainingConfig) -> None:
    argv = build_command(training_config, dataset_dir="/a b/ds", output_path="/o")
    rendered = render_command(argv)
    assert "'/a b/ds'" in rendered  # spaces are quoted
