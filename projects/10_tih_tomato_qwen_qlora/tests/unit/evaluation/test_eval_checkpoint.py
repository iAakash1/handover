"""Adapter checkpoint resolution: the directory mlx-vlm's LoRA loader needs.

Regression coverage for the bug where evaluation passed the checkpoint's
``adapters.safetensors`` FILE straight to ``mlx_vlm.load(..., adapter_path=...)``,
which requires a DIRECTORY containing both ``adapter_config.json`` and
``adapters.safetensors`` (mlx-vlm's own ``apply_lora_layers`` does
``open(Path(adapter_path) / "adapter_config.json")``) -- causing
``NotADirectoryError: .../adapters.safetensors/adapter_config.json``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plantdx.core.exceptions import DerivationError
from plantdx.evaluation.checkpoint import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    adapter_weights_path,
    resolve_adapter_dir,
    run_name_from_adapter_path,
)


@pytest.fixture
def valid_checkpoint_dir(tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoints" / "some_run"
    ckpt.mkdir(parents=True)
    (ckpt / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")
    (ckpt / ADAPTER_WEIGHTS_NAME).write_bytes(b"fake-weights")
    return ckpt


@pytest.mark.unit
def test_resolves_directory_form_directly(valid_checkpoint_dir: Path) -> None:
    assert resolve_adapter_dir(valid_checkpoint_dir) == valid_checkpoint_dir


@pytest.mark.unit
def test_resolves_weights_file_to_its_parent_directory(valid_checkpoint_dir: Path) -> None:
    """The exact reported-bug input: a path ending in adapters.safetensors must
    resolve to the checkpoint directory, not be passed through as-is."""
    weights_file = valid_checkpoint_dir / ADAPTER_WEIGHTS_NAME
    resolved = resolve_adapter_dir(weights_file)
    assert resolved == valid_checkpoint_dir
    assert resolved.is_dir()


@pytest.mark.unit
def test_missing_checkpoint_fails_closed_not_a_traceback(tmp_path: Path) -> None:
    with pytest.raises(DerivationError, match="not found"):
        resolve_adapter_dir(tmp_path / "no_such_run")


@pytest.mark.unit
def test_missing_config_is_a_friendly_error(tmp_path: Path) -> None:
    half_written = tmp_path / "half_written"
    half_written.mkdir()
    (half_written / ADAPTER_WEIGHTS_NAME).write_bytes(b"x")  # config missing
    with pytest.raises(DerivationError, match=r"adapter_config\.json"):
        resolve_adapter_dir(half_written)


@pytest.mark.unit
def test_missing_weights_is_a_friendly_error(tmp_path: Path) -> None:
    half_written = tmp_path / "half_written2"
    half_written.mkdir()
    (half_written / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")  # weights missing
    with pytest.raises(DerivationError, match=r"adapters\.safetensors"):
        resolve_adapter_dir(half_written)


@pytest.mark.unit
def test_adapter_weights_path(valid_checkpoint_dir: Path) -> None:
    assert adapter_weights_path(valid_checkpoint_dir) == valid_checkpoint_dir / ADAPTER_WEIGHTS_NAME


@pytest.mark.unit
def test_never_hardcodes_a_run_name(tmp_path: Path) -> None:
    """Same resolution logic must work for any run name (tomato, a future
    mango run, or anything else) -- nothing crop- or run-specific here."""
    for run_name in ("qwen25vl_tomato_qlora", "qwen25vl_mango_qlora", "some_other_run"):
        ckpt = tmp_path / run_name
        ckpt.mkdir()
        (ckpt / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")
        (ckpt / ADAPTER_WEIGHTS_NAME).write_bytes(b"x")
        assert resolve_adapter_dir(ckpt) == ckpt
        assert resolve_adapter_dir(ckpt / ADAPTER_WEIGHTS_NAME) == ckpt


@pytest.mark.unit
def test_run_name_from_adapter_path_directory_form() -> None:
    assert run_name_from_adapter_path("checkpoints/qwen25vl_mango_qlora") == "qwen25vl_mango_qlora"


@pytest.mark.unit
def test_run_name_from_adapter_path_weights_file_form() -> None:
    """Must resolve the same run name whether --adapter names the checkpoint
    directory or the adapters.safetensors file inside it, and requires no
    filesystem I/O (the checkpoint need not exist yet)."""
    path = "checkpoints/qwen25vl_mango_qlora/" + ADAPTER_WEIGHTS_NAME
    assert run_name_from_adapter_path(path) == "qwen25vl_mango_qlora"


@pytest.mark.unit
@pytest.mark.requires_dkb
def test_real_frozen_tomato_checkpoint_resolves() -> None:
    real = Path("checkpoints/qwen25vl_tomato_qlora")
    if not real.is_dir():
        pytest.skip("frozen tomato checkpoint not present in this checkout")
    resolved = resolve_adapter_dir(real)
    assert resolved == real
    assert adapter_weights_path(resolved).is_file()
    # The exact reported-bug path must ALSO resolve correctly (back-compat).
    assert resolve_adapter_dir(real / ADAPTER_WEIGHTS_NAME) == real
