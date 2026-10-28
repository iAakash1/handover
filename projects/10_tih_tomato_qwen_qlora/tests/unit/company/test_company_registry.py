"""Active-adapter registry: a pointer that never moves or overwrites a checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.company.registry import (
    ActiveAdapter,
    clear_active_adapter,
    read_active_adapter,
    read_registry,
    registry_path,
    write_active_adapter,
)
from plantdx.core.exceptions import ConfigError


def _entry(tmp_path: Path, run_name: str = "tih_tomato_qwen_qlora") -> ActiveAdapter:
    adapter = tmp_path / "checkpoints" / run_name
    adapter.mkdir(parents=True, exist_ok=True)
    return ActiveAdapter(
        crop="tomato",
        run_name=run_name,
        adapter_path=str(adapter),
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
        dataset="Tomato_12thAug26",
        backend="transformers_cuda",
        classes=("Aphid", "Late Blight"),
        metrics={"f1_macro": 0.91},
        activated_at="2026-09-11T10:00:00+00:00",
    )


@pytest.mark.unit
def test_empty_registry_reads_as_empty(tmp_path: Path) -> None:
    assert read_registry(tmp_path) == {}
    assert read_active_adapter("tomato", tmp_path) is None


@pytest.mark.unit
def test_write_then_read_round_trips(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    path = write_active_adapter(entry, tmp_path)
    assert path == registry_path(tmp_path)

    loaded = read_active_adapter("tomato", tmp_path)
    assert loaded is not None
    assert loaded.run_name == entry.run_name
    assert loaded.adapter_path == entry.adapter_path
    assert loaded.metrics == {"f1_macro": 0.91}
    assert loaded.classes == ("Aphid", "Late Blight")
    assert loaded.exists(tmp_path)


@pytest.mark.unit
def test_activating_stores_a_pointer_not_a_copy(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    (Path(entry.adapter_path) / "adapter_model.safetensors").write_bytes(b"x" * 1024)
    write_active_adapter(entry, tmp_path)

    payload = json.loads(registry_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["tomato"]["adapter_path"] == entry.adapter_path
    # nothing was copied next to the registry
    assert not (registry_path(tmp_path).parent / "adapter_model.safetensors").exists()


@pytest.mark.unit
def test_reactivating_replaces_only_that_crop(tmp_path: Path) -> None:
    write_active_adapter(_entry(tmp_path, "run_a"), tmp_path)
    write_active_adapter(_entry(tmp_path, "run_b"), tmp_path)
    registry = read_registry(tmp_path)
    assert set(registry) == {"tomato"}
    assert registry["tomato"].run_name == "run_b"


@pytest.mark.unit
def test_a_second_crop_is_kept_alongside(tmp_path: Path) -> None:
    write_active_adapter(_entry(tmp_path), tmp_path)
    other = ActiveAdapter(
        crop="mango",
        run_name="legacy_mango",
        adapter_path=str(tmp_path / "checkpoints" / "legacy_mango"),
        base_model="m",
        dataset="d",
        backend="mlx_vlm",
    )
    write_active_adapter(other, tmp_path)
    assert set(read_registry(tmp_path)) == {"tomato", "mango"}


@pytest.mark.unit
def test_clear_removes_only_the_named_crop(tmp_path: Path) -> None:
    write_active_adapter(_entry(tmp_path), tmp_path)
    assert clear_active_adapter("tomato", tmp_path) is True
    assert clear_active_adapter("tomato", tmp_path) is False
    assert read_registry(tmp_path) == {}


@pytest.mark.unit
def test_missing_adapter_directory_is_reported_not_raised(tmp_path: Path) -> None:
    entry = ActiveAdapter(
        crop="tomato",
        run_name="gone",
        adapter_path=str(tmp_path / "checkpoints" / "gone"),
        base_model="m",
        dataset="d",
        backend="transformers_cuda",
    )
    write_active_adapter(entry, tmp_path)
    loaded = read_active_adapter("tomato", tmp_path)
    assert loaded is not None
    assert loaded.exists(tmp_path) is False


@pytest.mark.unit
def test_corrupt_registry_is_a_clear_error(tmp_path: Path) -> None:
    path = registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="unreadable"):
        read_registry(tmp_path)


@pytest.mark.unit
def test_relative_adapter_path_resolves_against_the_base_dir(tmp_path: Path) -> None:
    (tmp_path / "checkpoints" / "rel").mkdir(parents=True)
    entry = ActiveAdapter(
        crop="tomato",
        run_name="rel",
        adapter_path="checkpoints/rel",
        base_model="m",
        dataset="d",
        backend="transformers_cuda",
    )
    write_active_adapter(entry, tmp_path)
    loaded = read_active_adapter("tomato", tmp_path)
    assert loaded is not None
    assert loaded.resolved_path(tmp_path) == tmp_path / "checkpoints" / "rel"
    assert loaded.exists(tmp_path)
