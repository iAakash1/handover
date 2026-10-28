"""Modeling helpers that need no GPU: resizing, row loading, adapter validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.company.modeling import (
    PATCH,
    JsonlRows,
    ModelingError,
    _common_prefix,
    adapter_is_valid,
    language_model_targets,
    resize_image,
    torch_dtype,
    vision_targets,
)

from .conftest import make_image


class _FakeLinear:
    """Stand-in for ``torch.nn.Linear`` with the class name bitsandbytes uses."""

    def __init__(self, name: str) -> None:
        self.__class__ = type("Linear4bit", (_FakeLinear,), {})
        self.name = name


class _FakeModel:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def named_modules(self):
        for name in self._names:
            yield name, _FakeLinear(name)


QWEN_MODULES = [
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.gate_proj",
    "model.visual.blocks.0.mlp.up_proj",
    "model.visual.blocks.0.mlp.down_proj",
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.0.self_attn.k_proj",
    "model.language_model.layers.0.self_attn.v_proj",
    "model.language_model.layers.0.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
]


@pytest.mark.unit
def test_lora_targets_exclude_the_vision_tower() -> None:
    """Suffix matching alone would adapt the visual MLP; full names must not."""
    targets = language_model_targets(
        _FakeModel(QWEN_MODULES), ("q_proj", "k_proj", "v_proj", "o_proj")
    )
    assert targets == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.0.self_attn.o_proj",
    ]
    assert not any("visual" in name for name in targets)


@pytest.mark.unit
def test_mlp_suffixes_never_leak_into_the_vision_tower() -> None:
    targets = language_model_targets(
        _FakeModel(QWEN_MODULES), ("gate_proj", "up_proj", "down_proj")
    )
    assert all(name.startswith("model.language_model") for name in targets)
    assert len(targets) == 3


@pytest.mark.unit
def test_vision_targets_select_only_the_vision_tower() -> None:
    targets = vision_targets(_FakeModel(QWEN_MODULES), ("qkv", "proj"))
    assert targets == [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("size", "max_side"),
    [((1064, 2304), 448), ((4624, 3468), 448), ((100, 100), 448), ((6144, 8160), 336)],
)
def test_resize_bounds_the_longest_side_and_snaps_to_the_patch_grid(
    size: tuple[int, int], max_side: int
) -> None:
    from PIL import Image

    resized = resize_image(Image.new("RGB", size), max_side)
    width, height = resized.size
    assert width % PATCH == 0 and height % PATCH == 0
    assert max(width, height) <= max_side + PATCH
    # aspect ratio preserved within one patch of rounding
    assert abs(width / height - size[0] / size[1]) < 0.15


@pytest.mark.unit
def test_resize_is_deterministic() -> None:
    from PIL import Image

    source = Image.new("RGB", (1000, 700))
    assert resize_image(source, 448).size == resize_image(source, 448).size


@pytest.mark.unit
@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1, 2, 3, 4], [1, 2, 3], 3),
        ([1, 2, 3], [1, 2, 3], 3),
        ([1, 9, 3], [1, 2, 3], 1),
        ([], [1], 0),
        ([5], [6], 0),
    ],
)
def test_common_prefix_length(a: list[int], b: list[int], expected: int) -> None:
    assert _common_prefix(a, b) == expected


@pytest.mark.unit
def test_jsonl_rows_reads_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps({"image": f"i{i}", "question": "q", "answer": "a"}) for i in range(5)),
        encoding="utf-8",
    )
    assert len(JsonlRows(path)) == 5
    assert len(JsonlRows(path, limit=2)) == 2
    assert JsonlRows(path)[0]["image"] == "i0"


@pytest.mark.unit
def test_missing_split_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ModelingError, match="dataset split not found"):
        JsonlRows(tmp_path / "absent.jsonl")


@pytest.mark.unit
def test_empty_split_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ModelingError, match="dataset split is empty"):
        JsonlRows(path)


@pytest.mark.unit
def test_adapter_validation_requires_both_files(tmp_path: Path) -> None:
    missing = tmp_path / "nothing"
    ok, reason = adapter_is_valid(missing)
    assert ok is False and "not found" in reason

    partial = tmp_path / "partial"
    partial.mkdir()
    partial.joinpath("adapter_config.json").write_text(
        '{"target_modules": ["q"]}', encoding="utf-8"
    )
    ok, reason = adapter_is_valid(partial)
    assert ok is False and "adapter_model.safetensors" in reason


@pytest.mark.unit
def test_adapter_validation_accepts_a_complete_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "good"
    adapter.mkdir()
    adapter.joinpath("adapter_config.json").write_text(
        json.dumps({"target_modules": ["q_proj"], "r": 16, "lora_alpha": 32}), encoding="utf-8"
    )
    adapter.joinpath("adapter_model.safetensors").write_bytes(b"weights")
    ok, reason = adapter_is_valid(adapter)
    assert ok is True and reason == ""


@pytest.mark.unit
def test_adapter_without_target_modules_is_rejected(tmp_path: Path) -> None:
    adapter = tmp_path / "empty_targets"
    adapter.mkdir()
    adapter.joinpath("adapter_config.json").write_text('{"target_modules": []}', encoding="utf-8")
    adapter.joinpath("adapter_model.safetensors").write_bytes(b"w")
    ok, reason = adapter_is_valid(adapter)
    assert ok is False and "target_modules" in reason


@pytest.mark.unit
def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(ModelingError, match="unsupported dtype"):
        torch_dtype("float8")


@pytest.mark.unit
def test_unreadable_image_is_a_clear_error(tmp_path: Path) -> None:
    from plantdx.company.modeling import load_image

    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    with pytest.raises(ModelingError, match="could not read image"):
        load_image(broken, 448)


@pytest.mark.unit
def test_load_image_resizes_a_real_file(tmp_path: Path) -> None:
    from plantdx.company.modeling import load_image

    path = tmp_path / "big.jpg"
    make_image(path, seed=3, size=(1400, 700))
    image = load_image(path, 448)
    assert max(image.size) <= 448 + PATCH
    assert image.mode == "RGB"
