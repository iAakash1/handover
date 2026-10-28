"""Data pipeline: discovery, pairing, splits, determinism, mlx-vlm row shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plantdx.core.exceptions import DerivationError
from plantdx.training.config import DataConfig
from plantdx.training.data import build_training_dataset, discover_images, load_label_map
from plantdx.training.data.splits import assign_splits


@pytest.mark.unit
def test_label_map_is_tomato_only() -> None:
    root = Path(__file__).resolve().parents[3]
    mapping = load_label_map("tomato", asset_path=root / "assets" / "metadata" / "label_map.json")
    assert len(mapping) == 10
    assert mapping["early_blight"] == "tomato_early_blight"
    assert all(v.startswith("tomato_") for v in mapping.values())


@pytest.mark.unit
def test_discovery_missing_tree_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(DerivationError, match="plantdx normalize"):
        discover_images(tmp_path / "nope", "tomato", {"healthy": "tomato_healthy"})


@pytest.mark.unit
def test_build_is_deterministic_and_grouped(data_config: DataConfig, tmp_path: Path) -> None:
    out1 = tmp_path / "o1"
    out2 = tmp_path / "o2"
    s1 = build_training_dataset(data_config, output_dir=out1)
    build_training_dataset(data_config, output_dir=out2)

    # byte-identical across builds
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()

    # 10 classes x 6 images = 60 images, x2 captions = 120 rows
    assert s1.image_count == 60
    assert s1.row_count == 120
    assert sum(s1.per_split.values()) == 120

    # image-grouped: no image id appears in more than one split
    def ids(name: str) -> set[str]:
        out = set()
        for line in (out1 / name).read_text().splitlines():
            path = json.loads(line)["image"]
            out.add("/".join(path.split("/")[-2:]))
        return out

    tr, va, te = ids("train.jsonl"), ids("validation.jsonl"), ids("test.jsonl")
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 60


@pytest.mark.unit
def test_rows_have_mlx_vlm_shape(data_config: DataConfig, tmp_path: Path) -> None:
    out = tmp_path / "o"
    build_training_dataset(data_config, output_dir=out)
    row = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert set(row.keys()) == {"image", "question", "answer"}
    assert row["image"].endswith(".JPG")
    assert row["answer"].startswith("This tomato leaf shows")


@pytest.mark.unit
def test_corpus_checksum_pinned(data_config: DataConfig, tmp_path: Path) -> None:
    stats = build_training_dataset(data_config, output_dir=tmp_path / "o")
    assert stats.corpus_checksum == "sha256:deadbeef"


@pytest.mark.unit
def test_splits_are_stratified(data_config: DataConfig, tmp_path: Path) -> None:
    imgs = discover_images(
        data_config.processed_dir,
        data_config.crop,
        load_label_map(
            "tomato",
            asset_path=Path(__file__).resolve().parents[3]
            / "assets"
            / "metadata"
            / "label_map.json",
        ),
    )
    split_of = assign_splits(imgs, seed=1, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    # each disease has 6 images -> 3 train / 1 val / 2 test (int flooring)
    by_disease: dict[str, dict[str, int]] = {}
    for it in imgs:
        d = by_disease.setdefault(it.disease_id, {"train": 0, "validation": 0, "test": 0})
        d[split_of[it.image_id]] += 1
    for counts in by_disease.values():
        assert counts["train"] == 3
        assert counts["validation"] == 1
        assert counts["test"] == 2


@pytest.mark.unit
def test_messages_format_has_no_image_marker(data_config: DataConfig, tmp_path: Path) -> None:
    import dataclasses

    cfg = dataclasses.replace(data_config, response_format="messages")
    out = tmp_path / "o"
    build_training_dataset(cfg, output_dir=out)
    row = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert "messages" in row
    assert "<image>" not in row["messages"][0]["content"]
