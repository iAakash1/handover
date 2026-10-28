"""Fixtures for the company (TIH) training-path tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

CLASSES = [
    ("Aphid", "aphid", "Aphid"),
    ("Late_Blight", "late_blight", "Late Blight"),
    ("Healthy_Leaf", "healthy_leaf", "Healthy Leaf"),
]


def make_image(path: Path, seed: int, size: tuple[int, int] = (64, 48)) -> None:
    """Write a tiny JPEG whose pixels are unique to ``seed``.

    Solid colors are not enough: adjacent RGB values quantize to identical JPEG
    bytes, which would make distinct fixture images look like duplicates.
    """
    import random

    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    image = Image.new("RGB", size)
    pixels = [
        (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for _ in range(size[0] * size[1])
    ]
    image.putdata(pixels)
    image.save(path, format="JPEG", quality=95)


@pytest.fixture
def raw_dataset(tmp_path: Path) -> Path:
    """A miniature company-shaped dataset with mixed-case extensions."""
    root = tmp_path / "Dataset"
    for index, (raw, _cls, _label) in enumerate(CLASSES):
        for n in range(6):
            suffix = [".Jpg", ".jpg", ".JPEG", ".png", ".webp", ".jpeg"][n % 6]
            make_image(root / raw / f"img_{n}{suffix}", seed=100 * index + n)
    return root


@pytest.fixture
def profile_path(tmp_path: Path, raw_dataset: Path) -> Path:
    """A class profile pointing at ``raw_dataset``."""
    payload = {
        "dataset": {
            "name": "Test_Tomato",
            "crop": "tomato",
            "root": str(raw_dataset),
            "candidate_roots": [],
            "extensions": [".jpg", ".jpeg", ".png", ".webp"],
        },
        "classes": [
            {"raw": raw, "class_id": cls, "label": label}
            for raw, cls, label in CLASSES
        ],
    }
    path = tmp_path / "classes.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def train_config_path(tmp_path: Path, profile_path: Path) -> Path:
    """A company training config wired to the miniature dataset."""
    payload = {
        "run_name": "test_run",
        "seed": 7,
        "model": {
            "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "compute_dtype": "bfloat16",
            "image_max_side": 224,
            "max_seq_length": 512,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "train_vision": False,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "optim": {
            "learning_rate": 5.0e-5,
            "epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "weight_decay": 0.0,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
            "grad_clip": 1.0,
            "grad_checkpoint": True,
            "optimizer": "paged_adamw_8bit",
        },
        "data": {
            "profile": str(profile_path),
            "instruction": "Identify the disease or condition in this tomato plant image.",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "split_seed": 7,
            "deduplicate": True,
        },
        "evaluation": {"max_new_tokens": 24, "limit": None},
        "output": {
            "dataset_dir": str(tmp_path / "out" / "dataset"),
            "checkpoint_dir": str(tmp_path / "out" / "checkpoints"),
            "log_dir": str(tmp_path / "out" / "logs"),
            "report_dir": str(tmp_path / "out" / "reports"),
        },
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def company_dataset(profile_path: Path):
    """The loaded company dataset profile for the miniature dataset."""
    from plantdx.company.classes import load_company_dataset

    return load_company_dataset(profile_path)


@pytest.fixture
def company_config(train_config_path: Path):
    """The loaded company training config for the miniature dataset."""
    from plantdx.company.config import load_company_train_config

    return load_company_train_config(train_config_path)
