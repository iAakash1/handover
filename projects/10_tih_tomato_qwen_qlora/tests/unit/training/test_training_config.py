"""Composed training-config loading + fail-closed validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plantdx.core.exceptions import ConfigError
from plantdx.training.config import load_training_config

_ROOT = Path(__file__).resolve().parents[3]
_TRAIN_CFG = _ROOT / "configs" / "train" / "qwen25vl_tomato.yaml"


@pytest.mark.unit
def test_real_config_composes_and_validates() -> None:
    cfg = load_training_config(_TRAIN_CFG, base_dir=_ROOT)
    assert cfg.data.crop == "tomato"
    assert len(cfg.data.classes) == 10
    assert cfg.model.model_path == "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    assert cfg.model.quantization_bits == 4
    assert cfg.lora.method == "qlora"
    assert cfg.optim.batch_size == 1


@pytest.mark.unit
def test_lora_reference_resolves_to_file(tmp_path: Path) -> None:
    cfg = load_training_config(_TRAIN_CFG, base_dir=_ROOT)
    # qlora.yaml supplied rank/alpha/dropout
    assert cfg.lora.rank == 16
    assert cfg.lora.alpha == 32


def _write(tmp_path: Path, overrides: dict) -> Path:
    base = yaml.safe_load(_TRAIN_CFG.read_text())
    base.update(overrides)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return p


@pytest.mark.unit
def test_ratios_must_sum_to_one(tmp_path: Path) -> None:
    base = yaml.safe_load(_TRAIN_CFG.read_text())
    base["data"]["train_ratio"] = 0.8
    base["data"]["val_ratio"] = 0.3  # sums to 1.15
    base["data"]["test_ratio"] = 0.05
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        load_training_config(p, base_dir=_ROOT)


@pytest.mark.unit
def test_qlora_requires_4bit(tmp_path: Path) -> None:
    base = yaml.safe_load(_TRAIN_CFG.read_text())
    base["model_override"] = {"quantization_bits": 8}
    p = tmp_path / "cfg8.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ConfigError, match="qlora requires a 4-bit"):
        load_training_config(p, base_dir=_ROOT)


@pytest.mark.unit
def test_dora_accepted_at_config_level(tmp_path: Path) -> None:
    # dora is a valid config value (forward-compat); the *backend* guard rejects it.
    p = _write(tmp_path, {"lora": "dora"})
    cfg = load_training_config(p, base_dir=_ROOT)
    assert cfg.lora.method == "dora"


@pytest.mark.unit
def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_training_config("configs/train/does_not_exist.yaml", base_dir=_ROOT)
