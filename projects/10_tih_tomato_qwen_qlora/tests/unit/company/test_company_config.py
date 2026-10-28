"""Company training config: composition, fail-closed validation, overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plantdx.company.config import apply_overrides, load_company_train_config
from plantdx.company.train import planned_steps
from plantdx.core.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_CONFIG = REPO_ROOT / "configs" / "train" / "qwen25vl_tih_tomato.yaml"


@pytest.mark.unit
def test_shipped_config_loads_and_validates() -> None:
    cfg = load_company_train_config(REAL_CONFIG)
    assert cfg.run_name == "tih_tomato_qwen_qlora"
    assert cfg.model.model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert cfg.model.load_in_4bit is True
    assert cfg.lora.rank == 16
    assert cfg.lora.alpha == 32
    assert cfg.lora.train_vision is False
    assert cfg.optim.batch_size == 1
    assert cfg.optim.effective_batch_size == 8
    assert cfg.optim.grad_checkpoint is True


@pytest.mark.unit
def test_shipped_config_points_at_an_isolated_company_dataset_dir() -> None:
    """The company run must not write into the research run's directories."""
    cfg = load_company_train_config(REAL_CONFIG)
    research = load_company_train_config(REAL_CONFIG).base_dir / "checkpoints"
    assert cfg.checkpoint_dir.parent == research
    assert cfg.checkpoint_dir.name == "tih_tomato_qwen_qlora"
    assert cfg.checkpoint_dir.name != "qwen25vl_tomato_qlora"
    assert "company" in cfg.dataset_dir.as_posix()


@pytest.mark.unit
def test_shipped_config_uses_the_company_class_profile() -> None:
    cfg = load_company_train_config(REAL_CONFIG)
    assert cfg.data.profile == "configs/company_tomato_classes.yaml"


@pytest.mark.unit
def test_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_company_train_config(tmp_path / "absent.yaml")


def _write(tmp_path: Path, mutate: dict[str, object]) -> Path:
    base = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))
    for dotted, value in mutate.items():
        section, key = dotted.split(".", 1)
        base[section][key] = value
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"optim.learning_rate": 0}, "learning_rate must be positive"),
        ({"optim.epochs": 0}, "epochs must be positive"),
        ({"optim.batch_size": 0}, "must be positive"),
        ({"optim.scheduler": "triangular"}, "scheduler must be one of"),
        ({"optim.warmup_ratio": 1.0}, "warmup_ratio must be in"),
        ({"lora.rank": 0}, "rank and alpha must be positive"),
        ({"lora.dropout": 1.5}, "dropout must be in"),
        ({"lora.target_modules": []}, "target_modules must not be empty"),
        ({"model.image_max_side": 450}, "multiple of 28"),
        ({"model.compute_dtype": "float64"}, "compute_dtype must be one of"),
        ({"model.bnb_4bit_quant_type": "int8"}, "bnb_4bit_quant_type must be one of"),
        ({"data.train_ratio": 0.5}, "sum to 1.0"),
        ({"data.instruction": "  "}, "instruction must not be empty"),
    ],
)
def test_invalid_values_fail_closed(tmp_path: Path, mutation: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_company_train_config(_write(tmp_path, mutation))


@pytest.mark.unit
def test_overrides_apply_to_the_right_fields() -> None:
    cfg = load_company_train_config(REAL_CONFIG)
    updated = apply_overrides(
        cfg,
        {
            "epochs": 5,
            "learning_rate": 2e-5,
            "lora_rank": 32,
            "lora_alpha": 64,
            "gradient_accumulation_steps": 4,
            "image_max_side": 336,
            "seed": 99,
        },
    )
    assert updated.optim.epochs == 5
    assert updated.optim.learning_rate == pytest.approx(2e-5)
    assert updated.lora.rank == 32
    assert updated.lora.alpha == 64
    assert updated.optim.effective_batch_size == 4
    assert updated.model.image_max_side == 336
    assert updated.seed == 99
    # the original is untouched
    assert cfg.optim.epochs == 3


@pytest.mark.unit
def test_unknown_override_is_rejected() -> None:
    cfg = load_company_train_config(REAL_CONFIG)
    with pytest.raises(ConfigError, match="unknown override"):
        apply_overrides(cfg, {"momentum": 0.9})


@pytest.mark.unit
def test_run_name_override_retargets_every_output_path() -> None:
    cfg = load_company_train_config(REAL_CONFIG)
    renamed = apply_overrides(cfg, {"run_name": "tih_tomato_experiment_b"})
    assert renamed.run_name == "tih_tomato_experiment_b"
    for path in (
        renamed.checkpoint_dir,
        renamed.log_dir,
        renamed.report_dir,
        renamed.dataset_dir,
    ):
        assert "tih_tomato_experiment_b" in path.as_posix()
        assert "tih_tomato_qwen_qlora" not in path.as_posix()


@pytest.mark.unit
def test_run_name_override_cannot_be_blank() -> None:
    cfg = load_company_train_config(REAL_CONFIG)
    with pytest.raises(ConfigError, match="must not be empty"):
        apply_overrides(cfg, {"run_name": "   "})


@pytest.mark.unit
def test_paths_resolve_independently_of_the_working_directory(tmp_path: Path) -> None:
    """The config must not depend on the shell's current directory."""
    cfg = load_company_train_config(REAL_CONFIG)
    assert cfg.checkpoint_dir.is_absolute()
    assert cfg.dataset_dir.is_absolute()
    assert cfg.base_dir == REPO_ROOT


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rows", "accum", "epochs", "expected"),
    [(5247, 8, 3, 1968), (40, 8, 1, 5), (1, 8, 1, 1), (16, 8, 2, 4)],
)
def test_planned_steps_matches_the_trainer_schedule(
    rows: int, accum: int, epochs: float, expected: int
) -> None:
    assert planned_steps(rows, batch_size=1, accumulation=accum, epochs=epochs) == expected
