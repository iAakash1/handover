"""CLI: prepare-training, train --dry-run, DoRA guard, infer error path.

These never launch training: only ``--dry-run`` / ``prepare-training`` paths and
the mlx-absent inference error are exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plantdx.cli import main


@pytest.mark.unit
def test_prepare_training_cli(inline_cfg: Path, tmp_path: Path, capsys) -> None:
    code = main(
        ["prepare-training", "--config", str(inline_cfg), "--dataset-dir", str(tmp_path / "ds")]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "No training was started" in out
    assert "mlx_vlm.lora" in out


@pytest.mark.unit
def test_train_dry_run_does_not_launch(inline_cfg: Path, tmp_path: Path, capsys) -> None:
    code = main(
        ["train", "--config", str(inline_cfg), "--dataset-dir", str(tmp_path / "ds"), "--dry-run"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "no training started" in out.lower()


@pytest.mark.unit
def test_train_crop_mismatch_errors(inline_cfg: Path, tmp_path: Path, capsys) -> None:
    code = main(
        [
            "train",
            "--config",
            str(inline_cfg),
            "--crop",
            "mango",
            "--dataset-dir",
            str(tmp_path / "ds"),
            "--dry-run",
        ]
    )
    assert code == 1
    assert "does not match config crop" in capsys.readouterr().err


@pytest.mark.unit
def test_train_dora_fails_closed(inline_cfg: Path, tmp_path: Path, capsys) -> None:
    raw = yaml.safe_load(inline_cfg.read_text())
    raw["lora"]["method"] = "dora"
    p = tmp_path / "dora.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    code = main(["train", "--config", str(p), "--dataset-dir", str(tmp_path / "ds"), "--dry-run"])
    assert code == 1
    assert "not supported" in capsys.readouterr().err


@pytest.mark.unit
def test_infer_without_mlx_is_clean_error(tmp_path: Path, capsys) -> None:
    img = tmp_path / "leaf.JPG"
    img.write_bytes(b"")
    try:
        import mlx_vlm  # noqa: F401
    except Exception:
        code = main(["infer", "--image", str(img)])
        assert code == 1
        assert "mlx-vlm is not importable" in capsys.readouterr().err
    else:
        pytest.skip("mlx-vlm is importable here; error path not applicable")
