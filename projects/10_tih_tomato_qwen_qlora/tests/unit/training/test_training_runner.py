"""Runner preparation writes everything and NEVER launches training."""

from __future__ import annotations

from pathlib import Path

import pytest

from plantdx.training.runner import prepare_run

_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_prepare_run_builds_everything_without_launching(inline_cfg: Path, tmp_path: Path) -> None:
    prepared = prepare_run(inline_cfg, base_dir=_ROOT, dataset_dir=tmp_path / "ds")

    # dataset written
    assert (prepared.dataset_dir / "train.jsonl").exists()
    assert (prepared.dataset_dir / "manifest.json").exists()
    # report written
    assert Path(prepared.report_paths["markdown"]).exists()
    assert Path(prepared.report_paths["json"]).exists()
    # command targets mlx_vlm.lora and the built dataset
    assert "mlx_vlm.lora" in prepared.argv
    assert str(prepared.dataset_dir) in prepared.argv
    # checkpoint root created, but NO adapter exists (nothing trained)
    assert prepared.layout.root.is_dir()
    assert not prepared.layout.adapter_path.exists()


@pytest.mark.unit
def test_report_contains_the_one_command(inline_cfg: Path, tmp_path: Path) -> None:
    prepared = prepare_run(inline_cfg, base_dir=_ROOT, dataset_dir=tmp_path / "ds")
    md = Path(prepared.report_paths["markdown"]).read_text()
    assert "The one command to run" in md
    assert "mlx_vlm.lora" in md
    assert "No training has been run" in md
