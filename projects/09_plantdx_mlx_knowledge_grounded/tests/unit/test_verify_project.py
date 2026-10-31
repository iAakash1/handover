"""Tests for the helpers of scripts/verify_project.py (the handover verification script)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_project.py"
_spec = importlib.util.spec_from_file_location("verify_project", _SCRIPT)
assert _spec is not None and _spec.loader is not None
verify_project = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_project)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("/Users/aakashjawle/Desktop/experiments/datasets/tomato/processed/healthy/a.JPG",
         "datasets/tomato/processed/healthy/a.JPG"),
        (r"C:\work\p09\datasets\mango\processed\die_back\b.jpg", "datasets/mango/processed/die_back/b.jpg"),
        ("relative/only.jpg", None),
    ],
)
def test_relative_to_datasets_strips_the_machine_prefix(image: str, expected: str | None) -> None:
    assert verify_project.relative_to_datasets(image) == expected


@pytest.mark.unit
def test_markdown_links_skips_urls_anchors_and_code(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(
        "[a](docs/A.md) ![img](fig/x%20y.png) [web](https://example.org) [top](#intro)\n"
        "[sec](docs/B.md#part)\n```\n[not a link](inside/code.md)\n```\n",
        encoding="utf-8",
    )
    assert verify_project.markdown_links(md) == ["docs/A.md", "fig/x y.png", "docs/B.md"]


@pytest.mark.unit
def test_every_run_has_a_model_card_and_config() -> None:
    root = _SCRIPT.parents[1]
    for run, (_crop, config, counts, raw) in verify_project.RUNS.items():
        assert (root / "models" / run / "README.md").is_file()
        assert (root / config).is_file()
        assert (root / raw).is_dir()
        assert len(counts) == 3
