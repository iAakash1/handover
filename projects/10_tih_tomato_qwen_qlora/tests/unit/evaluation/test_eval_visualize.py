"""Figure generation: files exist, are non-trivially sized, PNG+SVG both written."""

from __future__ import annotations

from pathlib import Path

import pytest

from plantdx.evaluation.visualize import plot_confusion_matrix, plot_grouped_comparison

pytest.importorskip("matplotlib", reason="matplotlib not installed (make install-eval)")
pytestmark = pytest.mark.requires_eval_stack


@pytest.mark.unit
def test_plot_grouped_comparison_writes_png_and_svg(tmp_path: Path) -> None:
    png, svg = plot_grouped_comparison(
        ["a", "b"],
        [0.5, 0.6],
        [0.7, 0.8],
        title="t",
        ylabel="y",
        out_dir=tmp_path,
        filename="chart",
    )
    assert png.is_file()
    assert svg.is_file()
    assert png.stat().st_size > 1000
    assert svg.stat().st_size > 500


@pytest.mark.unit
def test_plot_confusion_matrix_writes_png_and_svg(tmp_path: Path) -> None:
    png, svg = plot_confusion_matrix(
        [[5, 1], [0, 4]],
        ["a", "b"],
        normalized=False,
        title="cm",
        out_dir=tmp_path,
        filename="cm",
    )
    assert png.is_file()
    assert svg.is_file()


@pytest.mark.unit
def test_plot_grouped_comparison_single_category(tmp_path: Path) -> None:
    png, svg = plot_grouped_comparison(
        ["a"], [0.5], [0.7], title="t", ylabel="y", out_dir=tmp_path, filename="single"
    )
    assert png.is_file()
    assert svg.is_file()
