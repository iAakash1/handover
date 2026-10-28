"""Publication-quality figures (matplotlib only), each exported as PNG + SVG.

Color follows a small, CVD-validated design system (fixed categorical hue
order; one-hue sequential ramp for the confusion-matrix heatmap) rather than
matplotlib's defaults -- see the module-level constants below for the exact,
pre-validated hex values. Every figure is a grouped bar chart or a heatmap;
never a dual-axis chart, never a pie chart, never a rainbow colormap.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Fixed categorical order (CVD-validated, ΔE >= 12 target); base/fine-tuned use
# slots 1 (blue) and 6 (red) -- not adjacent, and read intuitively as "before"
# vs "after" via the blue<->red diverging convention.
_COLOR_BASE = "#2a78d6"
_COLOR_FINETUNED = "#e34948"
_COLOR_TEXT = "#0b0b0b"
_COLOR_GRID = "#d8d7d2"  # recessive gridlines
# One-hue sequential ramp (blue, light -> dark) for the confusion-matrix heatmap.
_SEQUENTIAL_BLUE = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)


def _apply_base_style(ax: object) -> None:
    ax.spines["top"].set_visible(False)  # type: ignore[attr-defined]
    ax.spines["right"].set_visible(False)  # type: ignore[attr-defined]
    ax.spines["left"].set_color(_COLOR_GRID)  # type: ignore[attr-defined]
    ax.spines["bottom"].set_color(_COLOR_GRID)  # type: ignore[attr-defined]
    ax.tick_params(colors=_COLOR_TEXT)  # type: ignore[attr-defined]
    ax.yaxis.grid(True, color=_COLOR_GRID, linewidth=0.8, zorder=0)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]


def _save(fig: object, out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")  # type: ignore[attr-defined]
    fig.savefig(svg_path, bbox_inches="tight")  # type: ignore[attr-defined]
    return png_path, svg_path


def plot_grouped_comparison(
    categories: Sequence[str],
    base_values: Sequence[float],
    finetuned_values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    out_dir: str | Path,
    filename: str,
) -> tuple[Path, Path]:
    """A base-vs-fine-tuned grouped bar chart across `categories`.

    Used for the metric comparison, per-disease F1, hallucination comparison,
    response-length, and latency figures -- all "compare two series across N
    categories" charts share this one implementation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(categories))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(categories)), 4.5))
    bars_base = ax.bar(x - width / 2, base_values, width, label="Base", color=_COLOR_BASE, zorder=3)
    bars_ft = ax.bar(
        x + width / 2, finetuned_values, width, label="Fine-tuned", color=_COLOR_FINETUNED, zorder=3
    )
    _apply_base_style(ax)
    ax.set_title(title, color=_COLOR_TEXT, fontsize=12, fontweight="bold", pad=28)
    ax.set_ylabel(ylabel, color=_COLOR_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", color=_COLOR_TEXT)
    # Legend outside the plotting area (above, centered) so it never collides
    # with a bar label regardless of data shape -- a plot-area legend
    # (e.g. "upper left") can overlap a tall bar's value label.
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0)
    ax.margins(y=0.12)  # headroom so the tallest bar's label never clips the top spine
    for bars in (bars_base, bars_ft):
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, color=_COLOR_TEXT)
    fig.tight_layout()

    paths = _save(fig, Path(out_dir), filename)
    plt.close(fig)
    return paths


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    *,
    normalized: bool,
    title: str,
    out_dir: str | Path,
    filename: str,
) -> tuple[Path, Path]:
    """A confusion-matrix heatmap using the one-hue sequential blue ramp."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("plantdx_sequential_blue", _SEQUENTIAL_BLUE)
    data = np.asarray(matrix, dtype=float)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * n), max(5.0, 0.6 * n)))
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=_COLOR_TEXT, fontsize=8)
    ax.set_yticklabels(labels, color=_COLOR_TEXT, fontsize=8)
    ax.set_xlabel("Predicted", color=_COLOR_TEXT)
    ax.set_ylabel("True", color=_COLOR_TEXT)
    ax.set_title(title, color=_COLOR_TEXT, fontsize=12, fontweight="bold")

    threshold = data.max() / 2 if data.max() > 0 else 0.5
    fmt = "{:.2f}" if normalized else "{:.0f}"
    for i in range(n):
        for j in range(n):
            value = data[i, j]
            text_color = "white" if value > threshold else _COLOR_TEXT
            ax.text(j, i, fmt.format(value), ha="center", va="center", color=text_color, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    paths = _save(fig, Path(out_dir), filename)
    plt.close(fig)
    return paths
