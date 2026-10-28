"""Dataset and image discovery.

Datasets and their roots come from the configuration (``paths.datasets``); no
disease/class names are hardcoded. A class is defined as the immediate parent
directory of each image file, so both flat layouts (``root/Class/img.jpg``) and
nested ones (``root/split/Class/img.jpg``) are handled the same way. Traversal is
sorted for determinism.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from plantdx.config.schema import PlantDxConfig


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset to audit, resolved from the configuration."""

    key: str  # config key, e.g. "tomato"
    name: str  # human name, e.g. "PlantVillage"
    root: Path  # absolute path to the dataset root
    configured_classes: int | None  # expected class count from config, if any


def build_specs(config: PlantDxConfig, base_dir: str | Path = ".") -> list[DatasetSpec]:
    """Build the list of :class:`DatasetSpec` from the configuration."""
    base = Path(base_dir)
    specs = []
    for key, dataset in config.paths.datasets.items():
        specs.append(
            DatasetSpec(
                key=key,
                name=dataset.name,
                root=(base / dataset.root).resolve(),
                configured_classes=dataset.classes,
            )
        )
    return specs


def _is_hidden(path: Path) -> bool:
    """Whether a path is a hidden/system entry (``.DS_Store``, ``._foo``)."""
    return path.name.startswith(".")


def discover_images(
    root: Path, supported_extensions: Iterable[str]
) -> tuple[list[tuple[Path, str]], list[Path]]:
    """Discover image files under ``root``.

    Returns ``(images, unsupported)`` where ``images`` is a sorted list of
    ``(path, class_name)`` pairs (class = the file's parent directory name) and
    ``unsupported`` lists non-hidden files whose extension is not supported.
    """
    supported = {ext.lower() for ext in supported_extensions}
    images: list[tuple[Path, str]] = []
    unsupported: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_hidden(path):
            continue
        if path.suffix.lower() in supported:
            images.append((path, path.parent.name))
        else:
            unsupported.append(path)
    return images, unsupported


def find_dir_issues(root: Path) -> tuple[list[Path], list[Path]]:
    """Find directory problems under ``root``.

    Returns ``(empty_dirs, unexpected_dirs)``: directories with no visible
    contents, and hidden/system directories (``__MACOSX`` and dot-directories).
    """
    empty: list[Path] = []
    unexpected: list[Path] = []
    for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
        if _is_hidden(directory) or directory.name == "__MACOSX":
            unexpected.append(directory)
            continue
        has_visible_child = any(not _is_hidden(child) for child in directory.iterdir())
        if not has_visible_child:
            empty.append(directory)
    return empty, unexpected
