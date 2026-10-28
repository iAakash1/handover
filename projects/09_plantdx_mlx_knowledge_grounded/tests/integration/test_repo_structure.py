"""Integration: the repository/config scaffold is well-formed and self-consistent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_FILES = [
    "config.yaml",
    "paths.yaml",
    "generation.yaml",
    "validation.yaml",
    "training.yaml",
]


@pytest.mark.integration
@pytest.mark.parametrize("name", CONFIG_FILES)
def test_config_files_exist_and_parse(configs_dir: Path, name: str) -> None:
    path = configs_dir / name
    assert path.is_file(), f"missing config: {name}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and data


@pytest.mark.integration
def test_master_config_includes_the_others(configs_dir: Path) -> None:
    master = yaml.safe_load((configs_dir / "config.yaml").read_text(encoding="utf-8"))
    includes = set(master.get("includes", []))
    assert {"paths.yaml", "generation.yaml", "validation.yaml", "training.yaml"} <= includes


@pytest.mark.integration
def test_paths_point_at_existing_datasets(configs_dir: Path, repo_root: Path) -> None:
    paths = yaml.safe_load((configs_dir / "paths.yaml").read_text(encoding="utf-8"))
    for crop in ("tomato", "mango"):
        root = repo_root / paths["paths"]["datasets"][crop]["root"]
        # The dataset root directory should exist even if images are gitignored.
        assert root.parent.is_dir(), f"dataset parent missing for {crop}: {root}"


@pytest.mark.integration
def test_package_tree_present(repo_root: Path) -> None:
    pkg = repo_root / "src" / "plantdx"
    for sub in (
        "core",
        "config",
        "knowledge_base",
        "ontology",
        "vocabulary",
        "generation",
        "validation",
        "diversity",
        "dataset",
        "qa",
        "training",
        "evaluation",
        "utils",
    ):
        assert (pkg / sub / "__init__.py").is_file(), f"missing package: {sub}"
    assert (pkg / "py.typed").is_file()


@pytest.mark.integration
@pytest.mark.requires_dkb
def test_dkb_present_and_has_18_classes(has_dkb: bool, dkb_path: Path) -> None:
    if not has_dkb:
        pytest.skip("DKB not present in this checkout")
    import json

    dkb = json.loads(dkb_path.read_text(encoding="utf-8"))
    assert len(dkb["diseases"]) == 18
