"""Shared pytest fixtures and path helpers for the PlantDx test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    """Path to the ``configs/`` directory."""
    return repo_root / "configs"


@pytest.fixture(scope="session")
def dkb_path(repo_root: Path) -> Path:
    """Path to the frozen Disease Knowledge Base JSON."""
    return repo_root / "knowledge_base" / "dkb.json"


@pytest.fixture(scope="session")
def has_dkb(dkb_path: Path) -> bool:
    """Whether the DKB file is present (guards ``requires_dkb`` tests)."""
    return dkb_path.is_file()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests backed by unpublished knowledge inputs in public checkouts."""
    del config
    if (REPO_ROOT / "knowledge_base" / "dkb.json").is_file():
        return
    missing_private_input = pytest.mark.skip(
        reason="requires unpublished knowledge inputs excluded from the public release"
    )
    for item in items:
        if "requires_dkb" in item.keywords:
            item.add_marker(missing_private_input)
