"""Fixtures for audit tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Allow `import _dataset` from this directory regardless of the runner's CWD.
sys.path.insert(0, str(Path(__file__).parent))
from _dataset import build_sample_dataset


@pytest.fixture
def sample_dataset(tmp_path: Path) -> dict[str, Any]:
    """Create the synthetic dataset and return its expected counts."""
    return build_sample_dataset(tmp_path / "ds")
