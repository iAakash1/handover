"""Fixtures for normalization tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _sample_raw_datasets import build_mango, build_plantvillage


@pytest.fixture
def plantvillage(tmp_path: Path) -> dict[str, Any]:
    return build_plantvillage(tmp_path / "pv")


@pytest.fixture
def mango(tmp_path: Path) -> dict[str, Any]:
    return build_mango(tmp_path / "mango")
