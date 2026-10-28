"""Shared fixture: load the authored template library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_TEMPLATES = Path(__file__).resolve().parents[3] / "assets" / "templates" / "templates.json"


@pytest.fixture(scope="session")
def library() -> Any:
    """Load the real authored template library once."""
    if not _TEMPLATES.is_file():
        pytest.skip("templates.json not present")
    from plantdx.templates import load_library

    return load_library(_TEMPLATES)
