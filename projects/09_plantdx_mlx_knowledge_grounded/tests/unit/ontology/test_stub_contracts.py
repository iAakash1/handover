"""Contract tests: Milestone-1 stubs raise NotImplementedError, not silent no-ops."""

from __future__ import annotations

import pytest

from plantdx.knowledge_base.loader import DKBLoader


@pytest.mark.unit
def test_dkb_loader_stub() -> None:
    """DKBLoader.load() is not implemented yet; it must fail loudly, not silently."""
    with pytest.raises(NotImplementedError):
        DKBLoader("knowledge_base/dkb.json").load()
