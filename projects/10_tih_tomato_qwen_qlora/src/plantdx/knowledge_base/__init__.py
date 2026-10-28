"""Read-only consumer of the FINAL Disease Knowledge Base (Stage 1).

The DKB (``knowledge_base/dkb.json``) is the single source of truth. This package
loads it into typed models; it never re-authors or mutates disease facts. Record
models live in :mod:`plantdx.knowledge_base.models`.
"""

from __future__ import annotations

from plantdx.knowledge_base.loader import DKBLoader
from plantdx.knowledge_base.models import KnowledgeBase

__all__ = ["DKBLoader", "KnowledgeBase"]
