"""
Model layer — one adapter per VLM family, selected by config.

    cfg.model.adapter: qwen2_5_vl   ->   Qwen25VLAdapter

Adding a model is a new file next to qwen2_5_vl.py plus one entry in ADAPTERS.
No other module in the project changes, because nothing else imports a model
class directly.
"""

from __future__ import annotations

from typing import Dict, Type

from ..config import Config
from .base import VLMAdapter
from .qwen2_5_vl import Qwen25VLAdapter

ADAPTERS: Dict[str, Type] = {
    Qwen25VLAdapter.name: Qwen25VLAdapter,
}

__all__ = ["VLMAdapter", "Qwen25VLAdapter", "ADAPTERS", "get_adapter"]


def get_adapter(cfg: Config) -> VLMAdapter:
    key = cfg.model.adapter
    if key not in ADAPTERS:
        raise ValueError(
            f"unknown model adapter '{key}'. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[key](cfg)
