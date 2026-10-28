"""Dataset export formats — pure, deterministic reshapers of one caption.

Every exporter is a pure function ``Caption -> dict``; the only difference
between formats is serialization shape. No exporter regenerates, edits, or
invents caption text; they add only the fixed instruction scaffold each format
expects. Because this milestone's corpus is image-independent (image grounding
is a later milestone), the image-grounded formats emit their text side only —
the caption plus the canonical instruction — which is exactly the part this
layer owns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plantdx.corpus.models import Caption

# One fixed, neutral instruction (no domain content). A diverse instruction bank
# and per-image pairing belong to the image-grounded dataset milestone.
CANONICAL_INSTRUCTION = "Describe the condition of this leaf."


def _generic(c: Caption) -> dict[str, Any]:
    """Alpaca-style instruction record."""
    return {"instruction": CANONICAL_INSTRUCTION, "input": "", "output": c.text}


def _llava(c: Caption) -> dict[str, Any]:
    """LLaVA-style ``conversations`` record (text side; no ``<image>`` yet)."""
    return {
        "id": c.caption_id,
        "conversations": [
            {"from": "human", "value": CANONICAL_INSTRUCTION},
            {"from": "gpt", "value": c.text},
        ],
    }


def _paligemma(c: Caption) -> dict[str, Any]:
    """PaliGemma-style ``prefix``/``suffix`` record."""
    return {"id": c.caption_id, "prefix": CANONICAL_INSTRUCTION, "suffix": c.text}


def _blip2(c: Caption) -> dict[str, Any]:
    """BLIP-2-style image-caption record (text side: the caption)."""
    return {"image_id": c.caption_id, "caption": c.text}


def _messages(c: Caption) -> dict[str, Any]:
    """Chat ``messages`` record (role/content list)."""
    return {
        "id": c.caption_id,
        "messages": [
            {"role": "user", "content": CANONICAL_INSTRUCTION},
            {"role": "assistant", "content": c.text},
        ],
    }


#: Registry mapping an export format name to its pure per-caption reshaper.
FORMAT_REGISTRY: dict[str, Callable[[Caption], dict[str, Any]]] = {
    "generic": _generic,
    "llava": _llava,
    "paligemma": _paligemma,
    "blip2": _blip2,
    "messages": _messages,
}

FORMATS: tuple[str, ...] = tuple(sorted(FORMAT_REGISTRY))
