"""Load the authored instruction bank (task prompts, no disease knowledge)."""

from __future__ import annotations

from pathlib import Path

from plantdx.core.exceptions import ConfigError
from plantdx.utils.io import read_json

_DEFAULT_ASSET = Path("assets/training/instructions.json")


def load_instructions(*, asset_path: str | Path = _DEFAULT_ASSET) -> tuple[str, ...]:
    """Return the ordered tuple of instruction paraphrases.

    Order is preserved (deterministic pairing indexes into it). Raises
    :class:`ConfigError` if the asset is missing or empty.
    """
    path = Path(asset_path)
    if not path.is_file():
        raise ConfigError(f"instruction bank not found: {path}")
    data = read_json(path)
    items = data.get("instructions", []) if isinstance(data, dict) else []
    if not isinstance(items, list) or not items:
        raise ConfigError(f"instruction bank has no instructions: {path}")
    return tuple(str(x) for x in items)
