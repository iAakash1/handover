"""Configuration loading and merging.

Reads ``configs/config.yaml``, resolves its ``includes:`` (each included file
contributes top-level sections), applies optional overrides, validates the result
against :class:`plantdx.config.schema.PlantDxConfig`, and computes the
``config_hash`` used in provenance.

Precedence (low -> high): included files < config.yaml root keys < ``overrides``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from plantdx.config.schema import PlantDxConfig
from plantdx.core.exceptions import ConfigError
from plantdx.utils.hashing import stable_json_hash
from plantdx.utils.io import read_yaml


def _deep_update(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into ``base`` (in place) and return ``base``."""
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path = "configs/config.yaml",
    *,
    overrides: Mapping[str, Any] | None = None,
) -> PlantDxConfig:
    """Load, compose, validate, and return the merged configuration.

    Args:
        path: Path to the master ``config.yaml``.
        overrides: Optional highest-precedence overrides (e.g., from the CLI).

    Returns:
        A validated, frozen :class:`PlantDxConfig`.

    Raises:
        plantdx.core.exceptions.ConfigError: If a file is missing or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    root = read_yaml(path)
    if not isinstance(root, dict):
        raise ConfigError(f"config root must be a mapping: {path}")

    includes = root.pop("includes", []) or []
    merged: dict[str, Any] = {}
    for include in includes:
        include_path = path.parent / include
        if not include_path.is_file():
            raise ConfigError(f"included config not found: {include_path}")
        included = read_yaml(include_path)
        if not isinstance(included, dict):
            raise ConfigError(f"included config must be a mapping: {include_path}")
        _deep_update(merged, included)

    _deep_update(merged, root)
    if overrides:
        _deep_update(merged, overrides)

    try:
        return PlantDxConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration ({path}):\n{exc}") from exc


def config_hash(config: PlantDxConfig) -> str:
    """Return a stable 16-char hash of the effective configuration (for provenance)."""
    return stable_json_hash(config.model_dump(mode="json"))[:16]
