"""Typed configuration schema and loader for PlantDx.

Public entry points: :func:`load_config`, :func:`config_hash`, and the top-level
:class:`PlantDxConfig`. The nested config models are defined in
:mod:`plantdx.config.schema` and imported from there when needed.
"""

from __future__ import annotations

from plantdx.config.loader import config_hash, load_config
from plantdx.config.schema import PlantDxConfig

__all__ = ["PlantDxConfig", "config_hash", "load_config"]
