"""Dataset package (component I): emitter, serialization, splits, label map, converters."""

from __future__ import annotations

from plantdx.dataset.converters import CONVERTER_REGISTRY, BaseConverter
from plantdx.dataset.emitter import Emitter
from plantdx.dataset.instructions import InstructionBank
from plantdx.dataset.label_map import LabelResolver
from plantdx.dataset.serialization import SCHEMA_VERSION, record_from_dict, record_to_dict
from plantdx.dataset.splits import SplitBuilder

__all__ = [
    "CONVERTER_REGISTRY",
    "SCHEMA_VERSION",
    "BaseConverter",
    "Emitter",
    "InstructionBank",
    "LabelResolver",
    "SplitBuilder",
    "record_from_dict",
    "record_to_dict",
]
