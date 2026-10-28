"""Per-model dataset converters (doc 04 §6).

A converter is a pure, deterministic function ``canonical_record -> trainer_line``.
It adds only role scaffolding and image placeholders; it NEVER alters the caption
or instruction text. Converters emit per-split ``{train,val}.jsonl`` and a manifest.

All five converters share the :class:`BaseConverter` abstraction and are exposed
through :data:`CONVERTER_REGISTRY`. Qwen2.5-VL and Qwen3-VL share a schema but
remain separate classes so per-model output dirs and manifests stay distinct.
Milestone 4 implements the bodies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from plantdx.core.enums import Split, TargetModel
from plantdx.core.types import CaptionRecord


class BaseConverter(ABC):
    """Abstract per-model converter.

    Subclasses set :attr:`model_key` and implement :meth:`convert`.
    """

    model_key: str = "base"

    @abstractmethod
    def convert(self, record: CaptionRecord) -> Mapping[str, Any]:
        """Map one canonical record to one trainer-format line."""
        raise NotImplementedError

    def convert_split(
        self,
        records: Iterable[CaptionRecord],
        split: Split,
        out_dir: str | Path,
    ) -> Path:
        """Convert all records of a split and write ``<split>.jsonl`` + a manifest."""
        raise NotImplementedError("Milestone 4: split conversion + manifest")

    def validate_line(self, line: Mapping[str, Any]) -> None:
        """Validate a produced line against the trainer's minimal schema (doc 04 §6.5).

        Raises:
            plantdx.core.exceptions.ConversionError: On an invalid line.
        """
        raise NotImplementedError("Milestone 4: per-line schema validation")


class Qwen2_5VLConverter(BaseConverter):  # noqa: N801 - name encodes the Qwen2.5-VL model; not renamed
    """Qwen2.5-VL messages / typed-content-list schema (doc 04 §6.1)."""

    model_key = "qwen2_5_vl"

    def convert(self, record: CaptionRecord) -> Mapping[str, Any]:  # noqa: D102
        raise NotImplementedError("Milestone 4: Qwen2.5-VL conversion")


class Qwen3VLConverter(BaseConverter):
    """Qwen3-VL messages schema (same data schema as Qwen2.5-VL) (doc 04 §6.1)."""

    model_key = "qwen3_vl"

    def convert(self, record: CaptionRecord) -> Mapping[str, Any]:  # noqa: D102
        raise NotImplementedError("Milestone 4: Qwen3-VL conversion")


class InternVL3Converter(BaseConverter):
    """InternVL3 LLaVA-style ``conversations`` schema with ``<image>`` (doc 04 §6.2)."""

    model_key = "internvl3"

    def convert(self, record: CaptionRecord) -> Mapping[str, Any]:  # noqa: D102
        raise NotImplementedError("Milestone 4: InternVL3 conversion")


class Gemma3Converter(BaseConverter):
    """Gemma-3 messages / content-list schema (doc 04 §6.3)."""

    model_key = "gemma3"

    def convert(self, record: CaptionRecord) -> Mapping[str, Any]:  # noqa: D102
        raise NotImplementedError("Milestone 4: Gemma-3 conversion")


class MLXVLMConverter(BaseConverter):
    """MLX / mlx-vlm LoRA chat schema (doc 04 §6.4) — the fine-tuning tool.

    VERSION CAVEAT: mlx-vlm's expected dataset schema has changed across releases.
    Target the INSTALLED version, pinned in ``datasets/mlx_vlm/README.md``, and
    validate one sample end-to-end before bulk conversion.
    """

    model_key = "mlx_vlm"

    def __init__(self, mlx_vlm_version: str | None = None) -> None:
        """Record the target mlx-vlm version for schema pinning."""
        self.mlx_vlm_version = mlx_vlm_version

    def convert(self, record: CaptionRecord) -> Mapping[str, Any]:  # noqa: D102
        raise NotImplementedError("Milestone 4: mlx-vlm conversion (version-pinned)")


#: Registry mapping a target model to its converter class (doc 04 §6).
CONVERTER_REGISTRY: dict[TargetModel, type[BaseConverter]] = {
    TargetModel.QWEN2_5_VL: Qwen2_5VLConverter,
    TargetModel.QWEN3_VL: Qwen3VLConverter,
    TargetModel.INTERNVL3: InternVL3Converter,
    TargetModel.GEMMA3: Gemma3Converter,
    TargetModel.MLX_VLM: MLXVLMConverter,
}

__all__ = [
    "CONVERTER_REGISTRY",
    "BaseConverter",
    "Gemma3Converter",
    "InternVL3Converter",
    "MLXVLMConverter",
    "Qwen2_5VLConverter",
    "Qwen3VLConverter",
]
