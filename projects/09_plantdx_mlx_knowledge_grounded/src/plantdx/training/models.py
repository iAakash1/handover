"""Model registry: which VLMs this training pipeline knows how to drive.

Small and explicit — one entry per supported model. Adding a model is a new
``ModelSpec`` plus a ``configs/models/*.yaml``, never a code-flow change.
"""

from __future__ import annotations

from dataclasses import dataclass

from plantdx.core.enums import TargetModel


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about a supported model + how mlx-vlm should treat it."""

    key: str  # logical name used in configs (model.name)
    target: TargetModel  # the core enum member
    hf_repo: str  # canonical HF repo id (already-downloaded 4-bit MLX build)
    supports_vision_finetune: bool


_REGISTRY: dict[str, ModelSpec] = {
    "qwen2_5_vl": ModelSpec(
        key="qwen2_5_vl",
        target=TargetModel.QWEN2_5_VL,
        hf_repo="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        supports_vision_finetune=True,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name`` or raise ``KeyError``."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown model {name!r}; registered models: {known}") from exc


def registered_models() -> tuple[str, ...]:
    """Return the sorted keys of all registered models."""
    return tuple(sorted(_REGISTRY))
