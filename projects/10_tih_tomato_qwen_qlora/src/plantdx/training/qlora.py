"""QLoRA training configuration models (doc 04 §6, ``configs/training.yaml``).

Milestone 5 wires these to the MLX runner. This module owns the typed training
config and the per-model resolution (shared defaults + per-model overrides).
"""

from __future__ import annotations

from dataclasses import dataclass

from plantdx.core.enums import TargetModel


@dataclass(frozen=True, slots=True)
class QLoRASettings:
    """Resolved QLoRA hyperparameters for one model."""

    model: TargetModel
    hf_id: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    learning_rate: float
    lr_schedule: str
    warmup_ratio: float
    batch_size: int
    grad_accum_steps: int
    max_seq_len: int
    epochs: int
    quantization_bits: int
    seed: int


def resolve_settings(training_config: dict[str, object], model: TargetModel) -> QLoRASettings:
    """Merge shared defaults with a model's overrides into :class:`QLoRASettings`."""
    raise NotImplementedError("Milestone 5: QLoRA settings resolution")
