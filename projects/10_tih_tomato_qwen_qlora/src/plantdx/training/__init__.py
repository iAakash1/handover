"""Training pipeline (Milestone 7): config -> tomato dataset -> mlx-vlm LoRA.

Public surface:
- ``load_training_config`` / ``TrainingConfig`` — composed, validated config.
- ``prepare_run`` / ``PreparedRun`` / ``launch`` — build everything, then (only on
  request) start the mlx-vlm training subprocess.
- ``build_command`` — the exact ``mlx_vlm.lora`` argv for a config.
- ``load_model`` / ``caption_image`` — inference with a trained adapter.

MLX is imported lazily inside ``runner.launch`` / ``inference`` so this package
imports cleanly where MLX is unavailable (CI, non-Apple hardware).

The M5-era ``MLXVLMRunner`` / ``QLoRASettings`` interface stubs are retained but
superseded by the modules above (same "new impl supersedes old stub" pattern as
``concepts/`` vs ``ontology/{builder,models}``).
"""

from __future__ import annotations

from plantdx.training.command import build_command, render_command
from plantdx.training.config import TrainingConfig, load_training_config
from plantdx.training.mlx_runner import MLXVLMRunner
from plantdx.training.planner import PlanEstimate, build_plan
from plantdx.training.qlora import QLoRASettings, resolve_settings
from plantdx.training.runner import PreparedRun, launch, prepare_run

__all__ = [
    "MLXVLMRunner",
    "PlanEstimate",
    "PreparedRun",
    "QLoRASettings",
    "TrainingConfig",
    "build_command",
    "build_plan",
    "launch",
    "load_training_config",
    "prepare_run",
    "render_command",
    "resolve_settings",
]
