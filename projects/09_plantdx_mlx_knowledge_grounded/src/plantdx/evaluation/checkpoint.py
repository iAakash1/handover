"""Adapter checkpoint resolution: the directory mlx-vlm's LoRA loader expects.

mlx-vlm's ``apply_lora_layers(model, adapter_path)`` requires ``adapter_path``
to be a **directory** containing exactly ``adapter_config.json`` and
``adapters.safetensors`` (both fixed filenames, read from inside that
directory) -- this is precisely the layout the frozen training pipeline's
``mlx_vlm.lora`` run writes to ``checkpoints/<run_name>/`` (see
``training/command.py``'s ``--output-path`` / ``training/checkpoints.py``'s
``CheckpointConfig.output_dir``). This module never hardcodes a run name or
crop, so it resolves the same way for tomato, a future mango run, or any other
``checkpoints/<run_name>/`` directory.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.core.exceptions import DerivationError

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapters.safetensors"


def _candidate_dir(adapter_path: str | Path) -> Path:
    """The checkpoint directory implied by ``adapter_path``, no filesystem I/O."""
    path = Path(adapter_path)
    return path.parent if path.name == ADAPTER_WEIGHTS_NAME else path


def run_name_from_adapter_path(adapter_path: str | Path) -> str:
    """The run name implied by an adapter path (e.g. ``qwen25vl_mango_qlora``).

    No filesystem I/O -- safe to call before the checkpoint exists, or in an
    environment (e.g. the ``analyze`` stage) that never touches it at all. Used
    to default the evaluation report directory to ``reports/<run_name>/``
    instead of a hardcoded run name, mirroring the training pipeline's own
    ``checkpoints/<run_name>/`` / ``reports/<run_name>/`` convention.
    """
    return _candidate_dir(adapter_path).name


def resolve_adapter_dir(adapter_path: str | Path) -> Path:
    """Resolve a configured adapter path to the checkpoint directory mlx-vlm needs.

    Accepts either the checkpoint directory itself, or -- for convenience and
    backward compatibility with configs/flags that name the weights file
    directly -- a path ending in ``adapters.safetensors``, in which case its
    parent directory is used. Fails closed with an actionable message (never a
    raw traceback) if the resolved directory doesn't exist or is missing either
    required file, so a malformed or half-written checkpoint is caught before
    mlx-vlm ever sees it.
    """
    candidate = _candidate_dir(adapter_path)

    if not candidate.is_dir():
        raise DerivationError(
            f"adapter checkpoint not found: {adapter_path!r} does not resolve to a "
            f"directory ({candidate}). mlx-vlm expects the training checkpoint "
            f"directory itself (e.g. checkpoints/<run_name>/), not a file inside it "
            f"-- pass that directory via --adapter, or point --adapter at its "
            f"'{ADAPTER_WEIGHTS_NAME}' file and the parent directory will be used."
        )

    required = (ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME)
    missing = [name for name in required if not (candidate / name).is_file()]
    if missing:
        raise DerivationError(
            f"malformed adapter checkpoint at {candidate}: missing "
            f"{', '.join(missing)}. Expected both '{ADAPTER_CONFIG_NAME}' and "
            f"'{ADAPTER_WEIGHTS_NAME}' side by side, exactly as mlx_vlm.lora writes "
            f"them during training (see docs/TRAINING.md)."
        )
    return candidate


def adapter_weights_path(adapter_dir: str | Path) -> Path:
    """The actual weights file inside a resolved adapter directory."""
    return Path(adapter_dir) / ADAPTER_WEIGHTS_NAME
