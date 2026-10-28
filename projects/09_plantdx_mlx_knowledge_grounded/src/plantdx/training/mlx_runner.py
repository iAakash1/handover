"""MLX / mlx-vlm QLoRA training runner (interface) (doc 04 §6.4).

Milestone 5 drives ``mlx_vlm.lora`` on Apple Silicon (M4 Pro, 24 GB) against the
per-model converted dataset. This module owns the runner contract; the mlx-vlm
version is pinned per ``datasets/mlx_vlm/README.md``.
"""

from __future__ import annotations

from pathlib import Path

from plantdx.training.qlora import QLoRASettings


class MLXVLMRunner:
    """Runs a QLoRA fine-tune via mlx-vlm.

    Args:
        settings: Resolved per-model QLoRA settings.
        dataset_dir: The model's converted dataset directory (``datasets/<model>/``).
        output_dir: Where adapters/checkpoints are written.
    """

    def __init__(
        self,
        settings: QLoRASettings,
        dataset_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        """Initialize the runner with the QLoRA settings and dataset/output paths."""
        self.settings = settings
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)

    def preflight(self) -> None:
        """Verify Apple Silicon + MLX availability and the mlx-vlm dataset schema."""
        raise NotImplementedError("Milestone 5: training preflight")

    def train(self) -> Path:
        """Run fine-tuning; return the path to the produced adapter."""
        raise NotImplementedError("Milestone 5: mlx-vlm QLoRA training")
