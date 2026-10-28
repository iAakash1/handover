"""
Training layer.

    dataset.py   JSONL rows -> batched tensors, via the model adapter
    run.py       run directories, frozen config, lineage
    trainer.py   the loop: accumulation, validation, early stopping, checkpoints

None of these modules contains model-specific code. Swapping the VLM changes the
adapter, not the trainer.
"""

from .dataset import CaptionDataset, build_loader, describe_sample
from .run import Run
from .trainer import Trainer, TrainResult, set_seed

__all__ = ["CaptionDataset", "build_loader", "describe_sample",
           "Run", "Trainer", "TrainResult", "set_seed"]
