"""Training, multi-positive contrastive losses, optimization, and checkpointing."""

from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.losses import MultiPositiveContrastiveLoss, create_loss
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.trainer import VLMTrainer

__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "MultiPositiveContrastiveLoss",
    "create_loss",
    "create_optimizer",
    "create_scheduler",
    "VLMTrainer",
]
