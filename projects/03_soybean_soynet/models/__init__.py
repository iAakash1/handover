"""
SoyNet Models Package
=====================
Model architecture modules including factory, custom CNN, and trainer.
"""

from models.model_factory import (
    create_model,
    get_backbone,
    add_classification_head,
    prepare_for_fine_tuning
)
from models.custom_cnn import create_custom_cnn
from models.trainer import ModelTrainer, train_model

__all__ = [
    'create_model',
    'get_backbone',
    'add_classification_head',
    'prepare_for_fine_tuning',
    'create_custom_cnn',
    'ModelTrainer',
    'train_model'
]
