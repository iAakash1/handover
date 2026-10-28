"""
SoyNet Trainer Module
=====================
Training orchestration with callbacks, history tracking, and fine-tuning support.

Author: SoyNet Research Team
Version: 1.0.0
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import tensorflow as tf

import config
from utils.logger import get_logger, TrainingLogger
from utils.helpers import save_json, format_time
from utils.callbacks import get_callbacks, get_fine_tuning_callbacks


class ModelTrainer:
    """
    Model trainer class for orchestrating the training process.
    Handles training, fine-tuning, and history tracking.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        dataset_info=None,
        model_name: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Compiled Keras model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            dataset_info: DatasetInfo object with class weights
            model_name: Name of the model
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataset_info = dataset_info
        self.model_name = model_name or config.MODEL_NAME
        
        self.logger = get_logger("SoyNet.Trainer")
        self.training_logger = TrainingLogger()
        
        self.history = None
        self.fine_tune_history = None
    
    def train(
        self,
        epochs: int = None,
        callbacks: list = None,
        class_weights: Dict[int, float] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs
            callbacks: List of callbacks
            class_weights: Class weights for imbalanced data
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        epochs = epochs or config.EPOCHS
        
        # Get class weights if available
        if class_weights is None and self.dataset_info:
            class_weights = self.dataset_info.class_weights
        
        # Get default callbacks if not provided
        if callbacks is None:
            callbacks = get_callbacks(logger=self.logger)
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING PHASE")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Epochs: {epochs}")
        
        if class_weights:
            self.logger.info(f"Using class weights: {class_weights}")
        
        # Train
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
        
        training_duration = datetime.now() - start_time
        self.logger.info(f"Training completed in {training_duration}")
        
        return self.history
    
    def fine_tune(
        self,
        epochs: int = None,
        unfreeze_from_layer: int = None,
        learning_rate: float = None,
        callbacks: list = None,
        class_weights: Dict[int, float] = None,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Fine-tune the model by unfreezing backbone layers.
        
        Args:
            epochs: Number of fine-tuning epochs
            unfreeze_from_layer: Layer index to start unfreezing
            learning_rate: Learning rate for fine-tuning
            callbacks: List of callbacks
            class_weights: Class weights
            verbose: Verbosity level
        
        Returns:
            Fine-tuning history
        """
        if not config.FINE_TUNE:
            self.logger.info("Fine-tuning disabled in config")
            return None
        
        epochs = epochs or (config.EPOCHS // 2)  # Default: half of initial epochs
        unfreeze_from = unfreeze_from_layer or config.FINE_TUNE_AT_LAYER
        learning_rate = learning_rate or config.FINE_TUNE_LEARNING_RATE
        
        self.logger.info("=" * 60)
        self.logger.info("FINE-TUNING PHASE")
        self.logger.info("=" * 60)
        
        # Prepare model for fine-tuning
        from models.model_factory import prepare_for_fine_tuning
        
        num_classes = 2 if self.dataset_info is None else self.dataset_info.num_classes
        self.model = prepare_for_fine_tuning(
            self.model,
            unfreeze_from_layer=unfreeze_from,
            learning_rate=learning_rate,
            num_classes=num_classes
        )
        
        # Get class weights
        if class_weights is None and self.dataset_info:
            class_weights = self.dataset_info.class_weights
        
        # Get fine-tuning callbacks
        if callbacks is None:
            callbacks = get_fine_tuning_callbacks(logger=self.logger)
        
        # Fine-tune
        start_time = datetime.now()
        
        # Get initial epoch from previous training
        initial_epoch = 0
        if self.history:
            initial_epoch = len(self.history.history.get('loss', []))
        
        self.fine_tune_history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=initial_epoch + epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
        
        fine_tune_duration = datetime.now() - start_time
        self.logger.info(f"Fine-tuning completed in {fine_tune_duration}")
        
        return self.fine_tune_history
    
    def save_model(self, path: Path = None):
        """
        Save the trained model.
        
        Args:
            path: Path to save model
        """
        if path is None:
            path = config.get_model_save_path()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path)
        self.logger.info(f"Model saved to: {path}")
    
    def save_history(self, path: Path = None):
        """
        Save training history to JSON.
        
        Args:
            path: Path to save history
        """
        if path is None:
            path = config.get_history_save_path()
        
        # Combine histories
        history_data = {}
        
        if self.history:
            history_data['initial_training'] = {
                k: [float(v) for v in vals] 
                for k, vals in self.history.history.items()
            }
        
        if self.fine_tune_history:
            history_data['fine_tuning'] = {
                k: [float(v) for v in vals] 
                for k, vals in self.fine_tune_history.history.items()
            }
        
        # Add metadata
        history_data['metadata'] = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'epochs': config.EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.INITIAL_LEARNING_RATE,
                'fine_tune_lr': config.FINE_TUNE_LEARNING_RATE,
                'image_size': config.IMAGE_SIZE
            }
        }
        
        save_json(history_data, path)
        self.logger.info(f"Training history saved to: {path}")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best metrics from training history.
        
        Returns:
            Dictionary of best metrics
        """
        best_metrics = {}
        
        # Combine all histories
        all_histories = []
        if self.history:
            all_histories.append(self.history.history)
        if self.fine_tune_history:
            all_histories.append(self.fine_tune_history.history)
        
        for history in all_histories:
            for key, values in history.items():
                if 'val_' in key:
                    if 'loss' in key:
                        # For loss, lower is better
                        current_best = min(values)
                        if key not in best_metrics or current_best < best_metrics[key]:
                            best_metrics[key] = current_best
                    else:
                        # For accuracy/AUC, higher is better
                        current_best = max(values)
                        if key not in best_metrics or current_best > best_metrics[key]:
                            best_metrics[key] = current_best
        
        return best_metrics


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    dataset_info=None,
    epochs: int = None,
    fine_tune: bool = None,
    callbacks: list = None,
    save_model_flag: bool = True,
    save_history_flag: bool = True
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Complete training pipeline function.
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        dataset_info: DatasetInfo object
        epochs: Number of epochs
        fine_tune: Whether to perform fine-tuning
        callbacks: Custom callbacks
        save_model_flag: Whether to save the model
        save_history_flag: Whether to save training history
    
    Returns:
        Tuple of (trained_model, training_info)
    """
    logger = get_logger("SoyNet.Training")
    
    epochs = epochs or config.EPOCHS
    fine_tune = fine_tune if fine_tune is not None else config.FINE_TUNE
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataset_info=dataset_info
    )
    
    # Initial training
    logger.info("Starting initial training phase...")
    history = trainer.train(epochs=epochs, callbacks=callbacks)
    
    # Fine-tuning (if enabled)
    if fine_tune:
        logger.info("Starting fine-tuning phase...")
        trainer.fine_tune()
    
    # Save model and history
    if save_model_flag:
        trainer.save_model()
    
    if save_history_flag:
        trainer.save_history()
    
    # Get best metrics
    best_metrics = trainer.get_best_metrics()
    logger.info("Best validation metrics:")
    for key, value in best_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Prepare training info
    training_info = {
        'best_metrics': best_metrics,
        'epochs_trained': len(history.history.get('loss', [])),
        'model_path': str(config.get_model_save_path()),
        'history_path': str(config.get_history_save_path())
    }
    
    return trainer.model, training_info
