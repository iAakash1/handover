"""
SoyNet Callbacks Module
=======================
Custom Keras callbacks for training monitoring and control.

Author: SoyNet Research Team
Version: 1.0.0
"""

from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger
)

import config
from utils.logger import get_logger


class TrainingProgressCallback(Callback):
    """
    Custom callback for logging training progress.
    Provides formatted output after each epoch.
    """
    
    def __init__(self, logger=None):
        """
        Initialize callback.
        
        Args:
            logger: Logger instance (creates default if None)
        """
        super().__init__()
        self.logger = logger or get_logger("SoyNet.Training")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        self.logger.info(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        self.logger.info("-" * 40)
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Format metrics
        metrics_parts = []
        for key, value in logs.items():
            metrics_parts.append(f"{key}: {value:.4f}")
        
        metrics_str = " | ".join(metrics_parts)
        self.logger.info(f"Results: {metrics_str}")
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
    
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)


class LearningRateLogger(Callback):
    """
    Callback to log learning rate changes.
    """
    
    def __init__(self, logger=None):
        """
        Initialize callback.
        
        Args:
            logger: Logger instance
        """
        super().__init__()
        self.logger = logger or get_logger("SoyNet.Training")
        self.prev_lr = None
    
    def on_epoch_end(self, epoch, logs=None):
        """Check and log learning rate changes."""
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        if self.prev_lr is not None and current_lr != self.prev_lr:
            self.logger.info(f"Learning rate changed: {self.prev_lr:.2e} -> {current_lr:.2e}")
        
        self.prev_lr = current_lr


class ModelCheckpointWithLogging(ModelCheckpoint):
    """
    Extended ModelCheckpoint that logs when model is saved.
    """
    
    def __init__(self, filepath, logger=None, **kwargs):
        """
        Initialize callback.
        
        Args:
            filepath: Path to save model
            logger: Logger instance
            **kwargs: Additional arguments for ModelCheckpoint
        """
        super().__init__(filepath, **kwargs)
        self.logger = logger or get_logger("SoyNet.Training")
    
    def on_epoch_end(self, epoch, logs=None):
        """Override to add logging."""
        super().on_epoch_end(epoch, logs)
        
        # Check if model was saved this epoch
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is not None:
                if self._is_improvement(current, self.best):
                    self.logger.info(f"Model checkpoint saved (best {self.monitor}: {current:.4f})")
    
    def _is_improvement(self, current, best):
        """Check if current is an improvement over best."""
        if best is None:
            return True
        # Use monitor_op from parent class if available, otherwise fallback to comparison
        try:
            return self.monitor_op(current, best)
        except AttributeError:
            # Fallback: compare based on metric name
            if 'loss' in self.monitor:
                return current < best
            else:
                return current > best


def get_callbacks(
    model_path: Path = None,
    log_dir: Path = None,
    enable_tensorboard: bool = False,
    enable_csv_logger: bool = True,
    logger=None
) -> List[Callback]:
    """
    Get list of callbacks for training.
    
    Args:
        model_path: Path to save model checkpoints
        log_dir: Directory for logs
        enable_tensorboard: Whether to enable TensorBoard logging
        enable_csv_logger: Whether to enable CSV logging of training history
        logger: Logger instance
    
    Returns:
        List of configured callbacks
    """
    callbacks = []
    logger = logger or get_logger("SoyNet.Training")
    
    # Model path
    if model_path is None:
        model_path = config.get_model_save_path()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log directory
    if log_dir is None:
        log_dir = config.LOGS_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Early Stopping
    early_stopping = EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        mode=config.EARLY_STOPPING_MODE,
        restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST,
        verbose=1
    )
    callbacks.append(early_stopping)
    logger.info(f"Early stopping enabled (patience: {config.EARLY_STOPPING_PATIENCE})")
    
    # 2. Model Checkpoint
    checkpoint = ModelCheckpointWithLogging(
        filepath=str(model_path),
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE,
        save_best_only=config.CHECKPOINT_SAVE_BEST_ONLY,
        verbose=0,
        logger=logger
    )
    callbacks.append(checkpoint)
    logger.info(f"Model checkpoint: {model_path}")
    
    # 3. Learning Rate Reduction
    reduce_lr = ReduceLROnPlateau(
        monitor=config.EARLY_STOPPING_MONITOR,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.REDUCE_LR_MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr)
    logger.info(f"LR reduction enabled (patience: {config.REDUCE_LR_PATIENCE})")
    
    # 4. Training Progress Logger
    progress_callback = TrainingProgressCallback(logger)
    callbacks.append(progress_callback)
    
    # 5. Learning Rate Logger
    lr_logger = LearningRateLogger(logger)
    callbacks.append(lr_logger)
    
    # 6. TensorBoard (optional)
    if enable_tensorboard:
        from utils.helpers import get_timestamp
        tb_log_dir = log_dir / f"tensorboard_{get_timestamp()}"
        tensorboard = TensorBoard(
            log_dir=str(tb_log_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        logger.info(f"TensorBoard enabled: {tb_log_dir}")
    
    # 7. CSV Logger (optional)
    if enable_csv_logger:
        from utils.helpers import get_timestamp
        csv_path = log_dir / f"training_log_{get_timestamp()}.csv"
        csv_logger = CSVLogger(str(csv_path))
        callbacks.append(csv_logger)
        logger.info(f"CSV logger: {csv_path}")
    
    return callbacks


def get_fine_tuning_callbacks(
    model_path: Path = None,
    logger=None
) -> List[Callback]:
    """
    Get callbacks specifically for fine-tuning phase.
    Uses lower learning rate and different patience values.
    
    Args:
        model_path: Path to save model checkpoints
        logger: Logger instance
    
    Returns:
        List of configured callbacks for fine-tuning
    """
    callbacks = []
    logger = logger or get_logger("SoyNet.FineTuning")
    
    # Model path for fine-tuned model
    if model_path is None:
        model_path = config.SAVED_MODELS_DIR / f"soynet_{config.MODEL_NAME}_finetuned.keras"
    
    # Early Stopping with more patience for fine-tuning
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE + 5,  # More patience for fine-tuning
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Checkpoint for fine-tuned model
    checkpoint = ModelCheckpointWithLogging(
        filepath=str(model_path),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        logger=logger
    )
    callbacks.append(checkpoint)
    
    # Learning rate reduction with lower min_lr
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Progress logging
    callbacks.append(TrainingProgressCallback(logger))
    callbacks.append(LearningRateLogger(logger))
    
    return callbacks
