"""
SoyNet Logger Module
====================
Structured logging system for the SoyNet pipeline.
Provides console and file logging with timestamps and different log levels.

Author: SoyNet Research Team
Version: 1.0.0
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


def setup_logging(
    log_level: str = None,
    log_to_file: bool = None,
    log_to_console: bool = None,
    log_dir: Path = None
) -> logging.Logger:
    """
    Set up the logging configuration for the SoyNet pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to a file
        log_to_console: Whether to log to console
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    # Use config values as defaults
    log_level = log_level or config.LOG_LEVEL
    log_to_file = log_to_file if log_to_file is not None else config.LOG_TO_FILE
    log_to_console = log_to_console if log_to_console is not None else config.LOG_TO_CONSOLE
    log_dir = log_dir or config.LOGS_DIR
    
    # Create logger
    logger = logging.getLogger("SoyNet")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create log directory if it doesn't exist
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"soynet_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {log_file}")
    
    return logger


def get_logger(name: str = "SoyNet") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default: "SoyNet")
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up basic configuration
    if not logger.handlers:
        setup_logging()
    
    return logger


class TrainingLogger:
    """
    Specialized logger for training progress.
    Provides formatted output for epochs, metrics, and training events.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize training logger.
        
        Args:
            logger: Base logger instance (creates new if None)
        """
        self.logger = logger or get_logger("SoyNet.Training")
        self.epoch_start_time = None
        self.training_start_time = None
    
    def log_training_start(self, model_name: str, num_epochs: int, 
                          train_size: int, val_size: int):
        """Log training start information."""
        self.training_start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Total Epochs: {num_epochs}")
        self.logger.info(f"Training Samples: {train_size}")
        self.logger.info(f"Validation Samples: {val_size}")
        self.logger.info(f"Start Time: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.epoch_start_time = datetime.now()
        self.logger.info(f"\nEpoch {epoch}/{total_epochs}")
        self.logger.info("-" * 40)
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """Log epoch end with metrics."""
        epoch_duration = datetime.now() - self.epoch_start_time
        
        # Format metrics string
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        self.logger.info(f"Epoch {epoch} completed in {epoch_duration}")
        self.logger.info(f"Metrics: {metrics_str}")
    
    def log_training_end(self, best_metrics: dict = None):
        """Log training end information."""
        training_duration = datetime.now() - self.training_start_time
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Duration: {training_duration}")
        
        if best_metrics:
            self.logger.info("Best Metrics:")
            for key, value in best_metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
        
        self.logger.info("=" * 60)
    
    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping event."""
        self.logger.info(f"\nEarly stopping triggered at epoch {epoch}")
        self.logger.info(f"No improvement for {patience} epochs")
    
    def log_lr_reduction(self, old_lr: float, new_lr: float):
        """Log learning rate reduction."""
        self.logger.info(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
    
    def log_model_saved(self, path: str, metric: str, value: float):
        """Log model checkpoint saved."""
        self.logger.info(f"Model saved to {path}")
        self.logger.info(f"Best {metric}: {value:.4f}")


class EvaluationLogger:
    """
    Specialized logger for model evaluation.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize evaluation logger.
        
        Args:
            logger: Base logger instance (creates new if None)
        """
        self.logger = logger or get_logger("SoyNet.Evaluation")
    
    def log_evaluation_start(self, dataset_name: str, num_samples: int):
        """Log evaluation start."""
        self.logger.info("=" * 60)
        self.logger.info(f"EVALUATION ON {dataset_name.upper()} SET")
        self.logger.info("=" * 60)
        self.logger.info(f"Number of samples: {num_samples}")
    
    def log_metrics(self, metrics: dict):
        """Log evaluation metrics."""
        self.logger.info("\nEvaluation Metrics:")
        self.logger.info("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_classification_report(self, report: str):
        """Log classification report."""
        self.logger.info("\nClassification Report:")
        self.logger.info("-" * 40)
        self.logger.info(f"\n{report}")
    
    def log_confusion_matrix_saved(self, path: str):
        """Log confusion matrix saved."""
        self.logger.info(f"Confusion matrix saved to: {path}")
    
    def log_plots_saved(self, plot_dir: str):
        """Log plots saved."""
        self.logger.info(f"Evaluation plots saved to: {plot_dir}")
