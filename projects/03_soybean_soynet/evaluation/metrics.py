"""
SoyNet Metrics Module
=====================
Metrics computation for model evaluation.
Includes accuracy, precision, recall, F1-score, and confusion matrix.

Author: SoyNet Research Team
Version: 1.0.0
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

import config
from utils.logger import get_logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
    class_names: List[str] = None,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC)
        class_names: List of class names
        average: Averaging method for multi-class ('binary', 'macro', 'weighted')
    
    Returns:
        Dictionary of computed metrics
    """
    logger = get_logger("SoyNet.Metrics")
    
    # Determine if binary or multi-class
    num_classes = len(np.unique(y_true))
    is_binary = num_classes == 2
    
    # Adjust average for binary classification
    if is_binary:
        average = 'binary'
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics['f1_score'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    # Per-class metrics
    if not is_binary:
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, name in enumerate(class_names or range(num_classes)):
            metrics[f'precision_{name}'] = float(per_class_precision[i])
            metrics[f'recall_{name}'] = float(per_class_recall[i])
            metrics[f'f1_{name}'] = float(per_class_f1[i])
    
    # ROC-AUC (if probabilities available)
    if y_pred_proba is not None:
        try:
            if is_binary:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
            else:
                # Multi-class AUC (one-vs-rest)
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(num_classes))
                metrics['roc_auc'] = float(roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr'))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: str = None
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization method ('true', 'pred', 'all', None)
    
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        # Normalize by row (true labels)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        # Normalize by column (predictions)
        cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        # Normalize by total
        cm = cm.astype('float') / cm.sum()
    
    # Handle NaN from division by zero
    cm = np.nan_to_num(cm)
    
    return cm


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    output_dict: bool = False
) -> str:
    """
    Generate a classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dict: Whether to return as dictionary
    
    Returns:
        Classification report string or dictionary
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


def compute_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    return fpr, tpr, thresholds


def compute_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return precision, recall, thresholds


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = 'youden'
) -> float:
    """
    Compute optimal classification threshold.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        method: Method for computing threshold ('youden', 'f1')
    
    Returns:
        Optimal threshold value
    """
    if method == 'youden':
        # Youden's J statistic (maximize TPR - FPR)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return float(thresholds[optimal_idx])
    
    elif method == 'f1':
        # Maximize F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        return float(thresholds[optimal_idx])
    
    else:
        return 0.5  # Default threshold


def get_predictions(
    model,
    dataset,
    threshold: float = 0.5,
    binary: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions from a model on a dataset.
    
    Args:
        model: Trained Keras model
        dataset: TensorFlow dataset
        threshold: Classification threshold (for binary)
        binary: Whether it's binary classification
    
    Returns:
        Tuple of (y_true, y_pred, y_pred_proba)
    """
    logger = get_logger("SoyNet.Metrics")
    logger.info("Generating predictions...")
    
    # Collect true labels and predictions
    y_true_list = []
    y_pred_proba_list = []
    
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy())
        y_pred_proba_list.extend(predictions.flatten())
    
    y_true = np.array(y_true_list)
    y_pred_proba = np.array(y_pred_proba_list)
    
    # Convert probabilities to predictions
    if binary:
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    
    return y_true, y_pred, y_pred_proba


class MetricsTracker:
    """
    Class for tracking metrics during training.
    """
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update with new metrics."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_best(self, metric: str, mode: str = 'max') -> Optional[float]:
        """Get best value for a metric."""
        if metric not in self.history or not self.history[metric]:
            return None
        
        values = self.history[metric]
        if mode == 'max':
            return max(values)
        else:
            return min(values)
    
    def summary(self) -> Dict[str, float]:
        """Get summary of best metrics."""
        return {
            'best_val_accuracy': self.get_best('val_accuracy', 'max'),
            'best_val_loss': self.get_best('val_loss', 'min'),
            'final_train_accuracy': self.history['train_accuracy'][-1] if self.history['train_accuracy'] else None,
            'final_val_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else None
        }
