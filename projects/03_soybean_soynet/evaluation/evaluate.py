"""
SoyNet Evaluation Module
========================
Model evaluation and visualization functions.
Generates confusion matrices, training curves, and evaluation reports.

Author: SoyNet Research Team
Version: 1.0.0
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
from utils.logger import get_logger, EvaluationLogger
from utils.helpers import save_json, get_timestamp
from evaluation.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_classification_report,
    get_predictions,
    compute_roc_curve
)


def evaluate_model(
    model,
    test_dataset,
    class_names: List[str] = None,
    save_plots: bool = True,
    plots_dir: Path = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation on test dataset.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        class_names: List of class names
        save_plots: Whether to save plots
        plots_dir: Directory for saving plots
    
    Returns:
        Dictionary containing evaluation results
    """
    logger = get_logger("SoyNet.Evaluation")
    eval_logger = EvaluationLogger()
    
    plots_dir = Path(plots_dir or config.PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Count test samples
    test_size = sum(1 for _ in test_dataset.unbatch())
    eval_logger.log_evaluation_start("test", test_size)
    
    # Get predictions
    logger.info("Generating predictions on test set...")
    y_true, y_pred, y_pred_proba = get_predictions(
        model, test_dataset, binary=(len(class_names) == 2 if class_names else True)
    )
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(
        y_true, y_pred, y_pred_proba,
        class_names=class_names,
        average='binary' if len(class_names or []) == 2 else 'weighted'
    )
    eval_logger.log_metrics(metrics)
    
    # Classification report
    report = compute_classification_report(y_true, y_pred, class_names=class_names)
    eval_logger.log_classification_report(report)
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    
    # Save plots
    if save_plots:
        # Plot and save confusion matrix
        cm_path = plots_dir / f"confusion_matrix_{get_timestamp()}.png"
        plot_confusion_matrix(cm, class_names, save_path=cm_path)
        eval_logger.log_confusion_matrix_saved(str(cm_path))
        
        # Plot and save normalized confusion matrix
        cm_normalized = compute_confusion_matrix(y_true, y_pred, normalize='true')
        cm_norm_path = plots_dir / f"confusion_matrix_normalized_{get_timestamp()}.png"
        plot_confusion_matrix(
            cm_normalized, class_names, 
            save_path=cm_norm_path, 
            title='Normalized Confusion Matrix'
        )
        
        # ROC curve (for binary classification)
        if len(class_names or []) == 2:
            roc_path = plots_dir / f"roc_curve_{get_timestamp()}.png"
            plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)
        
        eval_logger.log_plots_saved(str(plots_dir))
    
    # Compile results
    results = {
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist(),
        'y_pred_proba': y_pred_proba.tolist()
    }
    
    # Save results to JSON
    results_path = plots_dir / f"evaluation_results_{get_timestamp()}.json"
    save_json({
        'metrics': metrics,
        'confusion_matrix': cm.tolist()
    }, results_path)
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Path = None,
    title: str = 'Confusion Matrix',
    figsize: tuple = (8, 6),
    cmap: str = 'Blues'
):
    """
    Plot and optionally save confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    plt.figure(figsize=figsize)
    
    # Use seaborn heatmap for better visualization
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if cm.max() <= 1 else 'd',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar=True
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path = None,
    figsize: tuple = (12, 5)
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    ax1 = axes[0]
    if 'loss' in history:
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2 = axes[1]
    if 'accuracy' in history:
        ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Path = None,
    figsize: tuple = (8, 6)
):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import roc_auc_score
    
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Path = None,
    figsize: tuple = (10, 6)
):
    """
    Plot comparison of metrics across different models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        save_path: Path to save figure
        figsize: Figure size
    """
    model_names = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[model].get(metric, 0) for model in model_names]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_evaluation_report(
    model,
    test_dataset,
    dataset_info,
    history: Dict[str, List[float]] = None,
    save_dir: Path = None
) -> Path:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        dataset_info: DatasetInfo object
        history: Training history
        save_dir: Directory to save report
    
    Returns:
        Path to saved report
    """
    logger = get_logger("SoyNet.Evaluation")
    
    save_dir = Path(save_dir or config.RESULTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    
    # Evaluate model
    results = evaluate_model(
        model,
        test_dataset,
        class_names=dataset_info.class_names,
        save_plots=True,
        plots_dir=save_dir / 'plots'
    )
    
    # Save training history plots
    if history:
        history_path = save_dir / 'plots' / f"training_history_{timestamp}.png"
        plot_training_history(history, save_path=history_path)
    
    # Create report
    report = {
        'timestamp': timestamp,
        'model_name': config.MODEL_NAME,
        'dataset': {
            'num_classes': dataset_info.num_classes,
            'class_names': dataset_info.class_names,
            'train_size': dataset_info.train_size,
            'val_size': dataset_info.val_size,
            'test_size': dataset_info.test_size
        },
        'config': {
            'image_size': config.IMAGE_SIZE,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.INITIAL_LEARNING_RATE
        },
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix']
    }
    
    # Save report
    report_path = save_dir / f"evaluation_report_{timestamp}.json"
    save_json(report, report_path)
    
    logger.info(f"Evaluation report saved to: {report_path}")
    
    return report_path


def print_evaluation_summary(results: Dict[str, Any]):
    """
    Print a formatted evaluation summary.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    metrics = results.get('metrics', {})
    
    print("\nPerformance Metrics:")
    print("-" * 40)
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
    
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
    
    print("=" * 60 + "\n")
