"""
SoyNet Evaluation Package
=========================
Metrics computation and evaluation modules.
"""

from evaluation.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_classification_report
)
from evaluation.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history,
    generate_evaluation_report
)

__all__ = [
    'compute_metrics',
    'compute_confusion_matrix',
    'compute_classification_report',
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_training_history',
    'generate_evaluation_report'
]
