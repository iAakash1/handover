"""
SoyNet Evaluation Module
=========================
Test set evaluation with comprehensive metrics and confusion matrix.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from . import config


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_dir: Path
) -> Dict:
    """
    Evaluate model on test set and save metrics.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names (from ImageFolder.classes)
        device: PyTorch device
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Get raw logits
            outputs = model(images)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Threshold predictions
            preds = (probs > 0.5).astype(int)
            
            all_labels.extend(labels)
            all_probs.extend(probs)
            all_preds.extend(preds)
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),  # Use probabilities for AUC
        "total_samples": int(len(y_true)),
        "class_names": class_names,
    }
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Build confusion matrix dict with dynamic class names
    confusion_dict = {
        "raw_counts": cm.tolist(),
        "normalized": cm_normalized.tolist(),
        "class_names": class_names,
        "labels": {
            "true_positive": int(cm[1, 1]) if cm.shape == (2, 2) else None,
            "true_negative": int(cm[0, 0]) if cm.shape == (2, 2) else None,
            "false_positive": int(cm[0, 1]) if cm.shape == (2, 2) else None,
            "false_negative": int(cm[1, 0]) if cm.shape == (2, 2) else None,
        }
    }
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.json"
    with open(cm_path, 'w') as f:
        json.dump(confusion_dict, f, indent=2)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SET EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*50}\n")
    
    return metrics


def get_predictions_with_paths(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[List[str], List[int], List[float]]:
    """
    Get predictions with file paths for Grad-CAM sample selection.
    
    Returns:
        Tuple of (file_paths, true_labels, probabilities)
    """
    model.eval()
    
    file_paths = []
    labels = []
    probs = []
    
    # Access the underlying dataset through the subset
    dataset = test_loader.dataset
    if hasattr(dataset, 'dataset'):
        # It's a Subset
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))
    
    with torch.no_grad():
        batch_idx = 0
        for images, batch_labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            batch_probs = torch.sigmoid(outputs).cpu().numpy()
            
            for i, (label, prob) in enumerate(zip(batch_labels.numpy(), batch_probs)):
                # Get file path from dataset
                sample_idx = indices[batch_idx * test_loader.batch_size + i] if batch_idx * test_loader.batch_size + i < len(indices) else None
                if sample_idx is not None:
                    file_path, _ = base_dataset.samples[sample_idx]
                    file_paths.append(file_path)
                    labels.append(int(label))
                    probs.append(float(prob))
            
            batch_idx += 1
    
    return file_paths, labels, probs
