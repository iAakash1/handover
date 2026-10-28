#!/usr/bin/env python3
"""
SoyNet Multi-Model Experiment Runner
=====================================
Orchestrates training of multiple backbone architectures with identical settings.
Generates comparison metrics and Grad-CAM outputs for each model.

This script MUST be run for all model comparisons.
Do NOT manually train models one by one.

Usage:
    python run_experiments.py

Author: SoyNet Research Team
Version: 2.0.0 (GPU-Optimized)
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration first (before TensorFlow)
import config


def setup_gpu():
    """Configure GPU settings before any TensorFlow operations."""
    import tensorflow as tf
    
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    
    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("WARNING: No GPU detected! Training will be slow on CPU.")
        print("Please verify CUDA/cuDNN installation.")
        return False
    
    print(f"Found {len(gpus)} GPU(s)")
    
    for gpu in gpus:
        print(f"  Device: {gpu.name}")
        
        # Enable memory growth
        if config.ENABLE_GPU_MEMORY_GROWTH:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Memory growth: ENABLED")
            except RuntimeError as e:
                print(f"  Memory growth error: {e}")
    
    # Enable mixed precision
    if config.ENABLE_MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"Mixed precision: ENABLED (float16)")
    else:
        print(f"Mixed precision: DISABLED")
    
    print("=" * 60)
    return True


def train_single_model(
    model_name: str,
    dataset_cache: Dict = None
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Train a single model and return results.
    
    Args:
        model_name: Name of the backbone to use
        dataset_cache: Cached dataset to ensure identical splits
    
    Returns:
        Tuple of (trained_model, evaluation_results, training_time)
    """
    import tensorflow as tf
    from utils.logger import setup_logging, get_logger
    from utils.helpers import set_seeds, create_directories, log_gpu_usage
    from data.dataset_loader import load_dataset
    from data.preprocessing import get_preprocessing_function
    from models.model_factory import create_model
    from models.trainer import train_model
    from evaluation.evaluate import evaluate_model
    from xai.xai_visualizer import generate_xai_outputs
    
    # Setup logging
    logger = setup_logging()
    logger = get_logger(f"SoyNet.{model_name}")
    
    logger.info("=" * 60)
    logger.info(f"TRAINING MODEL: {model_name.upper()}")
    logger.info("=" * 60)
    
    # Set seeds for reproducibility
    set_seeds(config.RANDOM_SEED)
    
    # Log GPU usage
    log_gpu_usage(logger)
    
    # Get experiment directories
    plots_dir = config.get_plots_dir(model_name)
    xai_dir = config.get_xai_dir(model_name)
    
    # Update global model name temporarily
    original_model_name = config.MODEL_NAME
    config.MODEL_NAME = model_name
    
    try:
        # Get preprocessing function
        preprocess_fn = get_preprocessing_function(model_name)
        
        # Load dataset (use cache if available for identical splits)
        if dataset_cache is not None and 'raw_data' in dataset_cache:
            # Re-create datasets with new preprocessing
            train_dataset, val_dataset, test_dataset, dataset_info = load_dataset(
                data_dir=config.DATA_ROOT,
                batch_size=config.BATCH_SIZE,
                preprocess_fn=preprocess_fn
            )
        else:
            train_dataset, val_dataset, test_dataset, dataset_info = load_dataset(
                data_dir=config.DATA_ROOT,
                batch_size=config.BATCH_SIZE,
                preprocess_fn=preprocess_fn
            )
        
        logger.info(f"Dataset: {dataset_info.train_size} train, {dataset_info.val_size} val, {dataset_info.test_size} test")
        logger.info(f"Classes: {dataset_info.class_names}")
        
        # Create model
        logger.info(f"Creating {model_name} model...")
        model = create_model(
            model_name=model_name,
            num_classes=dataset_info.num_classes,
            input_shape=config.INPUT_SHAPE,
            freeze_backbone=config.FREEZE_BACKBONE,
            compile_model_flag=True,
            learning_rate=config.INITIAL_LEARNING_RATE
        )
        
        # Train model
        logger.info("Starting training...")
        start_time = time.time()
        
        trained_model, training_info = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dataset_info=dataset_info,
            epochs=config.EPOCHS,
            fine_tune=config.FINE_TUNE,
            save_model_flag=True,
            save_history_flag=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {timedelta(seconds=int(training_time))}")
        
        # Evaluate model
        logger.info("Evaluating on test set...")
        evaluation_results = evaluate_model(
            model=trained_model,
            test_dataset=test_dataset,
            class_names=dataset_info.class_names,
            save_plots=True,
            plots_dir=plots_dir
        )
        
        # Generate XAI outputs
        if config.ENABLE_XAI:
            logger.info("Generating Grad-CAM outputs...")
            generate_xai_outputs(
                model=trained_model,
                dataset_info=dataset_info,
                test_dataset=test_dataset,
                num_samples=config.XAI_NUM_SAMPLES,
                save_dir=xai_dir
            )
        
        # Log results summary
        metrics = evaluation_results.get('metrics', {})
        logger.info("=" * 40)
        logger.info(f"RESULTS FOR {model_name.upper()}")
        logger.info("=" * 40)
        logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
        logger.info(f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")
        logger.info(f"  Training Time: {timedelta(seconds=int(training_time))}")
        
        # Add training time to results
        evaluation_results['training_time_seconds'] = training_time
        evaluation_results['model_path'] = str(config.get_model_save_path(model_name))
        
        # Get model size
        model_path = config.get_model_save_path(model_name)
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            evaluation_results['model_size_mb'] = model_size_mb
            logger.info(f"  Model Size: {model_size_mb:.1f} MB")
        
        return trained_model, evaluation_results, training_time
        
    finally:
        # Restore original model name
        config.MODEL_NAME = original_model_name
        
        # Clear memory
        tf.keras.backend.clear_session()


def create_comparison_table(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Create a comprehensive comparison table across all models.
    
    Args:
        all_results: Dictionary mapping model names to their results
    
    Returns:
        Comparison data dictionary
    """
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'models': {},
        'best_by_metric': {},
        'recommendations': {}
    }
    
    # Extract metrics for each model
    for model_name, results in all_results.items():
        metrics = results.get('metrics', {})
        comparison['models'][model_name] = {
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'roc_auc': metrics.get('roc_auc', 0),
            'training_time_seconds': results.get('training_time_seconds', 0),
            'model_size_mb': results.get('model_size_mb', 0)
        }
    
    # Find best model for each metric
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    for metric in metric_names:
        best_model = max(
            comparison['models'].items(),
            key=lambda x: x[1].get(metric, 0)
        )[0]
        comparison['best_by_metric'][metric] = best_model
    
    # Find best deployment candidate (balance of accuracy and size)
    def deployment_score(model_data):
        # Higher accuracy, smaller size = better
        accuracy = model_data.get('accuracy', 0)
        size_mb = model_data.get('model_size_mb', 100)
        # Normalize: accuracy is already 0-1, size is inverted
        return accuracy * 0.7 + (1 - min(size_mb / 200, 1)) * 0.3
    
    best_deployment = max(
        comparison['models'].items(),
        key=lambda x: deployment_score(x[1])
    )[0]
    
    # Recommendations
    comparison['recommendations'] = {
        'best_accuracy': comparison['best_by_metric']['accuracy'],
        'best_recall_disease_safety': comparison['best_by_metric']['recall'],
        'best_deployment_candidate': best_deployment,
        'notes': []
    }
    
    # Add notes based on results
    best_acc_model = comparison['best_by_metric']['accuracy']
    best_recall_model = comparison['best_by_metric']['recall']
    
    if best_acc_model != best_recall_model:
        comparison['recommendations']['notes'].append(
            f"Trade-off detected: {best_acc_model} has best accuracy, "
            f"but {best_recall_model} has best recall (important for disease detection)."
        )
    
    # Check for class imbalance impact
    comparison['recommendations']['notes'].append(
        "Dataset is imbalanced (Disease: 3164, Healthy: 556). "
        "Recall is critical for disease detection safety."
    )
    
    return comparison


def print_comparison_table(comparison: Dict[str, Any]):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    # Header
    print(f"\n{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10} {'Size(MB)':<10}")
    print("-" * 80)
    
    # Data rows
    for model_name, metrics in comparison['models'].items():
        print(f"{model_name:<15} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['roc_auc']:<10.4f} "
              f"{metrics['model_size_mb']:<10.1f}")
    
    print("-" * 80)
    
    # Best by metric
    print("\nBEST MODEL BY METRIC:")
    for metric, model in comparison['best_by_metric'].items():
        print(f"  {metric}: {model}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print(f"  Best Accuracy: {comparison['recommendations']['best_accuracy']}")
    print(f"  Best Recall (Disease Safety): {comparison['recommendations']['best_recall_disease_safety']}")
    print(f"  Best Deployment Candidate: {comparison['recommendations']['best_deployment_candidate']}")
    
    print("\nNOTES:")
    for note in comparison['recommendations']['notes']:
        print(f"  - {note}")
    
    print("=" * 80 + "\n")


def run_all_experiments():
    """
    Run training experiments for all required models.
    
    Models trained:
    - ResNet50 (baseline)
    - EfficientNetB0
    - EfficientNetB3
    - MobileNetV2 (deployment candidate)
    """
    from utils.helpers import create_directories, save_json
    
    print("\n" + "=" * 60)
    print("SOYNET MULTI-MODEL EXPERIMENT RUNNER")
    print("GPU-Accelerated Training Pipeline")
    print("=" * 60 + "\n")
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    if not gpu_available:
        response = input("Continue without GPU? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please configure GPU and try again.")
            return
    
    # Create directories
    create_directories()
    
    # Models to train (MANDATORY - ALL)
    models_to_train = [
        'resnet50',
        'efficientnetb0',
        'efficientnetb3',
        'mobilenetv2'
    ]
    
    print(f"\nModels to train: {models_to_train}")
    print(f"Epochs per model: {config.EPOCHS}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"Mixed precision: {config.ENABLE_MIXED_PRECISION}")
    print("\n")
    
    # Track all results
    all_results = {}
    total_start_time = time.time()
    
    # Train each model
    for i, model_name in enumerate(models_to_train):
        print(f"\n{'#' * 60}")
        print(f"# MODEL {i + 1}/{len(models_to_train)}: {model_name.upper()}")
        print(f"{'#' * 60}\n")
        
        try:
            trained_model, results, training_time = train_single_model(model_name)
            all_results[model_name] = results
            
            print(f"\n✓ {model_name} completed in {timedelta(seconds=int(training_time))}")
            
        except Exception as e:
            print(f"\n✗ ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}
    
    # Calculate total time
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"{'=' * 60}\n")
    
    # Create comparison
    comparison = create_comparison_table(all_results)
    
    # Print comparison
    print_comparison_table(comparison)
    
    # Save comparison to file
    comparison_path = config.RESULTS_DIR / "experiments" / "model_comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(comparison, comparison_path)
    print(f"Comparison saved to: {comparison_path}")
    
    # Save all results
    all_results_path = config.RESULTS_DIR / "experiments" / "all_results.json"
    
    # Convert results to serializable format
    serializable_results = {}
    for model_name, results in all_results.items():
        if 'error' in results:
            serializable_results[model_name] = results
        else:
            serializable_results[model_name] = {
                'metrics': results.get('metrics', {}),
                'training_time_seconds': results.get('training_time_seconds', 0),
                'model_size_mb': results.get('model_size_mb', 0),
                'model_path': results.get('model_path', '')
            }
    
    save_json(serializable_results, all_results_path)
    print(f"All results saved to: {all_results_path}")
    
    # Generate final report
    generate_final_report(comparison, all_results)
    
    return all_results, comparison


def generate_final_report(comparison: Dict, all_results: Dict):
    """Generate a final markdown report with recommendations."""
    report_path = config.RESULTS_DIR / "experiments" / "final_report.md"
    
    report = f"""# SoyNet Multi-Model Experiment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of training four deep learning backbone architectures
for soybean disease detection on the SoyNet dataset using GPU-accelerated training.

### Dataset
- **Disease samples:** 3164
- **Healthy samples:** 556
- **Image size:** {config.IMAGE_SIZE}
- **Class imbalance ratio:** ~5.7:1 (Disease:Healthy)

### Training Configuration
- **Epochs:** {config.EPOCHS}
- **Batch size:** {config.BATCH_SIZE}
- **Early stopping patience:** {config.EARLY_STOPPING_PATIENCE}
- **Mixed precision:** {'Enabled' if config.ENABLE_MIXED_PRECISION else 'Disabled'}

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Size (MB) |
|-------|----------|-----------|--------|----------|---------|-----------|
"""
    
    for model_name, metrics in comparison['models'].items():
        report += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} | {metrics['model_size_mb']:.1f} |\n"
    
    report += f"""
---

## Recommendations

### Best Overall Accuracy
**{comparison['recommendations']['best_accuracy']}**

### Best Recall (Disease Safety Critical)
**{comparison['recommendations']['best_recall_disease_safety']}**

> **Note:** In disease detection, high recall is crucial to minimize false negatives 
> (missing diseased plants). A model with slightly lower accuracy but higher recall 
> may be preferred for real-world agricultural applications.

### Best Deployment Candidate
**{comparison['recommendations']['best_deployment_candidate']}**

This recommendation balances accuracy with model size for efficient deployment
on edge devices or mobile applications.

---

## Key Observations

"""
    
    for note in comparison['recommendations']['notes']:
        report += f"- {note}\n"
    
    report += """
---

## Next Steps

1. **GUI Development:** Implement user interface for inference
2. **API Integration:** Create REST API for model serving
3. **Mobile Deployment:** Optimize MobileNetV2 for on-device inference
4. **Paper Writing:** Document methodology and results for publication

---

## Artifacts Generated

- Trained models: `results/experiments/{model_name}/`
- Evaluation plots: `results/experiments/{model_name}/plots/`
- Grad-CAM outputs: `results/experiments/{model_name}/xai/`
- Comparison data: `results/experiments/model_comparison.json`
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Final report saved to: {report_path}")


if __name__ == "__main__":
    run_all_experiments()
