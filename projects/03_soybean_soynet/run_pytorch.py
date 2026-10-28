#!/usr/bin/env python3
"""
SoyNet PyTorch Training Pipeline
=================================
Single entry point for training all models.

Usage:
    python run_pytorch.py           # Full training
    python run_pytorch.py --dry-run # Verify setup without training
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# ENVIRONMENT VERIFICATION
# =============================================================================

def verify_environment():
    """Verify Python, PyTorch, CUDA, and dependencies."""
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch not installed!")
        sys.exit(1)
    
    # CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available! GPU is required.")
        sys.exit(1)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # timm
    try:
        import timm
        print(f"timm version: {timm.__version__}")
    except ImportError:
        print("ERROR: timm not installed! Run: pip install timm")
        sys.exit(1)
    
    # sklearn
    try:
        import sklearn
        print(f"scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("ERROR: scikit-learn not installed! Run: pip install scikit-learn")
        sys.exit(1)
    
    # matplotlib
    try:
        import matplotlib
        print(f"matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("ERROR: matplotlib not installed! Run: pip install matplotlib")
        sys.exit(1)
    
    print("=" * 60)
    print("All dependencies verified successfully!")
    print("=" * 60 + "\n")
    
    return torch.device("cuda")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(dry_run: bool = False):
    """
    Run the complete training pipeline for all models.
    
    Args:
        dry_run: If True, verify setup without training
    """
    # Verify environment first
    device = verify_environment()
    
    # Import PyTorch modules (absolute imports from pytorch package)
    from pytorch import config
    from pytorch.dataset import get_dataloaders
    from pytorch.models import ModelFactory, get_model_info
    from pytorch.trainer import train_model
    from pytorch.evaluation import evaluate_model
    from pytorch.gradcam import generate_gradcam_for_model
    
    # Print configuration
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Results dir: {config.RESULTS_DIR}")
    print(f"Models to train: {config.MODELS_TO_TRAIN}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Freeze epochs: {config.FREEZE_EPOCHS}")
    print(f"Fine-tune epochs: {config.FINE_TUNE_EPOCHS}")
    print(f"Initial LR: {config.INITIAL_LR}")
    print(f"Fine-tune LR: {config.FINE_TUNE_LR}")
    print("=" * 60 + "\n")
    
    # Create dataloaders
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders()
    
    # Use dynamic class names from ImageFolder
    class_names = dataset_info["class_names"]
    class_weights = dataset_info["class_weights"]
    
    print(f"Number of classes: {dataset_info['num_classes']}")
    print(f"Class names: {class_names}")
    print(f"Class weights: {class_weights}")
    print(f"Train size: {dataset_info['train_size']}")
    print(f"Val size: {dataset_info['val_size']}")
    print(f"Test size: {dataset_info['test_size']}")
    print("=" * 60 + "\n")
    
    # Save dataset info
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_info_path = config.RESULTS_DIR / "dataset_info.json"
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Dataset info saved to: {dataset_info_path}\n")
    
    # Dry run: just verify models can be created
    if dry_run:
        print("=" * 60)
        print("DRY RUN: Verifying model creation")
        print("=" * 60)
        
        for model_name in config.MODELS_TO_TRAIN:
            print(f"\nCreating {model_name}...")
            model = ModelFactory.create_model(model_name, pretrained=True, freeze=True)
            model = model.to(device)
            info = get_model_info(model, model_name)
            print(f"  Architecture: {info['architecture']}")
            print(f"  Total params: {info['total_params']:,}")
            print(f"  Trainable params: {info['trainable_params']:,}")
            print(f"  Grad-CAM layer: {info['gradcam_layer']}")
            del model
        
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE - All models verified successfully!")
        print("Run without --dry-run to start training.")
        print("=" * 60)
        return
    
    # Full training pipeline
    print("=" * 60)
    print("STARTING FULL TRAINING PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    results_summary = {}
    
    for model_idx, model_name in enumerate(config.MODELS_TO_TRAIN):
        print("\n" + "#" * 60)
        print(f"MODEL {model_idx + 1}/{len(config.MODELS_TO_TRAIN)}: {model_name.upper()}")
        print("#" * 60)
        
        # Get output directory
        output_dir = config.get_model_dir(model_name)
        
        # Create model
        print(f"\nCreating {model_name}...")
        model = ModelFactory.create_model(model_name, pretrained=True, freeze=True)
        info = get_model_info(model, model_name)
        print(f"Total params: {info['total_params']:,}")
        print(f"Trainable params (frozen): {info['trainable_params']:,}")
        
        # Train model
        print(f"\nTraining {model_name}...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            model_name=model_name,
            device=device
        )
        
        # Evaluate on test set
        print(f"\nEvaluating {model_name} on test set...")
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            output_dir=output_dir
        )
        
        # Generate Grad-CAM visualizations
        print(f"\nGenerating Grad-CAM for {model_name}...")
        gradcam_metadata = generate_gradcam_for_model(
            model=model,
            model_name=model_name,
            test_loader=test_loader,
            class_names=class_names,
            output_dir=output_dir,
            num_samples_per_class=config.GRADCAM_NUM_SAMPLES
        )
        
        # Store results
        results_summary[model_name] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "output_dir": str(output_dir)
        }
        
        # Cleanup
        del model
        import torch
        torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print(f"COMPLETED: {model_name}")
        print(f"{'='*60}\n")
    
    # Save overall summary
    summary_path = config.RESULTS_DIR / "training_summary.json"
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "models_trained": list(config.MODELS_TO_TRAIN),
        "dataset_info": dataset_info,
        "results": results_summary
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\nRESULTS SUMMARY:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10} {'ROC-AUC':>10}")
    print("-" * 60)
    for model_name, result in results_summary.items():
        print(f"{model_name:<20} {result['accuracy']:>10.4f} {result['f1_score']:>10.4f} {result['roc_auc']:>10.4f}")
    print("-" * 60)
    print(f"\nAll outputs saved to: {config.RESULTS_DIR}")
    print(f"Summary saved to: {summary_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SoyNet PyTorch Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pytorch.py           # Full training for all 4 models
    python run_pytorch.py --dry-run # Verify setup without training
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify setup and model creation without training"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
