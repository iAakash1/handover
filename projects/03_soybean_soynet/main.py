#!/usr/bin/env python3
"""
SoyNet Main Orchestrator
========================
Main entry point for the SoyNet soybean disease detection pipeline.
This script orchestrates all stages: data loading, model creation, training,
evaluation, and XAI generation.

STRICT RULE: This file contains NO ML logic - only function calls and orchestration.

Usage:
    python main.py

All configuration is controlled via config.py.

Author: SoyNet Research Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """
    Main orchestration function.
    
    Pipeline stages:
    1. Load configuration and initialize logging
    2. Set random seeds for reproducibility
    3. Configure GPU/hardware settings
    4. Create output directories
    5. Load and split dataset
    6. Build model using model factory
    7. Train model with callbacks
    8. Evaluate model on test set
    9. Save model and training history
    10. Generate XAI outputs (if enabled)
    """
    # ==========================================================================
    # STAGE 1: Initialize Configuration and Logging
    # ==========================================================================
    
    import config
    from utils.logger import setup_logging, get_logger
    
    # Setup logging
    logger = setup_logging()
    logger = get_logger("SoyNet.Main")
    
    logger.info("=" * 60)
    logger.info("SoyNet - Soybean Disease Detection System")
    logger.info("=" * 60)
    
    # Print configuration
    config.print_config()
    
    # Validate configuration
    try:
        config.validate_config()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # ==========================================================================
    # STAGE 2: Set Random Seeds for Reproducibility
    # ==========================================================================
    
    from utils.helpers import set_seeds
    
    set_seeds(config.RANDOM_SEED)
    logger.info(f"Random seeds set to {config.RANDOM_SEED}")
    
    # ==========================================================================
    # STAGE 3: Configure GPU/Hardware
    # ==========================================================================
    
    from utils.helpers import configure_gpu, get_gpu_info
    
    configure_gpu()
    
    gpu_info = get_gpu_info()
    if gpu_info['num_gpus'] > 0:
        logger.info(f"Using {gpu_info['num_gpus']} GPU(s)")
    else:
        logger.info("Using CPU for training")
    
    # ==========================================================================
    # STAGE 4: Create Output Directories
    # ==========================================================================
    
    from utils.helpers import create_directories
    
    create_directories()
    logger.info("Output directories created")
    
    # ==========================================================================
    # STAGE 5: Load Dataset
    # ==========================================================================
    
    from data.dataset_loader import load_dataset
    from data.preprocessing import get_preprocessing_function
    
    logger.info("Loading dataset...")
    
    # Get preprocessing function for selected model
    preprocess_fn = get_preprocessing_function(config.MODEL_NAME)
    
    # Load train, validation, and test datasets
    train_dataset, val_dataset, test_dataset, dataset_info = load_dataset(
        data_dir=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE,
        preprocess_fn=preprocess_fn
    )
    
    logger.info(f"Dataset loaded: {dataset_info}")
    logger.info(f"Classes: {dataset_info.class_names}")
    logger.info(f"Train: {dataset_info.train_size}, Val: {dataset_info.val_size}, Test: {dataset_info.test_size}")
    
    # Update NUM_CLASSES in config if needed
    num_classes = dataset_info.num_classes
    
    # ==========================================================================
    # STAGE 6: Build Model
    # ==========================================================================
    
    from models.model_factory import create_model
    
    logger.info(f"Building model: {config.MODEL_NAME}")
    
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=num_classes,
        input_shape=config.INPUT_SHAPE,
        freeze_backbone=config.FREEZE_BACKBONE,
        compile_model_flag=True,
        learning_rate=config.INITIAL_LEARNING_RATE
    )
    
    logger.info("Model built successfully")
    
    # ==========================================================================
    # STAGE 7: Train Model
    # ==========================================================================
    
    from models.trainer import train_model
    
    logger.info("Starting training...")
    
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
    
    logger.info("Training completed")
    logger.info(f"Best metrics: {training_info['best_metrics']}")
    
    # ==========================================================================
    # STAGE 8: Evaluate Model
    # ==========================================================================
    
    from evaluation.evaluate import evaluate_model, print_evaluation_summary
    
    logger.info("Evaluating model on test set...")
    
    evaluation_results = evaluate_model(
        model=trained_model,
        test_dataset=test_dataset,
        class_names=dataset_info.class_names,
        save_plots=True,
        plots_dir=config.PLOTS_DIR
    )
    
    print_evaluation_summary(evaluation_results)
    
    # ==========================================================================
    # STAGE 9: Generate XAI Outputs
    # ==========================================================================
    
    if config.ENABLE_XAI:
        from xai.xai_visualizer import generate_xai_outputs
        
        logger.info("Generating XAI outputs...")
        
        generate_xai_outputs(
            model=trained_model,
            dataset_info=dataset_info,
            test_dataset=test_dataset,
            num_samples=config.XAI_NUM_SAMPLES,
            save_dir=config.XAI_OUTPUTS_DIR
        )
        
        logger.info(f"XAI outputs saved to: {config.XAI_OUTPUTS_DIR}")
    else:
        logger.info("XAI disabled in config")
    
    # ==========================================================================
    # STAGE 10: Final Summary
    # ==========================================================================
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {training_info['model_path']}")
    logger.info(f"Training history: {training_info['history_path']}")
    logger.info(f"Evaluation plots: {config.PLOTS_DIR}")
    
    if config.ENABLE_XAI:
        logger.info(f"XAI outputs: {config.XAI_OUTPUTS_DIR}")
    
    logger.info("=" * 60)
    
    return trained_model, evaluation_results


if __name__ == "__main__":
    main()
