"""
SoyNet Configuration Module
============================
Central configuration hub for the entire SoyNet pipeline.
All configurable parameters are defined here and imported by other modules.

STRICT RULE: This is the ONLY file where configuration values should be defined.
No other module should override these values.

Author: SoyNet Research Team
Version: 1.0.0
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root directory (automatically determined)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Dataset paths
# NOTE: Using the preprocessed Camera Clicks dataset (256x256 images)
DATA_ROOT = PROJECT_ROOT / "Preprocessing_SoyNet_Data" / "Camera Clicks_256_256"

# Raw data path (for reference, not used in training)
RAW_DATA_ROOT = PROJECT_ROOT / "Raw_SoyNet_Data"

# Output directories
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
XAI_OUTPUTS_DIR = RESULTS_DIR / "xai_outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Image configuration
IMAGE_SIZE = (256, 256)  # Height, Width - matching preprocessed data
IMAGE_CHANNELS = 3  # RGB
INPUT_SHAPE = (*IMAGE_SIZE, IMAGE_CHANNELS)

# Data split ratios (must sum to 1.0)
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of classes (inferred dynamically, but can be set for validation)
# Will be automatically determined from dataset subdirectories
NUM_CLASSES = None  # Set to None for dynamic inference

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available models (supported backbones)
AVAILABLE_MODELS = [
    "resnet50",
    "efficientnetb0",
    "efficientnetb3",
    "mobilenetv2",
    "densenet121",
    "custom_cnn"
]

# Selected model for training (CHANGE THIS TO SWITCH MODELS)
# Options: resnet50, efficientnetb0, efficientnetb3, mobilenetv2, densenet121, custom_cnn
MODEL_NAME = "resnet50"

# Transfer learning configuration
USE_PRETRAINED_WEIGHTS = True  # Use ImageNet weights for pretrained models
FREEZE_BACKBONE = True  # Freeze backbone layers initially
FINE_TUNE = True  # Enable fine-tuning after initial training
FINE_TUNE_AT_LAYER = 100  # Layer index to start fine-tuning (backbone-specific)

# Classification head configuration
DENSE_UNITS = 256  # Number of units in dense layer before output
DROPOUT_RATE = 0.5  # Dropout rate for regularization

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 50  # Increased for GPU training
INITIAL_LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5  # Lower LR for fine-tuning phase

# Optimizer configuration
OPTIMIZER = "adam"  # Options: adam, sgd, rmsprop

# Loss function
LOSS_FUNCTION = "binary_crossentropy"  # For binary classification
# For multi-class: "categorical_crossentropy" or "sparse_categorical_crossentropy"

# Class weights for imbalanced dataset
USE_CLASS_WEIGHTS = True  # Enable class weight computation

# =============================================================================
# CALLBACKS CONFIGURATION
# =============================================================================

# Early stopping
EARLY_STOPPING_PATIENCE = 15  # Increased for deeper training
EARLY_STOPPING_MONITOR = "val_loss"
EARLY_STOPPING_MODE = "min"
EARLY_STOPPING_RESTORE_BEST = True

# Model checkpoint
CHECKPOINT_MONITOR = "val_accuracy"
CHECKPOINT_MODE = "max"
CHECKPOINT_SAVE_BEST_ONLY = True

# Learning rate reduction
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_MIN_LR = 1e-7

# =============================================================================
# DATA AUGMENTATION CONFIGURATION
# =============================================================================

# Enable/disable augmentation
ENABLE_AUGMENTATION = True

# Augmentation parameters (only applied to training data)
AUGMENTATION_CONFIG = {
    "rotation_range": 20,  # Degrees
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "vertical_flip": False,
    "zoom_range": 0.15,
    "brightness_range": (0.8, 1.2),
    "fill_mode": "nearest"
}

# =============================================================================
# EXPLAINABLE AI (XAI) CONFIGURATION
# =============================================================================

# Enable/disable XAI pipeline
ENABLE_XAI = True

# Sample image for XAI visualization (will be selected automatically if None)
XAI_SAMPLE_IMAGE_PATH = None  # Set to a specific image path or None for auto-selection

# Number of samples for XAI visualization
XAI_NUM_SAMPLES = 10

# Grad-CAM configuration
GRADCAM_COLORMAP = "jet"  # Colormap for heatmap
GRADCAM_ALPHA = 0.4  # Overlay transparency

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_TO_CONSOLE = True

# =============================================================================
# GPU/HARDWARE CONFIGURATION
# =============================================================================

# Memory growth for GPU (prevents TensorFlow from allocating all GPU memory)
ENABLE_GPU_MEMORY_GROWTH = True

# Mixed precision training (faster on compatible GPUs like RTX 4500)
ENABLE_MIXED_PRECISION = True

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Experiment name (used for organizing results per model)
# Set to None for auto-generation using MODEL_NAME
EXPERIMENT_NAME = None

# Fine-tuning layer indices per backbone (where to start unfreezing)
# These are optimized starting points for each architecture
FINE_TUNE_LAYERS_MAP = {
    'resnet50': 143,
    'efficientnetb0': 100,
    'efficientnetb3': 200,
    'mobilenetv2': 100,
    'densenet121': 300,
    'custom_cnn': 0
}

# =============================================================================
# MODEL-SPECIFIC PREPROCESSING FUNCTIONS
# =============================================================================

# Maps model names to their preprocessing functions
# Will be populated by the model factory
MODEL_PREPROCESS_MAP = {
    "resnet50": "tensorflow.keras.applications.resnet50.preprocess_input",
    "efficientnetb0": "tensorflow.keras.applications.efficientnet.preprocess_input",
    "efficientnetb3": "tensorflow.keras.applications.efficientnet.preprocess_input",
    "mobilenetv2": "tensorflow.keras.applications.mobilenet_v2.preprocess_input",
    "densenet121": "tensorflow.keras.applications.densenet.preprocess_input",
    "custom_cnn": None  # Custom CNN uses simple normalization
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_experiment_name(model_name: str = None) -> str:
    """Get the experiment name for organizing outputs."""
    if EXPERIMENT_NAME:
        return EXPERIMENT_NAME
    return model_name or MODEL_NAME


def get_experiment_dir(model_name: str = None) -> Path:
    """Get the experiment directory for a model."""
    exp_name = get_experiment_name(model_name)
    return RESULTS_DIR / "experiments" / exp_name


def get_model_save_path(model_name: str = None) -> Path:
    """Get the path for saving the trained model."""
    name = model_name or MODEL_NAME
    exp_dir = get_experiment_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir / f"soynet_{name}_best.keras"


def get_history_save_path(model_name: str = None) -> Path:
    """Get the path for saving training history."""
    name = model_name or MODEL_NAME
    exp_dir = get_experiment_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir / f"soynet_{name}_history.json"


def get_plots_dir(model_name: str = None) -> Path:
    """Get the plots directory for a model experiment."""
    exp_dir = get_experiment_dir(model_name)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def get_xai_dir(model_name: str = None) -> Path:
    """Get the XAI outputs directory for a model experiment."""
    exp_dir = get_experiment_dir(model_name)
    xai_dir = exp_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    return xai_dir


def get_fine_tune_layer(model_name: str = None) -> int:
    """Get the fine-tuning layer index for a specific backbone."""
    name = (model_name or MODEL_NAME).lower()
    return FINE_TUNE_LAYERS_MAP.get(name, FINE_TUNE_AT_LAYER)


def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Check split ratios
    total_split = TRAIN_SPLIT + VALIDATION_SPLIT + TEST_SPLIT
    if abs(total_split - 1.0) > 1e-6:
        errors.append(f"Split ratios must sum to 1.0, got {total_split}")
    
    # Check model name
    if MODEL_NAME not in AVAILABLE_MODELS:
        errors.append(f"Invalid MODEL_NAME: {MODEL_NAME}. Must be one of {AVAILABLE_MODELS}")
    
    # Check data directory exists
    if not DATA_ROOT.exists():
        errors.append(f"Data directory does not exist: {DATA_ROOT}")
    
    # Check image size
    if IMAGE_SIZE[0] <= 0 or IMAGE_SIZE[1] <= 0:
        errors.append(f"Invalid IMAGE_SIZE: {IMAGE_SIZE}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True


def print_config():
    """Print current configuration for logging."""
    print("\n" + "=" * 60)
    print("SoyNet Configuration")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {INITIAL_LEARNING_RATE}")
    print(f"Data Root: {DATA_ROOT}")
    print(f"Use Class Weights: {USE_CLASS_WEIGHTS}")
    print(f"Enable Augmentation: {ENABLE_AUGMENTATION}")
    print(f"Enable XAI: {ENABLE_XAI}")
    print(f"Freeze Backbone: {FREEZE_BACKBONE}")
    print(f"Fine-Tune: {FINE_TUNE}")
    print("=" * 60 + "\n")


# Validate configuration on import (optional, can be disabled)
# validate_config()
