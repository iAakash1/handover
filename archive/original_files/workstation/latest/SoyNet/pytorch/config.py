"""
SoyNet PyTorch Configuration
=============================
Central configuration for PyTorch-based training pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_ROOT = PROJECT_ROOT / "Preprocessing_SoyNet_Data" / "Camera Clicks_256_256"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"

# =============================================================================
# DATASET CONFIG
# =============================================================================

IMAGE_SIZE = (256, 256)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Class names are dynamically detected from ImageFolder.classes
# Do NOT hardcode class names here - they must match dataset folders exactly

# =============================================================================
# MODEL CONFIG
# =============================================================================

MODELS_TO_TRAIN = [
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b3",
    "mobilenetv2"
]

# =============================================================================
# TRAINING CONFIG
# =============================================================================

BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
WEIGHT_DECAY = 1e-4

# Early stopping
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.2

# Phase 1: frozen backbone epochs, Phase 2: fine-tuning epochs
FREEZE_EPOCHS = 10
FINE_TUNE_EPOCHS = 40

# Dropout
DROPOUT_RATE = 0.5

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

AUGMENTATION_CONFIG = {
    "rotation": 20,
    "horizontal_flip": True,
    "vertical_flip": False,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "hue": 0.1
    },
    "random_crop_scale": (0.85, 1.0),
}

# =============================================================================
# GRAD-CAM
# =============================================================================

GRADCAM_NUM_SAMPLES = 5  # Samples per class
GRADCAM_COLORMAP = "jet"
GRADCAM_ALPHA = 0.4

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_dir(model_name: str) -> Path:
    """Get output directory for a specific model."""
    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def get_device():
    """Get PyTorch device (CUDA or CPU)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! GPU is required for training.")
    return torch.device("cuda")
