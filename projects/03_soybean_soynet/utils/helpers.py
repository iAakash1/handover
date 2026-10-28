"""
SoyNet Helpers Module
=====================
Utility functions for reproducibility, directory management, and GPU configuration.

Author: SoyNet Research Team
Version: 1.0.0
"""

import os
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

import config


def set_seeds(seed: int = None):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (uses config.RANDOM_SEED if None)
    """
    seed = seed or config.RANDOM_SEED
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow (imported here to avoid circular imports)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # Set Python hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # NOTE: TF_DETERMINISTIC_OPS disabled for GPU compatibility
        # Not all GPU ops (e.g., UnsortedSegmentSum) have deterministic implementations
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
    except ImportError:
        pass
    
    print(f"Random seeds set to: {seed}")


def create_directories():
    """
    Create all necessary output directories for the project.
    """
    directories = [
        config.SAVED_MODELS_DIR,
        config.RESULTS_DIR,
        config.PLOTS_DIR,
        config.XAI_OUTPUTS_DIR,
        config.LOGS_DIR
    ]
    
    for directory in directories:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory ensured: {directory}")


def configure_gpu():
    """
    Configure GPU settings for TensorFlow.
    Enables memory growth and optionally mixed precision.
    """
    try:
        import tensorflow as tf
        
        # Get list of GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")
            
            for gpu in gpus:
                # Enable memory growth
                if config.ENABLE_GPU_MEMORY_GROWTH:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for: {gpu}")
            
            # Enable mixed precision if configured
            if config.ENABLE_MIXED_PRECISION:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision (float16) enabled")
        else:
            print("No GPU found. Using CPU.")
            
    except ImportError:
        print("TensorFlow not available for GPU configuration")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "num_gpus": 0,
        "gpu_names": [],
        "memory_info": []
    }
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        info["num_gpus"] = len(gpus)
        
        for gpu in gpus:
            info["gpu_names"].append(gpu.name)
            
            # Try to get memory info
            try:
                mem_info = tf.config.experimental.get_memory_info(gpu.name)
                info["memory_info"].append(mem_info)
            except:
                pass
                
    except ImportError:
        pass
    
    return info


def log_gpu_usage(logger=None):
    """
    Log GPU name, memory usage, and CUDA status.
    Provides runtime verification that GPU is being used.
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            msg = "WARNING: No GPU detected! Training will use CPU."
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return
        
        # Log GPU info
        for i, gpu in enumerate(gpus):
            gpu_name = gpu.name
            msg = f"GPU {i}: {gpu_name}"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            
            # Try to get memory info (may not be available on all systems)
            try:
                mem_info = tf.config.experimental.get_memory_info(gpu.name)
                current_mb = mem_info.get('current', 0) / (1024 * 1024)
                peak_mb = mem_info.get('peak', 0) / (1024 * 1024)
                msg = f"  Memory - Current: {current_mb:.1f} MB, Peak: {peak_mb:.1f} MB"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
            except Exception:
                pass
        
        # Log mixed precision status
        policy = tf.keras.mixed_precision.global_policy()
        msg = f"Mixed Precision Policy: {policy.name}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
            
    except ImportError:
        pass
    except Exception as e:
        if logger:
            logger.error(f"Error logging GPU usage: {e}")
        else:
            print(f"Error logging GPU usage: {e}")


def verify_gpu_available() -> bool:
    """
    Verify GPU is available and properly configured.
    Returns True if GPU is available, False otherwise.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("ERROR: No GPU detected! Please verify CUDA installation.")
            return False
        
        # Test GPU by creating a simple tensor
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            result = tf.reduce_sum(test_tensor)
        
        print(f"GPU Verification PASSED: {len(gpus)} GPU(s) available")
        return True
        
    except Exception as e:
        print(f"GPU Verification FAILED: {e}")
        return False


def save_json(data: Dict[str, Any], filepath: Path):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    data = convert_numpy(data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved: {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp() -> str:
    """
    Get current timestamp as formatted string.
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def compute_class_weights(class_counts: Dict[int, int]) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    Uses balanced class weighting formula.
    
    Args:
        class_counts: Dictionary mapping class index to sample count
    
    Returns:
        Dictionary mapping class index to weight
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        # Balanced class weight formula
        weight = total_samples / (num_classes * count)
        class_weights[class_idx] = weight
    
    return class_weights


def get_class_distribution(labels: np.ndarray) -> Dict[int, int]:
    """
    Get class distribution from labels array.
    
    Args:
        labels: Array of class labels
    
    Returns:
        Dictionary mapping class index to count
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def print_class_distribution(class_names: List[str], class_counts: Dict[int, int]):
    """
    Print formatted class distribution.
    
    Args:
        class_names: List of class names
        class_counts: Dictionary mapping class index to count
    """
    total = sum(class_counts.values())
    
    print("\nClass Distribution:")
    print("-" * 40)
    
    for idx, name in enumerate(class_names):
        count = class_counts.get(idx, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {name}: {count} ({percentage:.1f}%)")
    
    print(f"  Total: {total}")
    print("-" * 40)


def get_sample_image_path(data_dir: Path) -> Optional[Path]:
    """
    Get a sample image path from the dataset for XAI visualization.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        Path to a sample image or None if not found
    """
    data_dir = Path(data_dir)
    
    # Look for image files in subdirectories
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            for ext in image_extensions:
                images = list(class_dir.glob(f"*{ext}"))
                if images:
                    return images[0]
    
    return None


def count_parameters(model) -> Tuple[int, int]:
    """
    Count trainable and non-trainable parameters in a model.
    
    Args:
        model: Keras model
    
    Returns:
        Tuple of (trainable_params, non_trainable_params)
    """
    trainable = sum([np.prod(w.shape) for w in model.trainable_weights])
    non_trainable = sum([np.prod(w.shape) for w in model.non_trainable_weights])
    
    return int(trainable), int(non_trainable)


def print_model_summary(model, model_name: str):
    """
    Print formatted model summary.
    
    Args:
        model: Keras model
        model_name: Name of the model
    """
    trainable, non_trainable = count_parameters(model)
    total = trainable + non_trainable
    
    print("\n" + "=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Non-trainable Parameters: {non_trainable:,}")
    print("=" * 60 + "\n")
