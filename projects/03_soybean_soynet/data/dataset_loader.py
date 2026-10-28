"""
SoyNet Dataset Loader Module
============================
Handles data loading using TensorFlow/Keras data pipelines.
Supports dynamic class inference, stratified splitting, and class weight computation.

Author: SoyNet Research Team
Version: 1.0.0
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
from utils.logger import get_logger
from utils.helpers import compute_class_weights, print_class_distribution


@dataclass
class DatasetInfo:
    """
    Container for dataset information.
    """
    num_classes: int
    class_names: List[str]
    class_indices: Dict[str, int]
    train_size: int
    val_size: int
    test_size: int
    class_weights: Optional[Dict[int, float]] = None
    
    def __str__(self):
        return (
            f"DatasetInfo(\n"
            f"  num_classes={self.num_classes},\n"
            f"  class_names={self.class_names},\n"
            f"  train_size={self.train_size},\n"
            f"  val_size={self.val_size},\n"
            f"  test_size={self.test_size}\n"
            f")"
        )


def get_class_names(data_dir: Path = None) -> List[str]:
    """
    Dynamically infer class names from dataset subdirectories.
    
    Args:
        data_dir: Root data directory (uses config if None)
    
    Returns:
        Sorted list of class names
    """
    data_dir = Path(data_dir or config.DATA_ROOT)
    
    class_names = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            class_names.append(item.name)
    
    return class_names


def _collect_file_paths(data_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    """
    Collect all image file paths and their labels.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        Tuple of (file_paths, labels, class_names)
    """
    logger = get_logger("SoyNet.Data")
    
    data_dir = Path(data_dir)
    class_names = get_class_names(data_dir)
    
    if not class_names:
        raise ValueError(f"No class directories found in {data_dir}")
    
    logger.info(f"Found {len(class_names)} classes: {class_names}")
    
    file_paths = []
    labels = []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        
        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() in image_extensions:
                file_paths.append(str(file_path))
                labels.append(class_idx)
    
    logger.info(f"Collected {len(file_paths)} images")
    
    return file_paths, labels, class_names


def _stratified_split(
    file_paths: List[str],
    labels: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """
    Perform stratified train/val/test split.
    
    Args:
        file_paths: List of file paths
        labels: List of labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
    
    Returns:
        Tuple of ((train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels))
    """
    logger = get_logger("SoyNet.Data")
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed
    )
    
    logger.info(f"Split sizes - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def _create_tf_dataset(
    file_paths: List[str],
    labels: List[int],
    batch_size: int,
    shuffle: bool = False,
    augment: bool = False,
    preprocess_fn=None
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from file paths and labels.
    
    Args:
        file_paths: List of image file paths
        labels: List of integer labels
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        augment: Whether to apply augmentation
        preprocess_fn: Preprocessing function to apply
    
    Returns:
        tf.data.Dataset
    """
    # Convert to tensors
    paths_tensor = tf.constant(file_paths)
    labels_tensor = tf.constant(labels, dtype=tf.int32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    
    # Shuffle if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=config.RANDOM_SEED)
    
    # Load and preprocess images
    def load_and_preprocess(path, label):
        # Read file
        image = tf.io.read_file(path)
        
        # Decode image
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize (should already be correct size, but ensure consistency)
        image = tf.image.resize(image, config.IMAGE_SIZE)
        
        # Apply preprocessing if provided
        if preprocess_fn is not None:
            image = preprocess_fn(image)
        else:
            # Default: normalize to [0, 1]
            image = tf.cast(image, tf.float32) / 255.0
        
        return image, label
    
    # Apply preprocessing
    dataset = dataset.map(
        load_and_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation if needed
    if augment and config.ENABLE_AUGMENTATION:
        from data.augmentation import get_augmentation_layer
        augment_layer = get_augmentation_layer()
        
        def apply_augment(image, label):
            image = augment_layer(image, training=True)
            return image, label
        
        dataset = dataset.map(apply_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def load_dataset(
    data_dir: Path = None,
    batch_size: int = None,
    seed: int = None,
    preprocess_fn=None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, DatasetInfo]:
    """
    Load dataset and create train/val/test splits.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        seed: Random seed
        preprocess_fn: Preprocessing function for images
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, dataset_info)
    """
    logger = get_logger("SoyNet.Data")
    logger.info("Loading dataset...")
    
    # Use config defaults
    data_dir = Path(data_dir or config.DATA_ROOT)
    batch_size = batch_size or config.BATCH_SIZE
    seed = seed or config.RANDOM_SEED
    
    # Collect file paths and labels
    file_paths, labels, class_names = _collect_file_paths(data_dir)
    
    # Print class distribution
    from utils.helpers import get_class_distribution
    class_distribution = get_class_distribution(np.array(labels))
    print_class_distribution(class_names, class_distribution)
    
    # Perform stratified split
    train_data, val_data, test_data = _stratified_split(
        file_paths, labels,
        train_ratio=config.TRAIN_SPLIT,
        val_ratio=config.VALIDATION_SPLIT,
        test_ratio=config.TEST_SPLIT,
        seed=seed
    )
    
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    # Compute class weights if enabled
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        train_distribution = get_class_distribution(np.array(train_labels))
        class_weights = compute_class_weights(train_distribution)
        logger.info(f"Class weights: {class_weights}")
    
    # Create TF datasets
    train_dataset = _create_tf_dataset(
        train_paths, train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        preprocess_fn=preprocess_fn
    )
    
    val_dataset = _create_tf_dataset(
        val_paths, val_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        preprocess_fn=preprocess_fn
    )
    
    test_dataset = _create_tf_dataset(
        test_paths, test_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        preprocess_fn=preprocess_fn
    )
    
    # Create dataset info
    dataset_info = DatasetInfo(
        num_classes=len(class_names),
        class_names=class_names,
        class_indices={name: idx for idx, name in enumerate(class_names)},
        train_size=len(train_paths),
        val_size=len(val_paths),
        test_size=len(test_paths),
        class_weights=class_weights
    )
    
    logger.info(f"Dataset loaded successfully")
    logger.info(str(dataset_info))
    
    return train_dataset, val_dataset, test_dataset, dataset_info


def create_data_generators(
    data_dir: Path = None,
    batch_size: int = None,
    seed: int = None,
    validation_split: float = None
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, 
           tf.keras.preprocessing.image.DirectoryIterator,
           Dict[str, int]]:
    """
    Alternative: Create data generators using Keras ImageDataGenerator.
    This is an alternative approach that uses Keras' built-in utilities.
    
    Note: This method requires a different data structure (train/val/test folders).
    Use load_dataset() for the current data structure.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        seed: Random seed
        validation_split: Validation split ratio
    
    Returns:
        Tuple of (train_generator, val_generator, class_indices)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    data_dir = Path(data_dir or config.DATA_ROOT)
    batch_size = batch_size or config.BATCH_SIZE
    seed = seed or config.RANDOM_SEED
    validation_split = validation_split or config.VALIDATION_SPLIT
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=config.AUGMENTATION_CONFIG.get('rotation_range', 20),
        width_shift_range=config.AUGMENTATION_CONFIG.get('width_shift_range', 0.1),
        height_shift_range=config.AUGMENTATION_CONFIG.get('height_shift_range', 0.1),
        horizontal_flip=config.AUGMENTATION_CONFIG.get('horizontal_flip', True),
        zoom_range=config.AUGMENTATION_CONFIG.get('zoom_range', 0.15),
        validation_split=validation_split
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='binary' if len(get_class_names(data_dir)) == 2 else 'categorical',
        subset='training',
        seed=seed
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='binary' if len(get_class_names(data_dir)) == 2 else 'categorical',
        subset='validation',
        seed=seed
    )
    
    return train_generator, val_generator, train_generator.class_indices


def get_sample_batch(dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a sample batch from a dataset for visualization or testing.
    
    Args:
        dataset: TensorFlow dataset
    
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    for images, labels in dataset.take(1):
        return images.numpy(), labels.numpy()


def get_test_images(
    data_dir: Path = None,
    num_samples: int = 10,
    class_name: str = None
) -> Tuple[List[str], List[int]]:
    """
    Get sample image paths for testing/XAI visualization.
    
    Args:
        data_dir: Root data directory
        num_samples: Number of samples to return
        class_name: Specific class to sample from (None for all classes)
    
    Returns:
        Tuple of (file_paths, labels)
    """
    data_dir = Path(data_dir or config.DATA_ROOT)
    class_names = get_class_names(data_dir)
    
    file_paths = []
    labels = []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for class_idx, name in enumerate(class_names):
        if class_name is not None and name != class_name:
            continue
            
        class_dir = data_dir / name
        class_images = [
            str(f) for f in class_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        # Sample images from this class
        n_samples = min(num_samples // len(class_names) + 1, len(class_images))
        sampled = np.random.choice(class_images, size=n_samples, replace=False)
        
        file_paths.extend(sampled.tolist())
        labels.extend([class_idx] * n_samples)
    
    # Limit to requested number
    if len(file_paths) > num_samples:
        indices = np.random.choice(len(file_paths), size=num_samples, replace=False)
        file_paths = [file_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    
    return file_paths, labels
