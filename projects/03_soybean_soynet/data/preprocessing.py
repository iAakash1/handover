"""
SoyNet Preprocessing Module
===========================
Data preprocessing utilities for image loading and model-specific normalization.
NOTE: This module does NOT modify the original dataset files.

Author: SoyNet Research Team
Version: 1.0.0
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

import config


def get_preprocessing_function(model_name: str = None) -> Optional[Callable]:
    """
    Get the appropriate preprocessing function for a specific model.
    Each pretrained model expects inputs normalized in a specific way.
    
    Args:
        model_name: Name of the model (uses config.MODEL_NAME if None)
    
    Returns:
        Preprocessing function or None for simple normalization
    """
    model_name = (model_name or config.MODEL_NAME).lower()
    
    if model_name == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    
    elif model_name in ['efficientnetb0', 'efficientnetb3']:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        return preprocess_input
    
    elif model_name == 'mobilenetv2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    
    elif model_name == 'densenet121':
        from tensorflow.keras.applications.densenet import preprocess_input
        return preprocess_input
    
    elif model_name == 'custom_cnn':
        # Custom CNN uses simple [0, 1] normalization
        return None
    
    else:
        # Default: no preprocessing (assume [0, 1] normalization)
        return None


def preprocess_image(
    image: tf.Tensor,
    model_name: str = None,
    normalize: bool = True
) -> tf.Tensor:
    """
    Preprocess a single image for model input.
    
    Args:
        image: Input image tensor (uint8, [0, 255])
        model_name: Name of the model
        normalize: Whether to apply normalization
    
    Returns:
        Preprocessed image tensor
    """
    # Ensure float32
    image = tf.cast(image, tf.float32)
    
    # Get preprocessing function
    preprocess_fn = get_preprocessing_function(model_name)
    
    if preprocess_fn is not None:
        # Apply model-specific preprocessing
        image = preprocess_fn(image)
    elif normalize:
        # Simple [0, 1] normalization
        image = image / 255.0
    
    return image


def load_image(
    image_path: str,
    target_size: Tuple[int, int] = None,
    model_name: str = None
) -> np.ndarray:
    """
    Load and preprocess a single image from disk.
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
        model_name: Model name for preprocessing
    
    Returns:
        Preprocessed image as numpy array
    """
    target_size = target_size or config.IMAGE_SIZE
    
    # Load with PIL for robust image handling
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
    
    # Convert to numpy
    img_array = np.array(img, dtype=np.float32)
    
    # Apply preprocessing
    preprocess_fn = get_preprocessing_function(model_name)
    
    if preprocess_fn is not None:
        img_array = preprocess_fn(img_array)
    else:
        img_array = img_array / 255.0
    
    return img_array


def load_image_tf(
    image_path: str,
    target_size: Tuple[int, int] = None,
    model_name: str = None
) -> tf.Tensor:
    """
    Load and preprocess a single image using TensorFlow operations.
    
    Args:
        image_path: Path to image file
        target_size: Target size (height, width)
        model_name: Model name for preprocessing
    
    Returns:
        Preprocessed image as TensorFlow tensor
    """
    target_size = target_size or config.IMAGE_SIZE
    
    # Read file
    image = tf.io.read_file(image_path)
    
    # Decode image
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize
    image = tf.image.resize(image, target_size)
    
    # Apply preprocessing
    image = preprocess_image(image, model_name)
    
    return image


def load_and_preprocess_batch(
    image_paths: list,
    target_size: Tuple[int, int] = None,
    model_name: str = None
) -> np.ndarray:
    """
    Load and preprocess a batch of images.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size (height, width)
        model_name: Model name for preprocessing
    
    Returns:
        Batch of preprocessed images as numpy array
    """
    target_size = target_size or config.IMAGE_SIZE
    
    images = []
    for path in image_paths:
        img = load_image(path, target_size, model_name)
        images.append(img)
    
    return np.array(images)


def validate_image(image_path: str) -> bool:
    """
    Validate that an image file can be loaded correctly.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False


def get_image_info(image_path: str) -> dict:
    """
    Get information about an image file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dictionary with image information
    """
    try:
        img = Image.open(image_path)
        return {
            'path': str(image_path),
            'size': img.size,  # (width, height)
            'mode': img.mode,
            'format': img.format
        }
    except Exception as e:
        return {
            'path': str(image_path),
            'error': str(e)
        }


def create_preprocessing_layer(model_name: str = None) -> tf.keras.layers.Layer:
    """
    Create a Keras layer for preprocessing that can be included in the model.
    
    Args:
        model_name: Model name for preprocessing
    
    Returns:
        Preprocessing layer
    """
    model_name = (model_name or config.MODEL_NAME).lower()
    
    if model_name == 'resnet50':
        return tf.keras.layers.Lambda(
            lambda x: tf.keras.applications.resnet50.preprocess_input(x),
            name='resnet_preprocessing'
        )
    
    elif model_name in ['efficientnetb0', 'efficientnetb3']:
        return tf.keras.layers.Lambda(
            lambda x: tf.keras.applications.efficientnet.preprocess_input(x),
            name='efficientnet_preprocessing'
        )
    
    elif model_name == 'mobilenetv2':
        return tf.keras.layers.Lambda(
            lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x),
            name='mobilenet_preprocessing'
        )
    
    elif model_name == 'densenet121':
        return tf.keras.layers.Lambda(
            lambda x: tf.keras.applications.densenet.preprocess_input(x),
            name='densenet_preprocessing'
        )
    
    else:
        # Simple rescaling for custom CNN
        return tf.keras.layers.Rescaling(1.0 / 255.0, name='rescaling')


class ImageNormalization(tf.keras.layers.Layer):
    """
    Custom normalization layer that handles different normalization schemes.
    """
    
    def __init__(self, method: str = 'simple', **kwargs):
        """
        Initialize normalization layer.
        
        Args:
            method: Normalization method ('simple', 'imagenet', 'centered')
        """
        super().__init__(**kwargs)
        self.method = method
    
    def call(self, inputs):
        if self.method == 'simple':
            # [0, 1] normalization
            return inputs / 255.0
        
        elif self.method == 'imagenet':
            # ImageNet mean subtraction
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
            return inputs - mean
        
        elif self.method == 'centered':
            # [-1, 1] normalization
            return (inputs / 127.5) - 1.0
        
        else:
            return inputs / 255.0
    
    def get_config(self):
        config = super().get_config()
        config.update({'method': self.method})
        return config
