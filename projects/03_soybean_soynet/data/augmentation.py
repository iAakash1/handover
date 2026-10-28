"""
SoyNet Data Augmentation Module
===============================
Configurable data augmentation pipeline using TensorFlow/Keras layers.
Only applied to training data.

Author: SoyNet Research Team
Version: 1.0.0
"""

import tensorflow as tf
from tensorflow.keras import layers

import config


def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Create a data augmentation layer based on configuration.
    
    Returns:
        Sequential model containing augmentation layers
    """
    aug_config = config.AUGMENTATION_CONFIG
    
    augmentation_layers = []
    
    # Random rotation
    if aug_config.get('rotation_range', 0) > 0:
        # Convert degrees to radians (keras uses factor of 2*pi)
        rotation_factor = aug_config['rotation_range'] / 360.0
        augmentation_layers.append(
            layers.RandomRotation(factor=rotation_factor, fill_mode='nearest')
        )
    
    # Random horizontal flip
    if aug_config.get('horizontal_flip', False):
        augmentation_layers.append(layers.RandomFlip(mode='horizontal'))
    
    # Random vertical flip
    if aug_config.get('vertical_flip', False):
        augmentation_layers.append(layers.RandomFlip(mode='vertical'))
    
    # Random zoom
    if aug_config.get('zoom_range', 0) > 0:
        zoom = aug_config['zoom_range']
        augmentation_layers.append(
            layers.RandomZoom(height_factor=(-zoom, zoom), width_factor=(-zoom, zoom))
        )
    
    # Random translation (width/height shift)
    width_shift = aug_config.get('width_shift_range', 0)
    height_shift = aug_config.get('height_shift_range', 0)
    if width_shift > 0 or height_shift > 0:
        augmentation_layers.append(
            layers.RandomTranslation(
                height_factor=height_shift,
                width_factor=width_shift,
                fill_mode='nearest'
            )
        )
    
    # Random brightness
    brightness_range = aug_config.get('brightness_range', None)
    if brightness_range is not None:
        # Convert range to factor
        lower, upper = brightness_range
        factor = (upper - lower) / 2
        augmentation_layers.append(
            layers.RandomBrightness(factor=factor)
        )
    
    # Random contrast (optional)
    contrast_range = aug_config.get('contrast_range', None)
    if contrast_range is not None:
        augmentation_layers.append(
            layers.RandomContrast(factor=contrast_range)
        )
    
    # Build sequential model
    augmentation = tf.keras.Sequential(
        augmentation_layers,
        name='data_augmentation'
    )
    
    return augmentation


def apply_augmentation(image: tf.Tensor, training: bool = True) -> tf.Tensor:
    """
    Apply augmentation to a single image.
    
    Args:
        image: Input image tensor
        training: Whether in training mode (augmentation only applied if True)
    
    Returns:
        Augmented image tensor
    """
    if not training or not config.ENABLE_AUGMENTATION:
        return image
    
    augmentation = get_augmentation_layer()
    return augmentation(image, training=True)


def create_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Create a comprehensive augmentation pipeline for plant disease detection.
    Includes augmentations specifically useful for leaf images.
    
    Returns:
        Sequential model with augmentation layers
    """
    return tf.keras.Sequential([
        # Spatial augmentations
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),  # +/- 10% of 2*pi = +/- 36 degrees
        layers.RandomZoom(0.1),  # +/- 10% zoom
        layers.RandomTranslation(0.1, 0.1),  # +/- 10% translation
        
        # Color augmentations (important for disease detection)
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ], name='augmentation_pipeline')


def visualize_augmentations(
    image_path: str,
    num_augmentations: int = 9,
    save_path: str = None
):
    """
    Visualize augmentation effects on a sample image.
    Useful for debugging and tuning augmentation parameters.
    
    Args:
        image_path: Path to input image
        num_augmentations: Number of augmented versions to generate
        save_path: Path to save visualization (displays if None)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, config.IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    
    # Get augmentation layer
    augmentation = get_augmentation_layer()
    
    # Generate augmented versions
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Data Augmentation Visualization', fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        if i == 0:
            # Show original
            ax.imshow(image.numpy())
            ax.set_title('Original')
        else:
            # Show augmented
            augmented = augmentation(tf.expand_dims(image, 0), training=True)
            ax.imshow(augmented[0].numpy())
            ax.set_title(f'Augmented {i}')
        
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Augmentation visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


class AugmentationLayer(tf.keras.layers.Layer):
    """
    Custom augmentation layer that can be included in the model.
    Augmentation is only applied during training.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.augmentation = get_augmentation_layer()
    
    def call(self, inputs, training=None):
        if training:
            return self.augmentation(inputs, training=True)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        return config
