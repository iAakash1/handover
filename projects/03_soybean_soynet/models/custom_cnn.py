"""
SoyNet Custom CNN Module
========================
Custom baseline CNN architecture for comparison against pretrained models.
Designed for plant disease classification at 256x256 resolution.

Author: SoyNet Research Team
Version: 1.0.0
"""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    Activation
)

import config


def conv_block(
    x,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = 'same',
    activation: str = 'relu',
    batch_norm: bool = True,
    name: str = None
):
    """
    Convolutional block with optional batch normalization.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size
        strides: Stride size
        padding: Padding mode
        activation: Activation function
        batch_norm: Whether to use batch normalization
        name: Block name prefix
    
    Returns:
        Output tensor
    """
    conv_name = f'{name}_conv' if name else None
    bn_name = f'{name}_bn' if name else None
    act_name = f'{name}_act' if name else None
    
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=not batch_norm,
        name=conv_name
    )(x)
    
    if batch_norm:
        x = BatchNormalization(name=bn_name)(x)
    
    x = Activation(activation, name=act_name)(x)
    
    return x


def create_custom_cnn(
    input_shape: Tuple[int, int, int] = None,
    num_classes: int = 2,
    include_classification_head: bool = True,
    dropout_rate: float = None
) -> Model:
    """
    Create a custom CNN architecture for plant disease classification.
    
    Architecture:
    - 5 convolutional blocks with progressive filter increase
    - Batch normalization after each convolution
    - MaxPooling for spatial reduction
    - Dropout for regularization
    - Global average pooling before classification head
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        include_classification_head: Whether to include classification layers
        dropout_rate: Dropout rate
    
    Returns:
        Custom CNN model
    """
    input_shape = input_shape or config.INPUT_SHAPE
    dropout_rate = dropout_rate or config.DROPOUT_RATE
    
    # Input layer
    inputs = Input(shape=input_shape, name='input')
    
    # Normalize input to [0, 1]
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name='rescaling')(inputs)
    
    # Block 1: 32 filters
    x = conv_block(x, filters=32, kernel_size=3, name='conv_block_1')
    x = conv_block(x, filters=32, kernel_size=3, name='conv_block_1b')
    x = MaxPooling2D(pool_size=2, name='pool_1')(x)
    x = Dropout(0.25, name='dropout_1')(x)
    
    # Block 2: 64 filters
    x = conv_block(x, filters=64, kernel_size=3, name='conv_block_2')
    x = conv_block(x, filters=64, kernel_size=3, name='conv_block_2b')
    x = MaxPooling2D(pool_size=2, name='pool_2')(x)
    x = Dropout(0.25, name='dropout_2')(x)
    
    # Block 3: 128 filters
    x = conv_block(x, filters=128, kernel_size=3, name='conv_block_3')
    x = conv_block(x, filters=128, kernel_size=3, name='conv_block_3b')
    x = MaxPooling2D(pool_size=2, name='pool_3')(x)
    x = Dropout(0.25, name='dropout_3')(x)
    
    # Block 4: 256 filters (last conv block - used for Grad-CAM)
    x = conv_block(x, filters=256, kernel_size=3, name='conv_block_4')
    x = conv_block(x, filters=256, kernel_size=3, name='conv_block_4b')
    x = MaxPooling2D(pool_size=2, name='pool_4')(x)
    x = Dropout(0.25, name='dropout_4')(x)
    
    # Block 5: 512 filters
    x = conv_block(x, filters=512, kernel_size=3, name='conv_block_5')
    x = conv_block(x, filters=512, kernel_size=3, name='conv_block_5b')
    
    # Classification head
    if include_classification_head:
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = BatchNormalization(name='head_bn')(x)
        x = Dense(256, activation='relu', name='head_dense')(x)
        x = Dropout(dropout_rate, name='head_dropout')(x)
        
        # Output layer
        if num_classes == 2:
            outputs = Dense(1, activation='sigmoid', name='predictions')(x)
        else:
            outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    else:
        outputs = x
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='soynet_custom_cnn')
    
    return model


def create_lightweight_cnn(
    input_shape: Tuple[int, int, int] = None,
    num_classes: int = 2
) -> Model:
    """
    Create a lightweight CNN for faster training/inference.
    Suitable for resource-constrained environments.
    
    Args:
        input_shape: Input shape
        num_classes: Number of classes
    
    Returns:
        Lightweight CNN model
    """
    input_shape = input_shape or config.INPUT_SHAPE
    
    inputs = Input(shape=input_shape, name='input')
    
    # Normalize
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    
    # Simple architecture
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='soynet_lightweight_cnn')
    
    return model


def create_deep_cnn(
    input_shape: Tuple[int, int, int] = None,
    num_classes: int = 2
) -> Model:
    """
    Create a deeper CNN with residual-like connections.
    For more complex feature extraction.
    
    Args:
        input_shape: Input shape
        num_classes: Number of classes
    
    Returns:
        Deep CNN model
    """
    input_shape = input_shape or config.INPUT_SHAPE
    
    inputs = Input(shape=input_shape, name='input')
    
    # Normalize
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    
    # Initial conv
    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual-like blocks
    for filters in [64, 128, 256, 512]:
        # Save for skip connection
        skip = x
        
        # Conv path
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Skip connection (with projection if needed)
        if skip.shape[-1] != filters:
            skip = Conv2D(filters, 1)(skip)
            skip = BatchNormalization()(skip)
        
        x = tf.keras.layers.Add()([x, skip])
        x = Activation('relu')(x)
        x = MaxPooling2D(2)(x)
    
    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(x)
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='soynet_deep_cnn')
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing Custom CNN creation...")
    
    model = create_custom_cnn()
    model.summary()
    
    print("\n\nTesting Lightweight CNN creation...")
    lightweight_model = create_lightweight_cnn()
    lightweight_model.summary()
