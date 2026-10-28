"""
SoyNet Model Factory Module
===========================
Factory pattern for creating models with pretrained backbones and custom classification heads.
Supports multiple backbone architectures with transfer learning and fine-tuning capabilities.

Author: SoyNet Research Team
Version: 1.0.0
"""

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetB0,
    EfficientNetB3,
    MobileNetV2,
    DenseNet121
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

import config
from utils.logger import get_logger
from utils.helpers import print_model_summary


# Mapping of model names to their Keras application classes
BACKBONE_MAP = {
    'resnet50': ResNet50,
    'efficientnetb0': EfficientNetB0,
    'efficientnetb3': EfficientNetB3,
    'mobilenetv2': MobileNetV2,
    'densenet121': DenseNet121
}

# Last convolutional layer names for Grad-CAM (per backbone)
GRADCAM_LAYER_MAP = {
    'resnet50': 'conv5_block3_out',
    'efficientnetb0': 'top_conv',
    'efficientnetb3': 'top_conv',
    'mobilenetv2': 'out_relu',
    'densenet121': 'relu',
    'custom_cnn': 'conv_block_4'  # Will be defined in custom CNN
}


def get_backbone(
    model_name: str = None,
    input_shape: Tuple[int, int, int] = None,
    include_top: bool = False,
    weights: str = 'imagenet',
    freeze: bool = True
) -> Model:
    """
    Load a pretrained backbone model.
    
    Args:
        model_name: Name of the backbone (resnet50, efficientnetb0, etc.)
        input_shape: Input shape (height, width, channels)
        include_top: Whether to include the classification head
        weights: Weight initialization ('imagenet' or None)
        freeze: Whether to freeze backbone layers
    
    Returns:
        Backbone model
    """
    logger = get_logger("SoyNet.Models")
    
    model_name = (model_name or config.MODEL_NAME).lower()
    input_shape = input_shape or config.INPUT_SHAPE
    
    if model_name == 'custom_cnn':
        from models.custom_cnn import create_custom_cnn
        backbone = create_custom_cnn(input_shape, include_classification_head=False)
        logger.info("Created custom CNN backbone")
        return backbone
    
    if model_name not in BACKBONE_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BACKBONE_MAP.keys())}")
    
    # Get backbone class
    BackboneClass = BACKBONE_MAP[model_name]
    
    # Determine weights
    if config.USE_PRETRAINED_WEIGHTS:
        weights = 'imagenet'
    else:
        weights = None
    
    # Create backbone
    backbone = BackboneClass(
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=None  # We'll add our own pooling
    )
    
    # Freeze layers if specified
    if freeze and config.FREEZE_BACKBONE:
        for layer in backbone.layers:
            layer.trainable = False
        logger.info(f"Loaded {model_name} backbone with frozen layers")
    else:
        logger.info(f"Loaded {model_name} backbone with trainable layers")
    
    return backbone


def add_classification_head(
    backbone: Model,
    num_classes: int,
    dense_units: int = None,
    dropout_rate: float = None
) -> Model:
    """
    Add a classification head to a backbone model.
    
    Args:
        backbone: Pretrained backbone model
        num_classes: Number of output classes
        dense_units: Number of units in dense layer
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Complete model with classification head
    """
    dense_units = dense_units or config.DENSE_UNITS
    dropout_rate = dropout_rate or config.DROPOUT_RATE
    
    # Get backbone output
    x = backbone.output
    
    # Add classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.BatchNormalization(name='head_bn')(x)
    x = layers.Dense(dense_units, activation='relu', name='head_dense')(x)
    x = layers.Dropout(dropout_rate, name='head_dropout')(x)
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        output = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    else:
        # Multi-class classification
        output = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs=backbone.input, outputs=output, name=f'soynet_{config.MODEL_NAME}')
    
    return model


def compile_model(
    model: Model,
    learning_rate: float = None,
    optimizer_name: str = None,
    loss: str = None,
    num_classes: int = 2
) -> Model:
    """
    Compile a model with optimizer and loss function.
    
    Args:
        model: Keras model
        learning_rate: Learning rate
        optimizer_name: Optimizer name (adam, sgd, rmsprop)
        loss: Loss function
        num_classes: Number of classes (for determining loss)
    
    Returns:
        Compiled model
    """
    learning_rate = learning_rate or config.INITIAL_LEARNING_RATE
    optimizer_name = (optimizer_name or config.OPTIMIZER).lower()
    
    # Determine loss function
    if loss is None:
        if num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
    
    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Determine metrics
    if num_classes == 2:
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        metrics = ['accuracy']
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def create_model(
    model_name: str = None,
    num_classes: int = 2,
    input_shape: Tuple[int, int, int] = None,
    freeze_backbone: bool = None,
    compile_model_flag: bool = True,
    learning_rate: float = None
) -> Model:
    """
    Create a complete model with backbone and classification head.
    This is the main entry point for model creation.
    
    Args:
        model_name: Name of the backbone model
        num_classes: Number of output classes
        input_shape: Input shape
        freeze_backbone: Whether to freeze backbone layers
        compile_model_flag: Whether to compile the model
        learning_rate: Learning rate for optimizer
    
    Returns:
        Complete model (compiled if specified)
    """
    logger = get_logger("SoyNet.Models")
    
    model_name = (model_name or config.MODEL_NAME).lower()
    input_shape = input_shape or config.INPUT_SHAPE
    freeze_backbone = freeze_backbone if freeze_backbone is not None else config.FREEZE_BACKBONE
    
    logger.info(f"Creating model: {model_name}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Handle custom CNN separately
    if model_name == 'custom_cnn':
        from models.custom_cnn import create_custom_cnn
        model = create_custom_cnn(input_shape, num_classes=num_classes)
    else:
        # Get backbone
        backbone = get_backbone(
            model_name=model_name,
            input_shape=input_shape,
            freeze=freeze_backbone
        )
        
        # Add classification head
        model = add_classification_head(backbone, num_classes)
    
    # Compile if requested
    if compile_model_flag:
        model = compile_model(model, learning_rate=learning_rate, num_classes=num_classes)
        logger.info("Model compiled successfully")
    
    # Print model summary
    print_model_summary(model, model_name)
    
    return model


def prepare_for_fine_tuning(
    model: Model,
    unfreeze_from_layer: int = None,
    learning_rate: float = None,
    num_classes: int = 2
) -> Model:
    """
    Prepare a trained model for fine-tuning by unfreezing backbone layers.
    
    Args:
        model: Trained model
        unfreeze_from_layer: Layer index to start unfreezing from
        learning_rate: Learning rate for fine-tuning (usually lower)
        num_classes: Number of classes
    
    Returns:
        Model prepared for fine-tuning
    """
    logger = get_logger("SoyNet.Models")
    
    # Use backbone-specific layer if not specified
    if unfreeze_from_layer is None:
        unfreeze_from = config.get_fine_tune_layer(config.MODEL_NAME)
    else:
        unfreeze_from = unfreeze_from_layer
    learning_rate = learning_rate or config.FINE_TUNE_LEARNING_RATE
    
    # Unfreeze layers from specified layer onwards
    for layer in model.layers[:unfreeze_from]:
        layer.trainable = False
    
    for layer in model.layers[unfreeze_from:]:
        # Don't unfreeze BatchNormalization layers (they can cause issues)
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    # Count trainable layers
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    total_count = len(model.layers)
    
    logger.info(f"Unfroze layers from index {unfreeze_from}")
    logger.info(f"Trainable layers: {trainable_count}/{total_count}")
    
    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=learning_rate, num_classes=num_classes)
    logger.info(f"Model recompiled with learning rate: {learning_rate}")
    
    return model


def get_gradcam_layer_name(model_name: str = None) -> str:
    """
    Get the name of the layer to use for Grad-CAM visualization.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Layer name for Grad-CAM
    """
    model_name = (model_name or config.MODEL_NAME).lower()
    
    if model_name in GRADCAM_LAYER_MAP:
        return GRADCAM_LAYER_MAP[model_name]
    else:
        raise ValueError(f"No Grad-CAM layer defined for model: {model_name}")


def load_trained_model(model_path: str = None) -> Model:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    logger = get_logger("SoyNet.Models")
    
    if model_path is None:
        model_path = config.get_model_save_path()
    
    logger.info(f"Loading model from: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    
    logger.info("Model loaded successfully")
    
    return model


def get_model_info(model: Model) -> dict:
    """
    Get information about a model.
    
    Args:
        model: Keras model
    
    Returns:
        Dictionary with model information
    """
    from utils.helpers import count_parameters
    
    trainable, non_trainable = count_parameters(model)
    
    return {
        'name': model.name,
        'total_layers': len(model.layers),
        'trainable_parameters': trainable,
        'non_trainable_parameters': non_trainable,
        'total_parameters': trainable + non_trainable,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
