"""
SoyNet Model Factory
=====================
PyTorch model creation with pretrained backbones.
Supports: ResNet50, EfficientNet-B0, EfficientNet-B3, MobileNetV2
"""

import torch
import torch.nn as nn
import timm

from . import config


class SoyNetClassifier(nn.Module):
    """Binary classifier wrapper for pretrained backbones."""
    
    def __init__(self, backbone: nn.Module, num_features: int, dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_features, 1)  # Binary classification
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits.squeeze(-1)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, unfreeze_from: int = 0):
        """Unfreeze backbone parameters starting from a specific layer."""
        params = list(self.backbone.parameters())
        for i, param in enumerate(params):
            if i >= unfreeze_from:
                param.requires_grad = True


class ModelFactory:
    """Factory for creating pretrained models."""
    
    # Mapping of model names to timm model identifiers
    MODEL_MAP = {
        "resnet50": "resnet50",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b3": "efficientnet_b3",
        "mobilenetv2": "mobilenetv2_100",
    }
    
    # Last conv layer names for Grad-CAM
    GRADCAM_LAYERS = {
        "resnet50": "layer4",
        "efficientnet_b0": "conv_head",
        "efficientnet_b3": "conv_head",
        "mobilenetv2": "conv_head",
    }
    
    @classmethod
    def create_model(cls, model_name: str, pretrained: bool = True, 
                     freeze: bool = True) -> SoyNetClassifier:
        """
        Create a model with pretrained backbone.
        
        Args:
            model_name: Name of the model (resnet50, efficientnet_b0, etc.)
            pretrained: Whether to use pretrained weights
            freeze: Whether to freeze backbone initially
            
        Returns:
            SoyNetClassifier model
        """
        if model_name not in cls.MODEL_MAP:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.MODEL_MAP.keys())}")
        
        timm_name = cls.MODEL_MAP[model_name]
        
        # Create backbone with features only (no classifier head)
        backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='avg'
        )
        
        # Get number of features from backbone
        num_features = backbone.num_features
        
        # Create classifier
        model = SoyNetClassifier(
            backbone=backbone,
            num_features=num_features,
            dropout=config.DROPOUT_RATE
        )
        
        if freeze:
            model.freeze_backbone()
            
        return model
    
    @classmethod
    def get_gradcam_layer_name(cls, model_name: str) -> str:
        """Get the target layer name for Grad-CAM."""
        return cls.GRADCAM_LAYERS.get(model_name, "layer4")
    
    @classmethod
    def count_parameters(cls, model: nn.Module) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable
        }


def get_model_info(model: nn.Module, model_name: str) -> dict:
    """Get comprehensive model information."""
    param_info = ModelFactory.count_parameters(model)
    return {
        "model_name": model_name,
        "architecture": ModelFactory.MODEL_MAP.get(model_name, model_name),
        **param_info,
        "gradcam_layer": ModelFactory.get_gradcam_layer_name(model_name)
    }
