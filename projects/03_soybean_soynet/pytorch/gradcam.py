"""
SoyNet Grad-CAM Module
=======================
Gradient-weighted Class Activation Mapping for model interpretability.
Supports all backbone architectures with proper hook cleanup.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from . import config
from .models import ModelFactory


class GradCAM:
    """
    Grad-CAM implementation with automatic hook cleanup.
    Disables AMP during computation for numerical stability.
    """
    
    def __init__(self, model: nn.Module, model_name: str):
        """
        Initialize Grad-CAM for a specific model.
        
        Args:
            model: The trained model
            model_name: Name of the model architecture
        """
        self.model = model
        self.model_name = model_name
        self.device = next(model.parameters()).device
        
        # Get target layer
        self.target_layer = self._get_target_layer()
        if self.target_layer is None:
            raise ValueError(f"Could not find target layer for {model_name}")
        
        # Hooks storage
        self.gradients = None
        self.activations = None
        self._hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _get_target_layer(self) -> Optional[nn.Module]:
        """Get the target layer for Grad-CAM based on model architecture."""
        backbone = self.model.backbone
        layer_name = ModelFactory.get_gradcam_layer_name(self.model_name)
        
        # Try to get layer by name
        if hasattr(backbone, layer_name):
            return getattr(backbone, layer_name)
        
        # For ResNet: layer4 is the last conv block
        if self.model_name == "resnet50":
            if hasattr(backbone, 'layer4'):
                return backbone.layer4
        
        # For EfficientNet/MobileNet: try conv_head or features[-1]
        if hasattr(backbone, 'conv_head'):
            return backbone.conv_head
        
        if hasattr(backbone, 'features'):
            return backbone.features[-1]
        
        # Last resort: find last Conv2d layer
        last_conv = None
        for module in backbone.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Register hooks and store handles for cleanup
        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
    
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for input image.
        
        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            
        Returns:
            Normalized heatmap as numpy array [H, W]
        """
        self.model.eval()
        
        # Disable AMP for numerical stability
        with torch.cuda.amp.autocast(enabled=False):
            input_tensor = input_tensor.float().to(self.device)
            input_tensor.requires_grad = True
            
            # Forward pass
            output = self.model(input_tensor)
            
            # For binary classification, use the output directly
            if output.dim() == 0:
                target = output
            else:
                target = output[0] if output.dim() == 1 else output[0, 0]
            
            # Backward pass
            self.model.zero_grad()
            target.backward()
            
            # Get gradients and activations
            gradients = self.gradients  # [batch, channels, h, w]
            activations = self.activations  # [batch, channels, h, w]
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [batch, channels, 1, 1]
            
            # Weighted combination
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [batch, 1, h, w]
            
            # ReLU and normalize
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
            
            # Normalize to [0, 1]
            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            return cam


def generate_gradcam_visualization(
    image_path: str,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.4,
    colormap: str = "jet"
) -> None:
    """
    Generate and save Grad-CAM overlay visualization.
    
    Args:
        image_path: Path to original image
        heatmap: Grad-CAM heatmap
        output_path: Path to save visualization
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
    """
    # Load original image
    original = Image.open(image_path).convert('RGB')
    original = original.resize(config.IMAGE_SIZE)
    original_np = np.array(original)
    
    # Resize heatmap to image size
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(config.IMAGE_SIZE))
    heatmap_resized = heatmap_resized.astype(np.float32) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Create overlay
    overlay = (1 - alpha) * original_np + alpha * heatmap_colored
    overlay = overlay.astype(np.uint8)
    
    # Create figure with original and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_resized, cmap=colormap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_gradcam_for_model(
    model: nn.Module,
    model_name: str,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    output_dir: Path,
    num_samples_per_class: int = 5
) -> Dict:
    """
    Generate Grad-CAM visualizations for a trained model.
    
    Args:
        model: Trained model
        model_name: Name of the model architecture
        test_loader: Test data loader
        class_names: List of class names (from ImageFolder.classes)
        output_dir: Base output directory
        num_samples_per_class: Number of samples per class
        
    Returns:
        Metadata dictionary
    """
    output_dir = Path(output_dir)
    gradcam_dir = output_dir / "gradcam"
    images_dir = gradcam_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, model_name)
    device = next(model.parameters()).device
    
    # Access the underlying dataset
    dataset = test_loader.dataset
    if hasattr(dataset, 'dataset'):
        base_dataset = dataset.dataset
        indices = dataset.indices
    else:
        base_dataset = dataset
        indices = list(range(len(dataset)))
    
    # Collect samples by class
    samples_by_class = {i: [] for i in range(len(class_names))}
    
    for idx in indices:
        file_path, label = base_dataset.samples[idx]
        if len(samples_by_class[label]) < num_samples_per_class:
            samples_by_class[label].append((idx, file_path))
    
    # Generate visualizations
    metadata = {
        "model_name": model_name,
        "num_samples_per_class": num_samples_per_class,
        "class_names": class_names,
        "samples": []
    }
    
    # Get transforms
    from .dataset import get_transforms
    transform = get_transforms(train=False)
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_idx, (dataset_idx, file_path) in enumerate(samples_by_class[class_idx]):
            # Load and transform image
            image = Image.open(file_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Generate heatmap
            heatmap = gradcam.generate(input_tensor)
            
            # Save visualization
            output_path = class_dir / f"sample_{sample_idx}.png"
            generate_gradcam_visualization(
                file_path,
                heatmap,
                output_path,
                alpha=config.GRADCAM_ALPHA,
                colormap=config.GRADCAM_COLORMAP
            )
            
            # Record metadata
            metadata["samples"].append({
                "class_name": class_name,
                "class_idx": class_idx,
                "sample_idx": sample_idx,
                "source_path": file_path,
                "output_path": str(output_path)
            })
    
    # Cleanup hooks
    gradcam.remove_hooks()
    
    # Save metadata
    metadata_path = gradcam_dir / "gradcam_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Grad-CAM visualizations saved to: {gradcam_dir}")
    print(f"Generated {len(metadata['samples'])} visualizations")
    
    return metadata
