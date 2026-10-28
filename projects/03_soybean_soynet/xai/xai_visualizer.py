"""
SoyNet XAI Visualizer Module
============================
Visualization utilities for Grad-CAM heatmaps and explainability outputs.
Generates heatmap overlays, side-by-side comparisons, and batch visualizations.

Author: SoyNet Research Team
Version: 1.0.0
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image

import config
from utils.logger import get_logger
from utils.helpers import get_timestamp
from xai.gradcam import GradCAM, make_gradcam_heatmap
from data.preprocessing import load_image


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = None,
    colormap: str = None
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (RGB, [0, 1] or [0, 255])
        heatmap: Grad-CAM heatmap (single channel, [0, 1])
        alpha: Overlay transparency (0 = only image, 1 = only heatmap)
        colormap: Matplotlib colormap name
    
    Returns:
        Image with heatmap overlay
    """
    alpha = alpha or config.GRADCAM_ALPHA
    colormap = colormap or config.GRADCAM_COLORMAP
    
    # Ensure image is in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Resize heatmap to image size
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Get RGB channels
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed


def visualize_gradcam(
    model,
    image_path: str,
    class_names: List[str] = None,
    save_path: str = None,
    layer_name: str = None,
    figsize: tuple = (15, 5)
) -> Tuple[np.ndarray, int, float]:
    """
    Generate and visualize Grad-CAM for a single image.
    Shows original image, heatmap, and overlay side by side.
    
    Args:
        model: Trained Keras model
        image_path: Path to input image
        class_names: List of class names for labeling
        save_path: Path to save visualization
        layer_name: Layer name for Grad-CAM
        figsize: Figure size
    
    Returns:
        Tuple of (overlayed_image, predicted_class, confidence)
    """
    logger = get_logger("SoyNet.XAI")
    
    # Load and preprocess image
    image = load_image(image_path)
    
    # Also load original image for display
    original_image = np.array(Image.open(image_path).resize(config.IMAGE_SIZE[::-1]))
    if original_image.max() > 1:
        display_image = original_image / 255.0
    else:
        display_image = original_image
    
    # Generate Grad-CAM heatmap
    logger.info(f"Generating Grad-CAM for: {image_path}")
    heatmap, pred_class, confidence = make_gradcam_heatmap(
        model, image, layer_name=layer_name
    )
    
    # Create overlay
    overlayed = overlay_heatmap(display_image, heatmap)
    
    # Resize heatmap for display
    heatmap_resized = cv2.resize(heatmap, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
    
    # Get class name
    class_label = class_names[pred_class] if class_names else f"Class {pred_class}"
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap_resized, cmap=config.GRADCAM_COLORMAP)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Prediction: {class_label}\nConfidence: {confidence:.2%}', 
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add overall title
    fig.suptitle('Grad-CAM Visualization - SoyNet Disease Detection', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grad-CAM visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return overlayed, pred_class, confidence


def visualize_batch_gradcam(
    model,
    image_paths: List[str],
    class_names: List[str] = None,
    save_dir: str = None,
    layer_name: str = None,
    num_cols: int = 3
):
    """
    Generate Grad-CAM visualizations for multiple images.
    
    Args:
        model: Trained Keras model
        image_paths: List of image paths
        class_names: List of class names
        save_dir: Directory to save visualizations
        layer_name: Layer name for Grad-CAM
        num_cols: Number of columns in the grid
    """
    logger = get_logger("SoyNet.XAI")
    
    save_dir = Path(save_dir or config.XAI_OUTPUTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create GradCAM instance
    gradcam = GradCAM(model, layer_name=layer_name)
    
    num_images = len(image_paths)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for i, image_path in enumerate(image_paths):
        # Load image
        image = load_image(image_path)
        original = np.array(Image.open(image_path).resize(config.IMAGE_SIZE[::-1]))
        if original.max() > 1:
            original = original / 255.0
        
        # Generate heatmap
        heatmap, pred_class, confidence = gradcam.compute_heatmap(image)
        
        # Create overlay
        overlayed = overlay_heatmap(original, heatmap)
        
        # Get class name
        class_label = class_names[pred_class] if class_names else f"Class {pred_class}"
        
        # Plot
        axes[i].imshow(overlayed)
        axes[i].set_title(f'{class_label}\n({confidence:.1%})', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Grad-CAM Batch Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"gradcam_batch_{get_timestamp()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Batch Grad-CAM visualization saved to: {save_path}")


def generate_xai_outputs(
    model,
    dataset_info,
    test_dataset=None,
    num_samples: int = None,
    save_dir: Path = None
):
    """
    Generate comprehensive XAI outputs for the trained model.
    
    Args:
        model: Trained Keras model
        dataset_info: DatasetInfo object
        test_dataset: Test dataset (optional, for sample selection)
        num_samples: Number of samples to visualize
        save_dir: Directory to save outputs
    """
    logger = get_logger("SoyNet.XAI")
    
    if not config.ENABLE_XAI:
        logger.info("XAI disabled in config. Skipping XAI output generation.")
        return
    
    num_samples = num_samples or config.XAI_NUM_SAMPLES
    save_dir = Path(save_dir or config.XAI_OUTPUTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating XAI outputs for {num_samples} samples...")
    
    # Get sample images from dataset
    from data.dataset_loader import get_test_images
    
    image_paths, true_labels = get_test_images(
        config.DATA_ROOT, 
        num_samples=num_samples
    )
    
    # Create GradCAM instance
    gradcam = GradCAM(model)
    
    # Generate individual visualizations
    for i, (image_path, true_label) in enumerate(zip(image_paths, true_labels)):
        class_name = dataset_info.class_names[true_label]
        
        save_path = save_dir / f"gradcam_{class_name}_{i+1}_{get_timestamp()}.png"
        
        try:
            visualize_gradcam(
                model,
                image_path,
                class_names=dataset_info.class_names,
                save_path=str(save_path)
            )
        except Exception as e:
            logger.error(f"Error generating Grad-CAM for {image_path}: {e}")
    
    # Generate batch visualization
    try:
        visualize_batch_gradcam(
            model,
            image_paths,
            class_names=dataset_info.class_names,
            save_dir=save_dir
        )
    except Exception as e:
        logger.error(f"Error generating batch Grad-CAM: {e}")
    
    logger.info(f"XAI outputs saved to: {save_dir}")


def create_attention_comparison(
    model,
    image_paths: List[str],
    class_names: List[str],
    save_path: str = None
):
    """
    Create a comparison showing model attention across different images.
    Useful for understanding how the model focuses on disease symptoms.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        class_names: Class names
        save_path: Path to save comparison
    """
    logger = get_logger("SoyNet.XAI")
    
    gradcam = GradCAM(model)
    
    n_images = len(image_paths)
    fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))
    
    for i, image_path in enumerate(image_paths):
        # Load image
        image = load_image(image_path)
        original = np.array(Image.open(image_path).resize(config.IMAGE_SIZE[::-1]))
        if original.max() > 1:
            original = original / 255.0
        
        # Generate heatmap
        heatmap, pred_class, confidence = gradcam.compute_heatmap(image)
        overlayed = overlay_heatmap(original, heatmap)
        
        # Get class name
        class_label = class_names[pred_class] if class_names else f"Class {pred_class}"
        
        # Plot original
        axes[0, i].imshow(original)
        axes[0, i].set_title(f'Sample {i + 1}', fontsize=11)
        axes[0, i].axis('off')
        
        # Plot overlay
        axes[1, i].imshow(overlayed)
        axes[1, i].set_title(f'{class_label}\n({confidence:.1%})', fontsize=10)
        axes[1, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=12, fontweight='bold')
    
    plt.suptitle('Model Attention Analysis - SoyNet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Attention comparison saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def analyze_misclassifications(
    model,
    test_dataset,
    dataset_info,
    num_samples: int = 5,
    save_dir: Path = None
):
    """
    Analyze and visualize misclassified samples using Grad-CAM.
    Helps understand why the model failed on certain samples.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        dataset_info: DatasetInfo object
        num_samples: Number of misclassified samples to analyze
        save_dir: Directory to save analysis
    """
    logger = get_logger("SoyNet.XAI")
    
    save_dir = Path(save_dir or config.XAI_OUTPUTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Analyzing misclassified samples...")
    
    # This would require access to individual image paths
    # For now, we'll work with the dataset directly
    misclassified = []
    
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        pred_classes = (preds > 0.5).astype(int).flatten()
        
        for i, (pred, true) in enumerate(zip(pred_classes, labels.numpy())):
            if pred != true:
                misclassified.append({
                    'image': images[i].numpy(),
                    'true_label': true,
                    'pred_label': pred,
                    'confidence': float(preds[i])
                })
        
        if len(misclassified) >= num_samples:
            break
    
    if not misclassified:
        logger.info("No misclassifications found!")
        return
    
    # Visualize misclassifications
    gradcam = GradCAM(model)
    
    n_samples = min(len(misclassified), num_samples)
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, sample in enumerate(misclassified[:n_samples]):
        image = sample['image']
        true_label = dataset_info.class_names[sample['true_label']]
        pred_label = dataset_info.class_names[sample['pred_label']]
        confidence = sample['confidence']
        
        # Display image (denormalize if needed)
        if image.max() <= 1:
            display_image = image
        else:
            display_image = image / 255.0
        
        # Generate heatmap
        heatmap, _, _ = gradcam.compute_heatmap(image)
        overlayed = overlay_heatmap(display_image, heatmap)
        
        # Plot original
        axes[0, i].imshow(display_image)
        axes[0, i].set_title(f'True: {true_label}', fontsize=10, color='green')
        axes[0, i].axis('off')
        
        # Plot overlay
        axes[1, i].imshow(overlayed)
        axes[1, i].set_title(f'Pred: {pred_label}\n({confidence:.1%})', fontsize=10, color='red')
        axes[1, i].axis('off')
    
    plt.suptitle('Misclassification Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f"misclassification_analysis_{get_timestamp()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Misclassification analysis saved to: {save_path}")
