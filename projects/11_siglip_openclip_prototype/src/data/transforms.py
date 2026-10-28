"""Backbone-compatible image transforms and moderate data augmentations."""

from typing import Any, Callable, Dict, Optional, Tuple, Union
from torchvision import transforms

# Standard ImageNet / CLIP normalization statistics
OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# SigLIP default normalization statistics
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def get_transforms(
    framework: str = "open_clip",
    image_size: int = 224,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    custom_preprocess: Optional[Callable] = None
) -> Tuple[Callable, Callable]:
    """
    Construct training and validation/test image transformations.
    
    Training transformations include moderate disease-preserving augmentations:
    - RandomResizedCrop (scale 0.8-1.0, preserving lesion context)
    - RandomHorizontalFlip (p=0.5)
    - Small RandomRotation (<= 15 deg)
    - Moderate ColorJitter (preserving pathology discoloration cues)
    
    Validation/Test transformations are strictly deterministic:
    - Resize(image_size) -> CenterCrop(image_size) -> ToTensor() -> Normalize
    """
    # Select normalization constants based on framework
    if mean is None or std is None:
        if framework.lower() == "siglip":
            mean = SIGLIP_MEAN
            std = SIGLIP_STD
        else:
            mean = OPENCLIP_MEAN
            std = OPENCLIP_STD

    # Training transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Validation and test transform
    if custom_preprocess is not None:
        val_transform = custom_preprocess
    else:
        val_transform = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return train_transform, val_transform
