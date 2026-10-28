"""
SoyNet Dataset Module
======================
PyTorch DataLoader implementation using ImageFolder.
Handles train/val/test splitting with stratification.
"""

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np

from . import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get data transforms for training or evaluation."""
    if train:
        aug = config.AUGMENTATION_CONFIG
        return transforms.Compose([
            transforms.RandomResizedCrop(
                config.IMAGE_SIZE, 
                scale=aug["random_crop_scale"]
            ),
            transforms.RandomRotation(aug["rotation"]),
            transforms.RandomHorizontalFlip() if aug["horizontal_flip"] else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(**aug["color_jitter"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_inverse_transform():
    """Get inverse transform for visualization."""
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])


def stratified_split(dataset: datasets.ImageFolder, 
                     train_ratio: float, 
                     val_ratio: float, 
                     test_ratio: float,
                     seed: int) -> Tuple[list, list, list]:
    """
    Perform stratified split maintaining class distribution.
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    np.random.seed(seed)
    
    # Group indices by class
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for label, indices in class_indices.items():
        np.random.shuffle(indices)
        n = len(indices)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_indices.extend(indices[:train_end])
        val_indices.extend(indices[train_end:val_end])
        test_indices.extend(indices[val_end:])
    
    return train_indices, val_indices, test_indices


def compute_class_weights(dataset: datasets.ImageFolder, indices: list) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    class_counts = {}
    for idx in indices:
        _, label = dataset.samples[idx]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def get_weighted_sampler(dataset: datasets.ImageFolder, indices: list) -> WeightedRandomSampler:
    """Create weighted sampler for imbalanced classes."""
    class_weights = compute_class_weights(dataset, indices)
    
    sample_weights = []
    for idx in indices:
        _, label = dataset.samples[idx]
        sample_weights.append(class_weights[label].item())
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def get_dataloaders(batch_size: int = config.BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_info)
    """
    set_seed()
    
    # Load full dataset without transforms first (for splitting)
    full_dataset = datasets.ImageFolder(str(config.DATA_ROOT))
    
    # Get stratified split indices
    train_idx, val_idx, test_idx = stratified_split(
        full_dataset,
        config.TRAIN_SPLIT,
        config.VAL_SPLIT,
        config.TEST_SPLIT,
        config.RANDOM_SEED
    )
    
    # Create datasets with appropriate transforms
    train_transform = get_transforms(train=True)
    eval_transform = get_transforms(train=False)
    
    train_dataset = datasets.ImageFolder(str(config.DATA_ROOT), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(config.DATA_ROOT), transform=eval_transform)
    test_dataset = datasets.ImageFolder(str(config.DATA_ROOT), transform=eval_transform)
    
    # Create subsets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)
    
    # Compute class weights
    class_weights = compute_class_weights(full_dataset, train_idx)
    
    # Create weighted sampler for training
    train_sampler = get_weighted_sampler(full_dataset, train_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Dataset info
    dataset_info = {
        "num_classes": len(full_dataset.classes),
        "class_names": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "class_weights": class_weights.tolist(),
    }
    
    return train_loader, val_loader, test_loader, dataset_info
