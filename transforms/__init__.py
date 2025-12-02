"""
Custom transforms for melanoma image preprocessing.

This module provides reusable transform pipelines for training and validation.
"""

from torchvision import transforms
from typing import List, Tuple


# ImageNet normalization statistics (used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, augment: bool = True) -> transforms.Compose:
    """
    Get training transforms with optional data augmentation.
    
    Args:
        image_size: Target image size (default: 224)
        augment: Whether to apply data augmentation (default: True)
    
    Returns:
        Composed transforms for training
    """
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # No augmentation - same as validation transforms
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size (default: 224)
    
    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_test_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get test transforms (same as validation).
    
    Args:
        image_size: Target image size (default: 224)
    
    Returns:
        Composed transforms for testing
    """
    return get_val_transforms(image_size)


__all__ = [
    'get_train_transforms',
    'get_val_transforms', 
    'get_test_transforms',
    'IMAGENET_MEAN',
    'IMAGENET_STD'
]
