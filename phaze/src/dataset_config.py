"""
Dataset configuration for PHAZE framework.

This module provides dataset-specific configurations for model creation
and RISC Zero guest program generation.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str
    input_size: int
    output_size: int
    input_channels: int
    spatial_size: int
    normalization_mean: tuple
    normalization_std: tuple
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "dataset_name": self.name,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "input_channels": self.input_channels,
            "spatial_size": self.spatial_size,
            "normalization_mean": self.normalization_mean,
            "normalization_std": self.normalization_std,
            "description": self.description,
        }


# Predefined dataset configurations
DATASET_CONFIGS = {
    "mnist": DatasetConfig(
        name="mnist",
        input_size=784,           # 28*28 flattened
        output_size=10,           # 10 digit classes
        input_channels=1,         # Grayscale
        spatial_size=28,          # 28x28 images
        normalization_mean=(0.1307,),
        normalization_std=(0.3081,),
        description="MNIST handwritten digits dataset"
    ),
    
    "cifar10": DatasetConfig(
        name="cifar10",
        input_size=3072,          # 32*32*3 flattened
        output_size=10,           # 10 object classes
        input_channels=3,         # RGB
        spatial_size=32,          # 32x32 images
        normalization_mean=(0.4914, 0.4822, 0.4465),
        normalization_std=(0.2023, 0.1994, 0.2010),
        description="CIFAR-10 object recognition dataset"
    ),
    
    "cifar100": DatasetConfig(
        name="cifar100",
        input_size=3072,          # 32*32*3 flattened
        output_size=100,          # 100 object classes
        input_channels=3,         # RGB
        spatial_size=32,          # 32x32 images
        normalization_mean=(0.5071, 0.4865, 0.4409),
        normalization_std=(0.2009, 0.1984, 0.2023),
        description="CIFAR-100 fine-grained object recognition dataset"
    ),
    
    "fashion_mnist": DatasetConfig(
        name="fashion_mnist",
        input_size=784,           # 28*28 flattened
        output_size=10,           # 10 clothing classes
        input_channels=1,         # Grayscale
        spatial_size=28,          # 28x28 images
        normalization_mean=(0.2860,),
        normalization_std=(0.3530,),
        description="Fashion-MNIST clothing classification dataset"
    ),
    
    "imagenet": DatasetConfig(
        name="imagenet",
        input_size=150528,        # 224*224*3 flattened
        output_size=1000,         # 1000 object classes
        input_channels=3,         # RGB
        spatial_size=224,         # 224x224 images
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        description="ImageNet large-scale object recognition dataset"
    ),
    
    # Common smaller variants for testing
    "tiny_imagenet": DatasetConfig(
        name="tiny_imagenet",
        input_size=12288,         # 64*64*3 flattened
        output_size=200,          # 200 object classes
        input_channels=3,         # RGB
        spatial_size=64,          # 64x64 images
        normalization_mean=(0.485, 0.456, 0.406),
        normalization_std=(0.229, 0.224, 0.225),
        description="Tiny ImageNet dataset (64x64 subset)"
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        DatasetConfig object
        
    Raises:
        ValueError: If dataset is not supported
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. "
            f"Available datasets: {available}"
        )
    
    return DATASET_CONFIGS[dataset_name]


def list_supported_datasets() -> list[str]:
    """List all supported dataset names."""
    return list(DATASET_CONFIGS.keys())


def create_custom_dataset_config(
    name: str,
    input_size: int,
    output_size: int,
    input_channels: int = 1,
    spatial_size: int = None,
    normalization_mean: tuple = None,
    normalization_std: tuple = None,
    description: str = ""
) -> DatasetConfig:
    """Create a custom dataset configuration.
    
    Args:
        name: Dataset name
        input_size: Total input size (flattened)
        output_size: Number of output classes
        input_channels: Number of input channels
        spatial_size: Spatial dimension (assumed square)
        normalization_mean: Mean for normalization
        normalization_std: Std for normalization
        description: Optional description
        
    Returns:
        DatasetConfig object
    """
    # Infer spatial size if not provided
    if spatial_size is None:
        spatial_size = int((input_size / input_channels) ** 0.5)
    
    # Default normalization for single channel
    if normalization_mean is None:
        normalization_mean = (0.5,) if input_channels == 1 else (0.5, 0.5, 0.5)
    if normalization_std is None:
        normalization_std = (0.5,) if input_channels == 1 else (0.5, 0.5, 0.5)
    
    return DatasetConfig(
        name=name,
        input_size=input_size,
        output_size=output_size,
        input_channels=input_channels,
        spatial_size=spatial_size,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        description=description or f"Custom dataset: {name}"
    )


def validate_dataset_config(config: DatasetConfig) -> bool:
    """Validate that a dataset configuration is consistent.
    
    Args:
        config: DatasetConfig to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    expected_input_size = config.spatial_size ** 2 * config.input_channels
    if config.input_size != expected_input_size:
        raise ValueError(
            f"Input size mismatch: expected {expected_input_size} "
            f"({config.spatial_size}²×{config.input_channels}), "
            f"got {config.input_size}"
        )
    
    if config.output_size <= 0:
        raise ValueError(f"Output size must be positive, got {config.output_size}")
    
    if len(config.normalization_mean) != config.input_channels:
        raise ValueError(
            f"Normalization mean length ({len(config.normalization_mean)}) "
            f"must match input channels ({config.input_channels})"
        )
    
    if len(config.normalization_std) != config.input_channels:
        raise ValueError(
            f"Normalization std length ({len(config.normalization_std)}) "
            f"must match input channels ({config.input_channels})"
        )
    
    return True


# Examples of usage
if __name__ == "__main__":
    # Print all supported datasets
    print("Supported datasets:")
    for name in list_supported_datasets():
        config = get_dataset_config(name)
        print(f"  {name}: {config.input_size} inputs → {config.output_size} classes")
        print(f"    Shape: {config.input_channels}×{config.spatial_size}×{config.spatial_size}")
        print(f"    Description: {config.description}")
        print()
    
    # Create custom dataset
    custom = create_custom_dataset_config(
        name="my_custom_dataset",
        input_size=256,  # 16x16 grayscale
        output_size=5,   # 5 classes
        input_channels=1,
        description="My custom 16x16 grayscale classification dataset"
    )
    print(f"Custom dataset: {custom}")
    validate_dataset_config(custom)