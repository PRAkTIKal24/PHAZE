"""
Example: Using PHAZE with different datasets.

This example demonstrates how to use the PHAZE framework with datasets
other than MNIST, including CIFAR-10, CIFAR-100, and custom datasets.
"""

import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

async def example_cifar10():
    """Example using CIFAR-10 dataset."""
    print("🖼️  CIFAR-10 Example")
    print("=" * 40)
    
    from phaze.src.dataset_config import get_dataset_config
    from phaze.src.model_architectures import PHAZEModelFactory, ModelComplexity
    from phaze.src.enhanced_zkml_benchmark import RiscZeroBackendWrapper
    import torch
    
    # Get CIFAR-10 configuration
    cifar10_config = get_dataset_config("cifar10")
    print(f"Dataset: {cifar10_config.description}")
    print(f"Input shape: {cifar10_config.input_channels}×{cifar10_config.spatial_size}×{cifar10_config.spatial_size}")
    print(f"Classes: {cifar10_config.output_size}")
    
    # Create a convolutional model suitable for CIFAR-10
    model = PHAZEModelFactory.create_full_model(
        architecture="conv",
        complexity=ModelComplexity.LIGHT,
        input_size=cifar10_config.input_size,
        output_size=cifar10_config.output_size,
        input_channels=cifar10_config.input_channels,
        spatial_size=cifar10_config.spatial_size
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create model info including dataset configuration
    model_info = model.get_model_info()
    model_info.update(cifar10_config.to_dict())
    
    # Create sample input (batch of CIFAR-10 images)
    sample_input = torch.randn(1, cifar10_config.input_channels, 
                               cifar10_config.spatial_size, cifar10_config.spatial_size)
    
    # For RISC Zero, we need flattened input
    sample_input_flat = sample_input.view(1, -1)
    
    print(f"Sample input shape: {sample_input.shape} → {sample_input_flat.shape}")
    
    # Test with RISC Zero backend
    try:
        wrapper = RiscZeroBackendWrapper(model, model_info)
        await wrapper.setup(sample_input_flat)
        print("✅ RISC Zero setup successful")
        
        # Note: This would require guest programs to be built for CIFAR-10
        # Run: uv run phaze-risc-build build --dataset cifar10
        
    except RuntimeError as e:
        print(f"⚠️  RISC Zero setup failed: {e}")
        print("💡 Run: uv run phaze-risc-build build --dataset cifar10")

async def example_custom_dataset():
    """Example using a custom dataset configuration."""
    print("\n🎯 Custom Dataset Example")
    print("=" * 40)
    
    from phaze.src.dataset_config import create_custom_dataset_config, validate_dataset_config
    from phaze.src.model_architectures import PHAZEModelFactory, ModelComplexity
    import torch
    
    # Create custom dataset configuration (16x16 RGB, 7 classes)
    custom_config = create_custom_dataset_config(
        name="my_dataset",
        input_size=768,  # 16*16*3
        output_size=7,   # 7 classes
        input_channels=3,
        spatial_size=16,
        description="Custom 16x16 RGB classification dataset"
    )
    
    # Validate the configuration
    validate_dataset_config(custom_config)
    print(f"✅ Custom dataset configuration valid")
    print(f"Dataset: {custom_config.description}")
    print(f"Input: {custom_config.input_size} ({custom_config.input_channels}×{custom_config.spatial_size}×{custom_config.spatial_size})")
    print(f"Output: {custom_config.output_size} classes")
    
    # Create a model for this dataset
    model = PHAZEModelFactory.create_full_model(
        architecture="simple",
        complexity=ModelComplexity.MINIMAL,
        input_size=custom_config.input_size,
        output_size=custom_config.output_size
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    sample_input = torch.randn(1, custom_config.input_size)
    output = model(sample_input)
    print(f"Forward pass: {sample_input.shape} → {output.shape}")

def example_dataset_comparison():
    """Compare different dataset configurations."""
    print("\n📊 Dataset Comparison")
    print("=" * 40)
    
    from phaze.src.dataset_config import list_supported_datasets, get_dataset_config
    
    datasets = ["mnist", "cifar10", "cifar100", "fashion_mnist"]
    
    print(f"{'Dataset':<15} {'Input Size':<10} {'Classes':<8} {'Channels':<8} {'Spatial':<8}")
    print("-" * 60)
    
    for name in datasets:
        config = get_dataset_config(name)
        print(f"{name:<15} {config.input_size:<10} {config.output_size:<8} "
              f"{config.input_channels:<8} {config.spatial_size}×{config.spatial_size}")

def show_build_commands():
    """Show commands for building RISC Zero support for different datasets."""
    print("\n🔨 Build Commands for Different Datasets")
    print("=" * 40)
    
    datasets = ["mnist", "cifar10", "cifar100", "fashion_mnist", "imagenet"]
    
    print("To build RISC Zero guest programs for specific datasets:")
    print()
    
    for dataset in datasets:
        print(f"# {dataset.upper()}")
        print(f"uv run phaze-risc-build build --dataset {dataset}")
        print(f"uv run phaze-risc-build scan --dataset {dataset}")
        print()
    
    print("To list all supported datasets:")
    print("uv run phaze-risc-build datasets")
    print()
    
    print("To use a custom dataset:")
    print("1. Create DatasetConfig in your code")
    print("2. Pass config to register_all_factory_models()")
    print("3. Build guest programs")

async def main():
    """Run all examples."""
    print("🚀 PHAZE Multi-Dataset Examples")
    print("=" * 50)
    
    # Show dataset comparison
    example_dataset_comparison()
    
    # Show build commands
    show_build_commands()
    
    # Run CIFAR-10 example
    await example_cifar10()
    
    # Run custom dataset example
    await example_custom_dataset()
    
    print("\n✅ All examples completed!")
    print("\n💡 Next steps:")
    print("1. Choose your dataset")
    print("2. Build guest programs: uv run phaze-risc-build build --dataset <name>")
    print("3. Update your training pipeline to use the dataset configuration")
    print("4. Run PHAZE with your dataset!")

if __name__ == "__main__":
    asyncio.run(main())