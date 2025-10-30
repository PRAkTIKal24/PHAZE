#!/usr/bin/env python3
"""
PHAZE Default Configuration (Python format)

This configuration file defines all parameters for training, benchmarking, and plotting.
Being in Python format allows for dynamic configuration, better IDE support, and easier maintenance.
"""

from phaze.src.plotting.plot_config import PlotConfig, PlotStyle
from phaze.src.training_config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    PHAZEConfig,
    TrainingConfig,
)

# Project metadata
PROJECT_NAME = "phaze_benchmark"
VERSION = "0.14.4"

# Dataset configuration
DATASET_CONFIG = DatasetConfig(
    source="mnist",  # Currently "mnist", future: custom directory path
    dataset_size=256,  # Small subset for fast training
    batch_size=64,  # Small batches for speed
    num_workers=4,  # Data loading workers
    shuffle=True,  # Shuffle training data
    normalize=True,  # Apply normalization
    augment=False,  # Keep simple for reproducibility
)

# Model configuration
MODEL_CONFIG = ModelConfig(
    architectures=["multi_exit"],
    complexities=["minimal", "light", "medium", "heavy", "extreme"],
    early_exit_ratios=[0.10, 0.30, 0.60, 0.85, 0.97],  # Parameter ratios for M_early
    input_size=784,  # MNIST flattened (28*28)
    output_size=10,  # MNIST classes
    input_channels=1,  # MNIST grayscale
    spatial_size=28,  # MNIST image size
)

# Training configuration
TRAINING_CONFIG = TrainingConfig(
    epochs=2,  # Minimal training
    learning_rate=0.001,  # Standard learning rate
    weight_decay=1e-5,  # L2 regularization
    optimizer="adam",  # adam, sgd
    device="auto",  # auto, cpu, cuda
    use_gpu_if_available=True,  # Fallback to CPU if no GPU
    early_stopping=True,  # Enable early stopping
    patience=3,  # Early stopping patience
    save_best_only=True,  # Save only best model
    verbose=True,  # Enable progress logging
    log_interval=10,  # Log every N batches
)

# Experiment configuration
EXPERIMENT_CONFIG = ExperimentConfig(
    seeds=[42, 123, 55, 61, 48],  # Random seeds for reproducibility
    deterministic=True,  # Enable deterministic training
    zkml_frameworks=["ezkl"],  # zkML frameworks to benchmark
    hashing_algorithms=["rabin", "shamir"],  # Crypto algorithms to benchmark
    polynomial_degree=2048,  # Degree for rabin fingerprints
    field_size=2**31 - 1,  # Field size for rabin fingerprints
    benchmark_iterations=5,  # Iterations per benchmark
    warmup_iterations=2,  # Warmup iterations
    profile_memory=True,  # Enable memory profiling
    memory_check_interval=0.1,  # Memory check interval (seconds)
)

# Output configuration
OUTPUT_CONFIG = OutputConfig(
    output_dir="benchmark_results",  # Main output directory
    models_dir="trained_models",  # Model artifacts subdirectory
    plots_dir="plots",  # Plots subdirectory
    logs_dir="logs",  # Logs subdirectory
    save_models=True,  # Save trained models
    export_onnx=True,  # Export models to ONNX
    save_checkpoints=False,  # Save training checkpoints
    generate_plots=True,  # Generate benchmark plots
    save_plots=True,  # Save plots to disk
    plot_formats=["png"],  # Plot export formats
    export_benchmark_data=True,  # Export benchmark data
    export_format="json",  # json, csv, pickle
)

# Plotting configuration
PLOTTING_CONFIG = PlotConfig(
    style=PlotStyle.NEURIPS,  # Plot style: neurips, publication, presentation, web
    dpi=300,  # Plot resolution
    figure_size=(12, 8),  # Default figure size (width, height)
    font_size=12,  # Base font size
    title_size=14,  # Title font size
    legend_size=10,  # Legend font size
    export_formats=["png"],  # Export formats
    primary_colors=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ],  # Primary color cycle
)


# Main configuration object
def get_default_config() -> PHAZEConfig:
    """Get the default PHAZE configuration."""
    return PHAZEConfig(
        project_name=PROJECT_NAME,
        version=VERSION,
        dataset=DATASET_CONFIG,
        model=MODEL_CONFIG,
        training=TRAINING_CONFIG,
        experiment=EXPERIMENT_CONFIG,
        output=OUTPUT_CONFIG,
        plotting=PLOTTING_CONFIG,
    )


# Quick test configurations
def get_quick_config() -> PHAZEConfig:
    """Get a quick test configuration for rapid iteration."""
    config = get_default_config()

    # Reduce for quick testing
    config.training.epochs = 1
    config.dataset.dataset_size = 500
    config.experiment.benchmark_iterations = 3
    config.model.architectures = ["simple", "conv"]  # Test subset
    config.model.complexities = ["minimal", "light"]  # Test subset
    config.experiment.seeds = [42, 123]  # Fewer seeds
    config.dataset.num_workers = 0  # Avoid worker warnings

    return config


def get_full_config() -> PHAZEConfig:
    """Get the full configuration for complete benchmarking."""
    return get_default_config()


def get_minimal_config() -> PHAZEConfig:
    """Get minimal configuration for basic testing."""
    config = get_default_config()

    # Minimal settings
    config.training.epochs = 1
    config.dataset.dataset_size = 100
    config.experiment.benchmark_iterations = 2
    config.model.architectures = ["simple"]
    config.model.complexities = ["minimal"]
    config.experiment.seeds = [42]
    config.dataset.num_workers = 0
    config.output.generate_plots = False  # Skip plotting for speed

    return config


# Custom configurations for specific use cases
def get_plotting_focused_config() -> PHAZEConfig:
    """Get configuration focused on generating good plotting data."""
    config = get_default_config()

    # Optimize for diverse plotting data
    config.training.epochs = 2
    config.dataset.dataset_size = 1000
    config.experiment.benchmark_iterations = 10
    config.model.architectures = ["simple", "conv", "transformer"]
    config.model.complexities = ["minimal", "light", "medium", "heavy"]
    config.experiment.seeds = [42, 123, 456]  # Multiple seeds for error bars

    return config


def get_performance_config() -> PHAZEConfig:
    """Get configuration for performance-focused benchmarking."""
    config = get_default_config()

    # Focus on performance metrics
    config.training.epochs = 3
    config.dataset.dataset_size = 2000
    config.experiment.benchmark_iterations = 20
    config.experiment.profile_memory = True
    config.output.save_checkpoints = True

    return config


# Environment-specific configurations
def get_ci_config() -> PHAZEConfig:
    """Get configuration suitable for CI/CD environments."""
    config = get_minimal_config()

    # CI-friendly settings
    config.training.device = "cpu"  # Force CPU for CI
    config.training.use_gpu_if_available = False
    config.training.verbose = False  # Reduce log noise
    config.output.save_models = False  # Don't save artifacts in CI
    config.output.export_onnx = False

    return config


def get_development_config() -> PHAZEConfig:
    """Get configuration for development and debugging."""
    config = get_quick_config()

    # Development-friendly settings
    config.training.verbose = True
    config.output.save_models = True
    config.output.export_onnx = True
    config.output.generate_plots = True

    return config


# Configuration factory function
def get_config(config_type: str = "default") -> PHAZEConfig:
    """Get configuration by type.

    Args:
        config_type: Type of configuration to get
            - "default": Standard configuration
            - "quick": Quick test configuration
            - "full": Full benchmarking configuration
            - "minimal": Minimal test configuration
            - "plotting": Plotting-focused configuration
            - "performance": Performance-focused configuration
            - "ci": CI/CD-friendly configuration
            - "dev": Development configuration

    Returns:
        PHAZEConfig instance
    """
    config_map = {
        "default": get_default_config,
        "quick": get_quick_config,
        "full": get_full_config,
        "minimal": get_minimal_config,
        "plotting": get_plotting_focused_config,
        "performance": get_performance_config,
        "ci": get_ci_config,
        "dev": get_development_config,
        "development": get_development_config,
    }

    if config_type not in config_map:
        available = ", ".join(config_map.keys())
        raise ValueError(f"Unknown config type '{config_type}'. Available: {available}")

    return config_map[config_type]()


# Validation function
def validate_config(config: PHAZEConfig) -> bool:
    """Validate a configuration and print any warnings.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration is valid
    """
    warnings = config.validate()
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        return False
    return True


# Export for backward compatibility
DEFAULT_CONFIG = get_default_config()

if __name__ == "__main__":
    # Example usage and testing
    print("PHAZE Configuration (Python format)")
    print("=" * 50)

    # Test default config
    config = get_default_config()
    is_valid = validate_config(config)
    print(f"Default config valid: {is_valid}")
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Architectures: {config.model.architectures}")
    print(f"Complexities: {config.model.complexities}")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Dataset size: {config.dataset.dataset_size}")

    print("\nAvailable configuration types:")
    config_types = [
        "default",
        "quick",
        "full",
        "minimal",
        "plotting",
        "performance",
        "ci",
        "dev",
    ]
    for config_type in config_types:
        try:
            test_config = get_config(config_type)
            print(
                f"  ✅ {config_type}: {test_config.training.epochs} epochs, "
                f"{test_config.dataset.dataset_size} samples"
            )
        except Exception as e:
            print(f"  ❌ {config_type}: {e}")

    print("\n✅ Python configuration system working correctly!")
