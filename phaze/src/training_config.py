"""
Training configuration system for PHAZE framework.

This module provides unified configuration for both training and plotting,
allowing users to control all parameters from a single location.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

from .plotting.plot_config import PlotConfig, PlotStyle


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""

    # Dataset source
    source: str = "mnist"  # Currently "mnist", future: custom directory path
    dataset_size: int = 1000  # Small subset for fast training

    # Data loading
    batch_size: int = 64
    num_workers: int = 4
    shuffle: bool = True

    # Preprocessing
    normalize: bool = True
    augment: bool = False  # Keep simple for reproducibility


@dataclass
class ModelConfig:
    """Configuration for model architectures and complexities."""

    # Available model types
    architectures: List[str] = field(
        default_factory=lambda: ["simple", "conv", "transformer", "multi_exit"]
    )
    complexities: List[str] = field(
        default_factory=lambda: ["minimal", "light", "medium", "heavy"]
    )

    # Multi-exit configuration
    early_exit_ratios: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.25, 0.50, 0.75]
    )

    # Model parameters
    input_size: int = 28 * 28  # MNIST flattened
    output_size: int = 10  # MNIST classes
    input_channels: int = 1  # MNIST grayscale
    spatial_size: int = 28  # MNIST image size


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Training hyperparameters
    epochs: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"

    # Device configuration
    device: str = "auto"  # auto, cpu, cuda
    use_gpu_if_available: bool = True

    # Training behavior
    early_stopping: bool = True
    patience: int = 3
    save_best_only: bool = True

    # Progress tracking
    verbose: bool = True
    log_interval: int = 10  # Log every N batches


@dataclass
class ExperimentConfig:
    """Configuration for experimental runs and reproducibility."""

    # Reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])
    deterministic: bool = True

    # Benchmarking
    zkml_frameworks: List[str] = field(default_factory=lambda: ["ezkl", "risc_zero"])
    hashing_algorithms: List[str] = field(default_factory=lambda: ["rabin", "shamir"])
    polynomial_degree: int = 100
    field_size: int = 2**31 - 1

    # Performance settings
    benchmark_iterations: int = 10  # Reduced for faster results
    warmup_iterations: int = 2

    # Memory profiling
    profile_memory: bool = True
    memory_check_interval: float = 0.1  # seconds


@dataclass
class OutputConfig:
    """Configuration for outputs and artifacts."""

    # Output directories
    output_dir: str = "benchmark_results"
    models_dir: str = "trained_models"
    plots_dir: str = "plots"
    logs_dir: str = "logs"

    # Model saving
    save_models: bool = True
    export_onnx: bool = True
    save_checkpoints: bool = False  # Keep simple for now

    # Plotting
    generate_plots: bool = True
    save_plots: bool = True
    plot_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])

    # Data export
    export_benchmark_data: bool = True
    export_format: str = "json"  # json, csv, pickle


@dataclass
class PHAZEConfig:
    """Unified configuration for PHAZE training and plotting."""

    # Sub-configurations
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)

    # Global settings
    project_name: str = "phaze_benchmark"
    version: str = "0.1.0"

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "PHAZEConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_python(cls, config_path: Union[str, Path]) -> "PHAZEConfig":
        """Load configuration from Python file."""
        import importlib.util

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load the Python module
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config module from {config_path}")

        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Try to get configuration from the module
        if hasattr(config_module, "get_default_config"):
            return config_module.get_default_config()
        elif hasattr(config_module, "DEFAULT_CONFIG"):
            return config_module.DEFAULT_CONFIG
        else:
            raise AttributeError(
                "Config module must have either 'get_default_config()' function "
                "or 'DEFAULT_CONFIG' attribute"
            )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "PHAZEConfig":
        """Load configuration from file (auto-detect format)."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Auto-detect file format
        if config_path.suffix.lower() in [".yml", ".yaml"]:
            return cls.from_yaml(config_path)
        elif config_path.suffix.lower() == ".py":
            return cls.from_python(config_path)
        else:
            # Try Python first, then YAML
            try:
                return cls.from_python(config_path)
            except Exception:
                try:
                    return cls.from_yaml(config_path)
                except Exception as e:
                    raise ValueError(
                        f"Could not load config from {config_path}. "
                        f"Supported formats: .py, .yml, .yaml"
                    ) from e

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PHAZEConfig":
        """Create configuration from dictionary."""
        # Extract sub-configurations
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        experiment_config = ExperimentConfig(**config_dict.get("experiment", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))

        # Handle plotting config
        plotting_dict = config_dict.get("plotting", {})
        if "style" in plotting_dict and isinstance(plotting_dict["style"], str):
            plotting_dict["style"] = PlotStyle(plotting_dict["style"])
        plotting_config = PlotConfig(**plotting_dict)

        # Create main config
        main_config = {
            k: v
            for k, v in config_dict.items()
            if k
            not in ["dataset", "model", "training", "experiment", "output", "plotting"]
        }

        return cls(
            dataset=dataset_config,
            model=model_config,
            training=training_config,
            experiment=experiment_config,
            output=output_config,
            plotting=plotting_config,
            **main_config,
        )

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        result = {
            "project_name": self.project_name,
            "version": self.version,
            "dataset": self._dataclass_to_dict(self.dataset),
            "model": self._dataclass_to_dict(self.model),
            "training": self._dataclass_to_dict(self.training),
            "experiment": self._dataclass_to_dict(self.experiment),
            "output": self._dataclass_to_dict(self.output),
            "plotting": self._dataclass_to_dict(self.plotting),
        }

        # Handle PlotStyle enum
        if "style" in result["plotting"]:
            if hasattr(result["plotting"]["style"], "value"):
                result["plotting"]["style"] = result["plotting"]["style"].value
            # If it's already a string, keep it as is

        return result

    def _dataclass_to_dict(self, obj) -> dict:
        """Convert dataclass to dictionary, handling enums."""
        result = {}
        for key, value in obj.__dict__.items():
            if hasattr(value, "value"):  # Enum
                result[key] = value.value
            else:
                result[key] = value
        return result

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []

        # Validate dataset
        if self.dataset.dataset_size <= 0:
            warnings.append("Dataset size must be positive")

        if self.dataset.batch_size <= 0:
            warnings.append("Batch size must be positive")

        # Validate training
        if self.training.epochs <= 0:
            warnings.append("Number of epochs must be positive")

        if self.training.learning_rate <= 0:
            warnings.append("Learning rate must be positive")

        # Validate model architectures
        valid_architectures = ["simple", "conv", "transformer", "multi_exit"]
        for arch in self.model.architectures:
            if arch not in valid_architectures:
                warnings.append(f"Unknown architecture: {arch}")

        valid_complexities = ["minimal", "light", "medium", "heavy"]
        for comp in self.model.complexities:
            if comp not in valid_complexities:
                warnings.append(f"Unknown complexity: {comp}")

        # Validate early exit ratios
        for ratio in self.model.early_exit_ratios:
            if not 0 < ratio < 1:
                warnings.append(f"Early exit ratio must be between 0 and 1: {ratio}")

        # Validate seeds
        if len(self.experiment.seeds) == 0:
            warnings.append("At least one seed must be specified")

        # Validate frameworks
        valid_frameworks = ["ezkl", "risc_zero"]
        for framework in self.experiment.zkml_frameworks:
            if framework not in valid_frameworks:
                warnings.append(f"Unknown zkML framework: {framework}")

        return warnings

    def create_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        base_dir = Path(self.output.output_dir)

        directories = [
            base_dir,
            base_dir / self.output.models_dir,
            base_dir / self.output.plots_dir,
            base_dir / self.output.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def create_default_config() -> PHAZEConfig:
    """Create a default configuration."""
    return PHAZEConfig()


def load_config(config_path: Optional[Union[str, Path]] = None) -> PHAZEConfig:
    """Load configuration from file or create default."""
    if config_path is None:
        return create_default_config()

    return PHAZEConfig.from_file(config_path)


def save_default_config(config_path: Union[str, Path]) -> None:
    """Save default configuration to file."""
    config = create_default_config()
    config.to_yaml(config_path)


# Example usage and testing
if __name__ == "__main__":
    # Create and validate default config
    config = create_default_config()
    warnings = config.validate()

    print("PHAZE Configuration Validation:")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✅ Configuration is valid")

    # Save example config
    config.to_yaml("example_config.yaml")
    print("📄 Example configuration saved to example_config.yaml")

    # Test loading
    loaded_config = PHAZEConfig.from_yaml("example_config.yaml")
    print("✅ Configuration loading test successful")
