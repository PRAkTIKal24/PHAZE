"""
MNIST training module for the PHAZE framework.

This module provides fast, minimal training capabilities for generating
benchmark data. Focus is on speed and reproducibility rather than accuracy.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from .dataset_config import get_dataset_config
from .model_architectures import ModelComplexity, PHAZEModelFactory
from .training_config import PHAZEConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MNISTTrainer:
    """Fast MNIST trainer for PHAZE benchmark models."""

    def __init__(self, config: PHAZEConfig):
        """Initialize trainer with configuration.

        Args:
            config: PHAZE configuration object
        """
        self.config = config
        self.device = self._get_device()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Dataset Parameters
        self.source = config.dataset.source if config.dataset.source else "mnist"
        dataset_cfg = get_dataset_config(self.source)
        self.input_size = dataset_cfg.input_size
        self.output_size = dataset_cfg.output_size
        self.input_channels = dataset_cfg.input_channels
        self.spatial_size = dataset_cfg.spatial_size

        # Training state
        self.current_model = None
        self.current_optimizer = None
        self.current_criterion = None
        self.training_history = {}

        # Setup data loaders
        self._setup_data_loaders()

    def _get_device(self) -> torch.device:
        """Determine the device to use for training."""
        if self.config.training.device == "auto":
            if self.config.training.use_gpu_if_available and torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            device = torch.device(self.config.training.device)
            logger.info(f"Using specified device: {device}")

        return device

    def _setup_data_loaders(self) -> None:
        """Setup MNIST data loaders with small subsets for fast training."""
        # Data transformations
        if self.config.dataset.normalize:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
                ]
            )
        else:
            transform = transforms.ToTensor()

        # Load full MNIST dataset
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        # Create small subsets for fast training
        train_subset = self._create_balanced_subset(
            train_dataset, self.config.dataset.dataset_size
        )

        val_subset = self._create_balanced_subset(
            test_dataset,
            min(self.config.dataset.dataset_size // 2, 500),  # Smaller validation set
        )

        test_subset = self._create_balanced_subset(
            test_dataset,
            min(self.config.dataset.dataset_size // 4, 250),  # Even smaller test set
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_subset,
            batch_size=self.config.dataset.batch_size,
            shuffle=self.config.dataset.shuffle,
            num_workers=self.config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        self.test_loader = DataLoader(
            test_subset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info("Data loaders created:")
        logger.info(f"  Train: {len(train_subset)} samples")
        logger.info(f"  Validation: {len(val_subset)} samples")
        logger.info(f"  Test: {len(test_subset)} samples")

    def _create_balanced_subset(self, dataset, target_size: int) -> Subset:
        """Create a balanced subset of the dataset with equal samples per class."""
        samples_per_class = target_size // 10  # 10 MNIST classes

        # Group indices by class
        class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        # Select balanced samples
        selected_indices = []
        for class_label in range(10):
            class_samples = class_indices[class_label][:samples_per_class]
            selected_indices.extend(class_samples)

        return Subset(dataset, selected_indices)

    def create_model(self, architecture: str, complexity: str) -> nn.Module:
        """Create a model with specified architecture and complexity.

        Args:
            architecture: Model architecture type
            complexity: Model complexity level

        Returns:
            PyTorch model instance
        """
        # Map string complexity to enum
        complexity_map = {
            "minimal": ModelComplexity.MINIMAL,
            "light": ModelComplexity.LIGHT,
            "medium": ModelComplexity.MEDIUM,
            "heavy": ModelComplexity.HEAVY,
            "extreme": ModelComplexity.EXTREME,
        }

        complexity_enum = complexity_map.get(complexity, ModelComplexity.MEDIUM)

        # Handle different input formats for different architectures
        if architecture == "conv":
            model = PHAZEModelFactory.create_full_model(
                architecture=architecture,
                complexity=complexity_enum,
                input_channels=self.config.model.input_channels,
                spatial_size=self.config.model.spatial_size,
                output_size=self.config.model.output_size,
            )
        else:
            model = PHAZEModelFactory.create_full_model(
                architecture=architecture,
                complexity=complexity_enum,
                input_size=self.config.model.input_size,
                output_size=self.config.model.output_size,
            )

        # Move model to device
        model = model.to(self.device)

        logger.info(f"Created {architecture} model with {complexity} complexity")
        logger.info(
            f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        return model

    def _setup_training(self, model: nn.Module) -> Tuple[optim.Optimizer, nn.Module]:
        """Setup optimizer and loss function for training."""
        # Setup optimizer
        if self.config.training.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

        # Setup loss function
        criterion = nn.CrossEntropyLoss()

        return optimizer, criterion

    def _preprocess_input(self, x: torch.Tensor, architecture: str) -> torch.Tensor:
        """Preprocess input based on architecture requirements."""
        if architecture == "conv":
            # Keep as image format (B, C, H, W)
            return x
        else:
            # Flatten for non-convolutional architectures
            return x.view(x.size(0), -1)

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        architecture: str,
    ) -> Dict[str, float]:
        """Train model for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
            disable=not self.config.training.verbose,
        )

        for _batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            data = self._preprocess_input(data, architecture)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Update progress bar
            if self.config.training.verbose:
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.2f}%",
                    }
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def validate(
        self, model: nn.Module, criterion: nn.Module, architecture: str
    ) -> Dict[str, float]:
        """Validate model on validation set."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = self._preprocess_input(data, architecture)

                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy}

    def train_model(
        self, model: nn.Module, architecture: str, complexity: str, seed: int
    ) -> Dict[str, Any]:
        """Train a model and return training results.

        Args:
            model: Model to train
            architecture: Architecture type
            complexity: Complexity level
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing training results and metadata
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Setup training
        optimizer, criterion = self._setup_training(model)

        # Training state
        best_val_loss = float("inf")
        patience_counter = 0
        training_start_time = time.time()

        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": [],
        }

        logger.info(f"Starting training: {architecture} {complexity} (seed {seed})")

        # Training loop
        for epoch in range(self.config.training.epochs):
            epoch_start_time = time.time()

            # Train one epoch
            train_metrics = self.train_epoch(model, optimizer, criterion, architecture)

            # Validate
            val_metrics = self.validate(model, criterion, architecture)

            epoch_time = time.time() - epoch_start_time

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["epoch_times"].append(epoch_time)

            # Logging
            if self.config.training.verbose:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.training.epochs}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                    f"Time: {epoch_time:.2f}s"
                )

            # Early stopping
            if self.config.training.early_stopping:
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.training.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

        total_training_time = time.time() - training_start_time

        # Final test evaluation
        test_metrics = self.validate(model, criterion, architecture)

        # Training results
        results = {
            "model_id": f"{architecture}_{complexity}_seed{seed}",
            "architecture": architecture,
            "complexity": complexity,
            "seed": seed,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "input_channels": self.input_channels,
            "spatial_size": self.spatial_size,
            "training_time": total_training_time,
            "epochs_completed": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_test_loss": test_metrics["loss"],
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "final_test_acc": test_metrics["accuracy"],
            "best_val_loss": best_val_loss,
            "history": history,
            "device": str(self.device),
        }

        logger.info(f"Training completed: {results['model_id']}")
        logger.info(f"  Final test accuracy: {test_metrics['accuracy']:.2f}%")
        logger.info(f"  Training time: {total_training_time:.2f}s")

        return results

    def save_model(
        self, model: nn.Module, model_info: Dict[str, Any], output_dir: Path
    ) -> Dict[str, str]:
        """Save trained model and metadata.

        Args:
            model: Trained model
            model_info: Model information and training results
            output_dir: Output directory

        Returns:
            Dictionary with saved file paths
        """
        model_id = model_info["model_id"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save model state dict
        if self.config.output.save_models:
            model_path = output_dir / f"{model_id}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_info": model_info,
                    "config": self.config.to_dict(),
                },
                model_path,
            )
            saved_files["model"] = str(model_path)

        # Save metadata
        metadata_path = output_dir / f"{model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(model_info, f, indent=2, default=str)
        saved_files["metadata"] = str(metadata_path)

        return saved_files

    def export_to_onnx(
        self, model: nn.Module, model_info: Dict[str, Any], output_dir: Path
    ) -> Optional[str]:
        """Export model to ONNX format for ezkl compatibility.

        Args:
            model: Trained model
            model_info: Model information
            output_dir: Output directory

        Returns:
            Path to exported ONNX file or None if export fails
        """
        if not self.config.output.export_onnx:
            return None

        model_id = model_info["model_id"]
        architecture = model_info["architecture"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = output_dir / f"{model_id}.onnx"

        try:
            # Create dummy input
            if architecture == "conv":
                dummy_input = torch.randn(
                    1,
                    self.config.model.input_channels,
                    self.config.model.spatial_size,
                    self.config.model.spatial_size,
                ).to(self.device)
            else:
                dummy_input = torch.randn(1, self.config.model.input_size).to(
                    self.device
                )

            # Export to ONNX
            export_start_time = time.time()

            model.eval()
            # Export to ONNX using legacy behavior for EZKL compatibility
            model.eval()
            try:
                import torch._dynamo
                with torch._dynamo.config.patch(suppress_errors=True), \
                     torch.no_grad():
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                        verbose=False,
                        keep_initializers_as_inputs=False,
                        training=torch.onnx.TrainingMode.EVAL
                    )
            except ImportError:
                # Fallback if dynamo not available
                with torch.no_grad():
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                    )

            export_time = time.time() - export_start_time

            logger.info(f"ONNX export completed: {onnx_path}")
            logger.info(f"  Export time: {export_time:.2f}s")

            # Update model info with export time
            model_info["onnx_export_time"] = export_time
            model_info["onnx_path"] = str(onnx_path)

            return str(onnx_path)

        except Exception as e:
            logger.error(f"ONNX export failed for {model_id}: {e}")
            return None

    def get_sample_input(self, architecture: str) -> torch.Tensor:
        """Get a sample input tensor for the given architecture.

        Args:
            architecture: Model architecture type

        Returns:
            Sample input tensor
        """
        if architecture == "conv":
            return torch.randn(
                1,
                self.config.model.input_channels,
                self.config.model.spatial_size,
                self.config.model.spatial_size,
            ).to(self.device)
        else:
            return torch.randn(1, self.config.model.input_size).to(self.device)


# Convenience functions
def train_single_model(
    config: PHAZEConfig, architecture: str, complexity: str, seed: int
) -> Dict[str, Any]:
    """Train a single model with given parameters.

    Args:
        config: PHAZE configuration
        architecture: Model architecture
        complexity: Model complexity
        seed: Random seed

    Returns:
        Training results dictionary
    """
    trainer = MNISTTrainer(config)
    model = trainer.create_model(architecture, complexity)
    results = trainer.train_model(model, architecture, complexity, seed)

    # Save model if configured
    if config.output.save_models or config.output.export_onnx:
        output_dir = Path(config.output.output_dir) / config.output.models_dir
        saved_files = trainer.save_model(model, results, output_dir)
        results["saved_files"] = saved_files

        # Export to ONNX
        onnx_path = trainer.export_to_onnx(model, results, output_dir)
        if onnx_path:
            results["onnx_path"] = onnx_path

    return results


if __name__ == "__main__":
    # Example usage
    from .training_config import create_default_config

    config = create_default_config()
    config.training.verbose = True
    config.training.epochs = 1  # Quick test

    # Test training
    results = train_single_model(config, "simple", "minimal", 42)
    print("Training completed!")
    print(f"Final accuracy: {results['final_test_acc']:.2f}%")
    print(f"Training time: {results['training_time']:.2f}s")
