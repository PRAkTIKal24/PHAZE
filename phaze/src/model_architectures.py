"""
Modular model architectures for the PHAZE framework.

This module provides various model architectures that can be used as M_early and M_full
in the PHAZE system, with support for different complexity levels and early-exit strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class ModelComplexity(Enum):
    """Enumeration of model complexity levels."""

    MINIMAL = "minimal"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    EXTREME = "extreme"


class PHAZEModelInterface(ABC, nn.Module):
    """Abstract base class for PHAZE models."""

    def __init__(self, input_size: int, output_size: int, complexity: ModelComplexity):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.complexity = complexity
        self.confidence_threshold = 0.8

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get confidence scores for the predictions."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        pass

    def should_exit_early(self, x: torch.Tensor) -> bool:
        """Determine if we should exit early based on confidence."""
        confidence = self.get_confidence(x)
        return torch.max(confidence) > self.confidence_threshold

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops_estimate(self, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for the model (rough approximation)."""
        # This is a very rough estimate - in practice you'd use tools like thop or fvcore
        total_params = self.count_parameters()
        # Rough estimate: 2 FLOPs per parameter per forward pass
        return total_params * 2


class SimpleEarlyExitModel(PHAZEModelInterface):
    """Simple early-exit model for basic testing."""

    def __init__(self, input_size: int = 10, output_size: int = 5):
        super().__init__(input_size, output_size, ModelComplexity.MINIMAL)

        self.fc1 = nn.Linear(input_size, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, output_size)
        self.confidence_head = nn.Linear(20, 1)  # For confidence estimation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        confidence = torch.sigmoid(self.confidence_head(x))
        return confidence

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": "SimpleEarlyExit",
            "complexity": self.complexity.value,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "parameters": self.count_parameters(),
            "layers": ["Linear(10->20)", "ReLU", "Linear(20->5)"],
        }


class ConvolutionalEarlyExitModel(PHAZEModelInterface):
    """Convolutional early-exit model for image-like data."""

    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 32,
        output_size: int = 10,
        complexity: ModelComplexity = ModelComplexity.LIGHT,
    ):
        super().__init__(
            input_channels * input_size * input_size, output_size, complexity
        )

        self.input_channels = input_channels
        self.input_spatial_size = input_size

        # Define architecture based on complexity
        if complexity == ModelComplexity.MINIMAL:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.feature_size = 16
        elif complexity == ModelComplexity.LIGHT:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.feature_size = 64
        else:  # MEDIUM and above
            self.conv_layers = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.feature_size = 256

        self.classifier = nn.Linear(self.feature_size, output_size)
        self.confidence_head = nn.Linear(self.feature_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if input is flattened
        if len(x.shape) == 2:
            x = x.view(
                -1,
                self.input_channels,
                self.input_spatial_size,
                self.input_spatial_size,
            )

        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.view(
                -1,
                self.input_channels,
                self.input_spatial_size,
                self.input_spatial_size,
            )

        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        confidence = torch.sigmoid(self.confidence_head(features))
        return confidence

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": "ConvolutionalEarlyExit",
            "complexity": self.complexity.value,
            "input_channels": self.input_channels,
            "input_spatial_size": self.input_spatial_size,
            "output_size": self.output_size,
            "parameters": self.count_parameters(),
            "feature_size": self.feature_size,
        }


class TransformerEarlyExitModel(PHAZEModelInterface):
    """Transformer-based early-exit model."""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 10,
        complexity: ModelComplexity = ModelComplexity.MEDIUM,
    ):
        super().__init__(input_size, output_size, complexity)

        # Define architecture based on complexity
        if complexity == ModelComplexity.MINIMAL:
            self.d_model = 64
            self.nhead = 2
            self.num_layers = 1
        elif complexity == ModelComplexity.LIGHT:
            self.d_model = 128
            self.nhead = 4
            self.num_layers = 2
        elif complexity == ModelComplexity.MEDIUM:
            self.d_model = 256
            self.nhead = 8
            self.num_layers = 4
        else:  # HEAVY and above
            self.d_model = 512
            self.nhead = 8
            self.num_layers = 6

        self.input_projection = nn.Linear(input_size, self.d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.classifier = nn.Linear(self.d_model, output_size)
        self.confidence_head = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input shapes
        if len(x.shape) == 2:
            # Treat as sequence of length 1
            x = x.unsqueeze(1)  # (batch, 1, features)

        seq_len = x.size(1)
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)

        return self.classifier(x)

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)

        confidence = torch.sigmoid(self.confidence_head(x))
        return confidence

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": "TransformerEarlyExit",
            "complexity": self.complexity.value,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "parameters": self.count_parameters(),
        }


class MultiExitModel(PHAZEModelInterface):
    """Model with multiple exit points for different confidence levels."""

    def __init__(
        self,
        input_size: int = 10,
        output_size: int = 5,
        complexity: ModelComplexity = ModelComplexity.MEDIUM,
    ):
        super().__init__(input_size, output_size, complexity)

        # Define layer sizes based on complexity
        if complexity == ModelComplexity.MINIMAL:
            self.layer_sizes = [input_size, 32, 64]
        elif complexity == ModelComplexity.LIGHT:
            self.layer_sizes = [input_size, 64, 128, 256]
        elif complexity == ModelComplexity.MEDIUM:
            self.layer_sizes = [input_size, 128, 256, 512, 256]
        else:  # HEAVY and above
            self.layer_sizes = [input_size, 256, 512, 1024, 512, 256]

        # Build the main backbone
        self.backbone_layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.backbone_layers.append(
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            )
            self.backbone_layers.append(nn.ReLU())

        # Exit heads at different layers
        self.exit_heads = nn.ModuleList()
        self.confidence_heads = nn.ModuleList()

        for i in range(1, len(self.layer_sizes)):
            self.exit_heads.append(nn.Linear(self.layer_sizes[i], output_size))
            self.confidence_heads.append(nn.Linear(self.layer_sizes[i], 1))

        self.num_exits = len(self.exit_heads)

    def forward(
        self, x: torch.Tensor, exit_layer: Optional[int] = None
    ) -> torch.Tensor:
        """Forward pass with optional early exit."""
        if exit_layer is None:
            exit_layer = self.num_exits - 1  # Use final exit by default

        # Forward through backbone layers
        for i in range(exit_layer + 1):
            layer_idx = i * 2  # Account for ReLU layers
            if layer_idx < len(self.backbone_layers):
                x = self.backbone_layers[layer_idx](x)  # Linear layer
                if layer_idx + 1 < len(self.backbone_layers):
                    x = self.backbone_layers[layer_idx + 1](x)  # ReLU layer

        # Apply exit head
        return self.exit_heads[exit_layer](x)

    def get_confidence(
        self, x: torch.Tensor, exit_layer: Optional[int] = None
    ) -> torch.Tensor:
        """Get confidence for a specific exit layer."""
        if exit_layer is None:
            exit_layer = self.num_exits - 1

        # Forward through backbone layers
        for i in range(exit_layer + 1):
            layer_idx = i * 2
            if layer_idx < len(self.backbone_layers):
                x = self.backbone_layers[layer_idx](x)
                if layer_idx + 1 < len(self.backbone_layers):
                    x = self.backbone_layers[layer_idx + 1](x)

        confidence = torch.sigmoid(self.confidence_heads[exit_layer](x))
        return confidence

    def get_all_exits(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get outputs and confidences for all exit points."""
        results = []
        current_x = x

        for i in range(self.num_exits):
            # Forward through one more layer
            layer_idx = i * 2
            if layer_idx < len(self.backbone_layers):
                current_x = self.backbone_layers[layer_idx](current_x)
                if layer_idx + 1 < len(self.backbone_layers):
                    current_x = self.backbone_layers[layer_idx + 1](current_x)

            # Get output and confidence for this exit
            output = self.exit_heads[i](current_x)
            confidence = torch.sigmoid(self.confidence_heads[i](current_x))
            results.append((output, confidence))

        return results

    def adaptive_forward(
        self, x: torch.Tensor, confidence_threshold: float = 0.8
    ) -> Tuple[torch.Tensor, int]:
        """Adaptive forward pass that exits early when confidence is high enough."""
        current_x = x

        for i in range(self.num_exits):
            # Forward through one more layer
            layer_idx = i * 2
            if layer_idx < len(self.backbone_layers):
                current_x = self.backbone_layers[layer_idx](current_x)
                if layer_idx + 1 < len(self.backbone_layers):
                    current_x = self.backbone_layers[layer_idx + 1](current_x)

            # Check confidence
            confidence = torch.sigmoid(self.confidence_heads[i](current_x))
            if torch.max(confidence) > confidence_threshold or i == self.num_exits - 1:
                output = self.exit_heads[i](current_x)
                return output, i

        # Fallback (should not reach here)
        return self.exit_heads[-1](current_x), self.num_exits - 1

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": "MultiExit",
            "complexity": self.complexity.value,
            "layer_sizes": self.layer_sizes,
            "num_exits": self.num_exits,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "parameters": self.count_parameters(),
        }


class PHAZEModelFactory:
    """Factory for creating PHAZE models."""

    @staticmethod
    def create_early_exit_model(
        architecture: str = "simple",
        complexity: ModelComplexity = ModelComplexity.LIGHT,
        input_size: int = 10,
        output_size: int = 5,
        **kwargs,
    ) -> PHAZEModelInterface:
        """Create an early-exit model."""

        if architecture == "simple":
            return SimpleEarlyExitModel(input_size, output_size)
        elif architecture == "conv":
            input_channels = kwargs.get("input_channels", 3)
            spatial_size = kwargs.get("spatial_size", 32)
            return ConvolutionalEarlyExitModel(
                input_channels, spatial_size, output_size, complexity
            )
        elif architecture == "transformer":
            return TransformerEarlyExitModel(input_size, output_size, complexity)
        elif architecture == "multi_exit":
            return MultiExitModel(input_size, output_size, complexity)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    @staticmethod
    def create_full_model(
        architecture: str = "simple",
        complexity: ModelComplexity = ModelComplexity.HEAVY,
        input_size: int = 10,
        output_size: int = 5,
        **kwargs,
    ) -> PHAZEModelInterface:
        """Create a full model (typically more complex than early-exit)."""

        # Full models are typically more complex versions of early-exit models
        if complexity == ModelComplexity.MINIMAL:
            complexity = ModelComplexity.LIGHT
        elif complexity == ModelComplexity.LIGHT:
            complexity = ModelComplexity.MEDIUM
        elif complexity == ModelComplexity.MEDIUM:
            complexity = ModelComplexity.HEAVY

        return PHAZEModelFactory.create_early_exit_model(
            architecture, complexity, input_size, output_size, **kwargs
        )

    @staticmethod
    def get_available_architectures() -> List[str]:
        """Get list of available architectures."""
        return ["simple", "conv", "transformer", "multi_exit"]

    @staticmethod
    def get_complexity_levels() -> List[ModelComplexity]:
        """Get list of available complexity levels."""
        return list(ModelComplexity)


# Convenience functions for backward compatibility
def create_simple_early_exit_model(
    input_size: int = 10, output_size: int = 5
) -> SimpleEarlyExitModel:
    """Create a simple early-exit model."""
    return SimpleEarlyExitModel(input_size, output_size)


def create_simple_full_model(
    input_size: int = 10, output_size: int = 5
) -> PHAZEModelInterface:
    """Create a simple full model."""
    return PHAZEModelFactory.create_full_model(
        "simple", ModelComplexity.MEDIUM, input_size, output_size
    )
