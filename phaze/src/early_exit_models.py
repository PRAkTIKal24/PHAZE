"""
Early exit models for the PHAZE framework.

This module provides backward compatibility while leveraging the new modular architecture.
"""

import torch
import torch.nn as nn
from .model_architectures import (
    SimpleEarlyExitModel as _SimpleEarlyExitModel,
    PHAZEModelFactory,
    ModelComplexity
)

# Re-export the main class for backward compatibility
class SimpleEarlyExitModel(_SimpleEarlyExitModel):
    """A simple toy model to demonstrate early-exit capabilities."""
    
    def __init__(self, input_size: int = 10, output_size: int = 5):
        super().__init__(input_size, output_size)


# Additional convenience functions
def create_early_exit_model(architecture: str = "simple", complexity: str = "light", **kwargs):
    """Create an early-exit model with specified architecture and complexity."""
    complexity_map = {
        "minimal": ModelComplexity.MINIMAL,
        "light": ModelComplexity.LIGHT,
        "medium": ModelComplexity.MEDIUM,
        "heavy": ModelComplexity.HEAVY,
        "extreme": ModelComplexity.EXTREME
    }
    
    complexity_enum = complexity_map.get(complexity, ModelComplexity.LIGHT)
    return PHAZEModelFactory.create_early_exit_model(architecture, complexity_enum, **kwargs)


def create_multi_exit_model(input_size: int = 10, output_size: int = 5, complexity: str = "medium"):
    """Create a multi-exit model."""
    return create_early_exit_model("multi_exit", complexity, input_size=input_size, output_size=output_size)


def create_transformer_model(input_size: int = 512, output_size: int = 10, complexity: str = "medium"):
    """Create a transformer-based early-exit model."""
    return create_early_exit_model("transformer", complexity, input_size=input_size, output_size=output_size)


def create_conv_model(input_channels: int = 3, spatial_size: int = 32, output_size: int = 10, complexity: str = "light"):
    """Create a convolutional early-exit model."""
    return create_early_exit_model(
        "conv", complexity, 
        input_channels=input_channels, 
        spatial_size=spatial_size, 
        output_size=output_size
    )

