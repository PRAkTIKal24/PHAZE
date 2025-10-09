"""
Multi-exit model generator for creating M_early variants from M_full models.

This module analyzes trained models and creates smaller variants by truncating
at different layers, generating models with specified parameter ratios.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .model_architectures import ModelComplexity, PHAZEModelFactory
from .training_config import PHAZEConfig

logger = logging.getLogger(__name__)


class MultiExitGenerator:
    """Generator for creating M_early variants from M_full models."""

    def __init__(self, config: PHAZEConfig):
        """Initialize generator with configuration.

        Args:
            config: PHAZE configuration object
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze_model_layers(self, model: nn.Module, architecture: str) -> Dict[str, Any]:
        """Analyze model structure to understand layer organization.

        Args:
            model: PyTorch model to analyze
            architecture: Model architecture type

        Returns:
            Dictionary with layer analysis information
        """
        layer_info = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'layers': [],
            'layer_parameters': [],
            'cumulative_parameters': [],
            'architecture': architecture
        }

        # Handle different model types
        if hasattr(model, 'base_model'):
            # PHAZEFullModel wrapper
            actual_model = model.base_model
        else:
            actual_model = model

        # Analyze based on architecture
        if architecture == "simple":
            layer_info = self._analyze_simple_model(actual_model, layer_info)
        elif architecture == "conv":
            layer_info = self._analyze_conv_model(actual_model, layer_info)
        elif architecture == "transformer":
            layer_info = self._analyze_transformer_model(actual_model, layer_info)
        elif architecture == "multi_exit":
            layer_info = self._analyze_multi_exit_model(actual_model, layer_info)
        else:
            logger.warning(f"Unknown architecture {architecture}, using generic analysis")
            layer_info = self._analyze_generic_model(actual_model, layer_info)

        return layer_info

    def _analyze_simple_model(self, model: nn.Module, layer_info: Dict) -> Dict:
        """Analyze simple model architecture."""
        for name, module in model.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                params = sum(p.numel() for p in module.parameters())
                layer_info['layers'].append(f"{name}_{type(module).__name__}")
                layer_info['layer_parameters'].append(params)

        # Calculate cumulative parameters
        cumulative = 0
        for params in layer_info['layer_parameters']:
            cumulative += params
            layer_info['cumulative_parameters'].append(cumulative)

        return layer_info

    def _analyze_conv_model(self, model: nn.Module, layer_info: Dict) -> Dict:
        """Analyze convolutional model architecture."""
        if hasattr(model, 'conv_layers'):
            # Analyze conv layers
            for i, layer in enumerate(model.conv_layers):
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    params = sum(p.numel() for p in layer.parameters())
                    layer_info['layers'].append(f"conv_{i}_{type(layer).__name__}")
                    layer_info['layer_parameters'].append(params)

        # Analyze classifier
        if hasattr(model, 'classifier'):
            params = sum(p.numel() for p in model.classifier.parameters())
            layer_info['layers'].append("classifier")
            layer_info['layer_parameters'].append(params)

        # Calculate cumulative parameters
        cumulative = 0
        for params in layer_info['layer_parameters']:
            cumulative += params
            layer_info['cumulative_parameters'].append(cumulative)

        return layer_info

    def _analyze_transformer_model(self, model: nn.Module, layer_info: Dict) -> Dict:
        """Analyze transformer model architecture."""
        # Input projection
        if hasattr(model, 'input_projection'):
            params = sum(p.numel() for p in model.input_projection.parameters())
            layer_info['layers'].append("input_projection")
            layer_info['layer_parameters'].append(params)

        # Transformer layers
        if hasattr(model, 'transformer'):
            for i, layer in enumerate(model.transformer.layers):
                params = sum(p.numel() for p in layer.parameters())
                layer_info['layers'].append(f"transformer_layer_{i}")
                layer_info['layer_parameters'].append(params)

        # Output layers
        if hasattr(model, 'classifier'):
            params = sum(p.numel() for p in model.classifier.parameters())
            layer_info['layers'].append("classifier")
            layer_info['layer_parameters'].append(params)

        # Calculate cumulative parameters
        cumulative = 0
        for params in layer_info['layer_parameters']:
            cumulative += params
            layer_info['cumulative_parameters'].append(cumulative)

        return layer_info

    def _analyze_multi_exit_model(self, model: nn.Module, layer_info: Dict) -> Dict:
        """Analyze multi-exit model architecture."""
        # Backbone layers
        if hasattr(model, 'backbone_layers'):
            for i, layer in enumerate(model.backbone_layers):
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    params = sum(p.numel() for p in layer.parameters())
                    layer_info['layers'].append(f"backbone_{i}")
                    layer_info['layer_parameters'].append(params)

        # Exit heads
        if hasattr(model, 'exit_heads'):
            for i, exit_head in enumerate(model.exit_heads):
                params = sum(p.numel() for p in exit_head.parameters())
                layer_info['layers'].append(f"exit_head_{i}")
                layer_info['layer_parameters'].append(params)

        # Calculate cumulative parameters
        cumulative = 0
        for params in layer_info['layer_parameters']:
            cumulative += params
            layer_info['cumulative_parameters'].append(cumulative)

        return layer_info

    def _analyze_generic_model(self, model: nn.Module, layer_info: Dict) -> Dict:
        """Generic model analysis for unknown architectures."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and len(list(module.children())) == 0:
                params = sum(p.numel() for p in module.parameters())
                layer_info['layers'].append(name)
                layer_info['layer_parameters'].append(params)

        # Calculate cumulative parameters
        cumulative = 0
        for params in layer_info['layer_parameters']:
            cumulative += params
            layer_info['cumulative_parameters'].append(cumulative)

        return layer_info

    def find_truncation_points(self, layer_info: Dict[str, Any],
                              target_ratios: List[float]) -> List[Tuple[int, float]]:
        """Find layer indices that achieve target parameter ratios.

        Args:
            layer_info: Layer analysis information
            target_ratios: List of target parameter ratios (0.0 to 1.0)

        Returns:
            List of (layer_index, actual_ratio) tuples
        """
        total_params = layer_info['total_parameters']
        cumulative_params = layer_info['cumulative_parameters']

        truncation_points = []

        for target_ratio in target_ratios:
            target_params = total_params * target_ratio

            # Find closest layer index
            best_idx = 0
            best_diff = float('inf')

            for i, cum_params in enumerate(cumulative_params):
                diff = abs(cum_params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            actual_ratio = cumulative_params[best_idx] / total_params
            truncation_points.append((best_idx, actual_ratio))

            logger.info(f"Target ratio {target_ratio:.2f} -> Layer {best_idx}, "
                       f"Actual ratio {actual_ratio:.3f}")

        return truncation_points

    def create_early_exit_model(self, full_model: nn.Module, architecture: str,
                               truncation_layer: int, model_info: Dict[str, Any]) -> nn.Module:
        """Create an early exit model by truncating at specified layer.

        Args:
            full_model: Original full model
            architecture: Model architecture type
            truncation_layer: Layer index to truncate at
            model_info: Original model information

        Returns:
            Truncated early exit model
        """
        # Handle different model types
        if hasattr(full_model, 'base_model'):
            source_model = full_model.base_model
        else:
            source_model = full_model

        # Create truncated model based on architecture
        if architecture == "simple":
            return self._create_simple_early_exit(source_model, truncation_layer, model_info)
        elif architecture == "conv":
            return self._create_conv_early_exit(source_model, truncation_layer, model_info)
        elif architecture == "transformer":
            return self._create_transformer_early_exit(source_model, truncation_layer, model_info)
        elif architecture == "multi_exit":
            return self._create_multi_exit_early_exit(source_model, truncation_layer, model_info)
        else:
            logger.warning(f"Unknown architecture {architecture}, using generic method")
            return self._create_generic_early_exit(source_model, truncation_layer, model_info)

    def _create_simple_early_exit(self, model: nn.Module, truncation_layer: int,
                                 model_info: Dict[str, Any]) -> nn.Module:
        """Create early exit from simple model."""
        # Create a new simple model with reduced layers
        from .model_architectures import SimpleEarlyExitModel

        # Determine appropriate sizes based on truncation
        if truncation_layer == 0:
            # Very early exit - minimal model
            early_model = SimpleEarlyExitModel(
                input_size=model_info.get('input_size', 784),
                output_size=model_info.get('output_size', 10)
            )
        else:
            # Copy weights up to truncation point
            early_model = SimpleEarlyExitModel(
                input_size=model_info.get('input_size', 784),
                output_size=model_info.get('output_size', 10)
            )

            # Copy available weights
            with torch.no_grad():
                if hasattr(model, 'fc1') and hasattr(early_model, 'fc1'):
                    early_model.fc1.weight.data = model.fc1.weight.data.clone()
                    early_model.fc1.bias.data = model.fc1.bias.data.clone()

        return early_model

    def _create_conv_early_exit(self, model: nn.Module, truncation_layer: int,
                               model_info: Dict[str, Any]) -> nn.Module:
        """Create early exit from convolutional model."""
        # Create a simpler conv model
        complexity_map = {
            0: ModelComplexity.MINIMAL,
            1: ModelComplexity.MINIMAL,
            2: ModelComplexity.LIGHT,
        }

        target_complexity = complexity_map.get(truncation_layer, ModelComplexity.MINIMAL)

        from .model_architectures import ConvolutionalEarlyExitModel
        early_model = ConvolutionalEarlyExitModel(
            input_channels=model_info.get('input_channels', 1),
            input_size=model_info.get('spatial_size', 28),
            output_size=model_info.get('output_size', 10),
            complexity=target_complexity
        )

        # Copy compatible weights
        self._copy_compatible_weights(model, early_model)

        return early_model

    def _create_transformer_early_exit(self, model: nn.Module, truncation_layer: int,
                                      model_info: Dict[str, Any]) -> nn.Module:
        """Create early exit from transformer model."""
        # Create transformer with fewer layers
        num_layers = max(1, truncation_layer // 2)  # Rough estimate

        complexity_map = {
            0: ModelComplexity.MINIMAL,
            1: ModelComplexity.MINIMAL,
            2: ModelComplexity.LIGHT,
            3: ModelComplexity.LIGHT,
        }

        target_complexity = complexity_map.get(num_layers, ModelComplexity.MINIMAL)

        from .model_architectures import TransformerEarlyExitModel
        early_model = TransformerEarlyExitModel(
            input_size=model_info.get('input_size', 512),
            output_size=model_info.get('output_size', 10),
            complexity=target_complexity
        )

        # Copy compatible weights
        self._copy_compatible_weights(model, early_model)

        return early_model

    def _create_multi_exit_early_exit(self, model: nn.Module, truncation_layer: int,
                                     model_info: Dict[str, Any]) -> nn.Module:
        """Create early exit from multi-exit model."""
        # For multi-exit models, we can use the existing exit points
        if hasattr(model, 'exit_heads') and truncation_layer < len(model.exit_heads):
            # Create a new model that only goes to the specified exit
            from .model_architectures import MultiExitModel

            # Determine complexity based on truncation layer
            complexity_map = {
                0: ModelComplexity.MINIMAL,
                1: ModelComplexity.LIGHT,
                2: ModelComplexity.MEDIUM,
            }

            target_complexity = complexity_map.get(truncation_layer, ModelComplexity.MINIMAL)

            early_model = MultiExitModel(
                input_size=model_info.get('input_size', 10),
                output_size=model_info.get('output_size', 5),
                complexity=target_complexity
            )

            # Copy compatible weights
            self._copy_compatible_weights(model, early_model)

            return early_model
        else:
            # Fall back to generic method
            return self._create_generic_early_exit(model, truncation_layer, model_info)

    def _create_generic_early_exit(self, model: nn.Module, truncation_layer: int,
                                  model_info: Dict[str, Any]) -> nn.Module:
        """Generic early exit creation for unknown architectures."""
        # Create a simple model as fallback
        from .model_architectures import SimpleEarlyExitModel

        early_model = SimpleEarlyExitModel(
            input_size=model_info.get('input_size', 784),
            output_size=model_info.get('output_size', 10)
        )

        return early_model

    def _copy_compatible_weights(self, source_model: nn.Module, target_model: nn.Module) -> None:
        """Copy compatible weights between models."""
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()

        # Copy weights that have matching names and shapes
        for name, param in target_dict.items():
            if name in source_dict and source_dict[name].shape == param.shape:
                with torch.no_grad():
                    param.copy_(source_dict[name])
                logger.debug(f"Copied weights for layer: {name}")

    def generate_early_exit_variants(self, full_model: nn.Module, architecture: str,
                                   complexity: str, model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple early exit variants from a full model.

        Args:
            full_model: Trained full model
            architecture: Model architecture type
            complexity: Original model complexity
            model_info: Original model information

        Returns:
            List of early exit model information dictionaries
        """
        logger.info(f"Generating early exit variants for {architecture} {complexity}")

        # Analyze model structure
        layer_info = self.analyze_model_layers(full_model, architecture)

        # Find truncation points for target ratios
        truncation_points = self.find_truncation_points(
            layer_info,
            self.config.model.early_exit_ratios
        )

        early_exit_variants = []

        for i, (truncation_layer, actual_ratio) in enumerate(truncation_points):
            try:
                # Create early exit model
                early_model = self.create_early_exit_model(
                    full_model, architecture, truncation_layer, model_info
                )

                # Move to device
                early_model = early_model.to(self.device)

                # Calculate actual parameters
                early_params = sum(p.numel() for p in early_model.parameters())

                # Create model info
                early_info = {
                    'model_id': f"{model_info['model_id']}_early_{int(actual_ratio*100)}pct",
                    'parent_model_id': model_info['model_id'],
                    'architecture': architecture,
                    'complexity': f"{complexity}_early_{int(actual_ratio*100)}pct",
                    'truncation_layer': truncation_layer,
                    'target_ratio': self.config.model.early_exit_ratios[i],
                    'actual_ratio': actual_ratio,
                    'parameters': early_params,
                    'parent_parameters': model_info['parameters'],
                    'parameter_reduction': 1 - (early_params / model_info['parameters']),
                    'model': early_model,
                    'input_size': model_info.get('input_size'),
                    'output_size': model_info.get('output_size'),
                    'input_channels': model_info.get('input_channels'),
                    'spatial_size': model_info.get('spatial_size')
                }

                early_exit_variants.append(early_info)

                logger.info(f"Created early exit variant: {early_info['model_id']}")
                logger.info(f"  Parameters: {early_params:,} ({actual_ratio:.1%} of original)")

            except Exception as e:
                logger.error(f"Failed to create early exit variant {i}: {e}")
                continue

        logger.info(f"Generated {len(early_exit_variants)} early exit variants")
        return early_exit_variants

    def select_largest_models(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the largest model from each architecture for early exit generation.

        Args:
            training_results: Dictionary of training results

        Returns:
            Dictionary mapping architecture to largest model info
        """
        largest_models = {}

        for _model_id, model_info in training_results.items():
            architecture = model_info['architecture']
            parameters = model_info['parameters']

            if (architecture not in largest_models or
                parameters > largest_models[architecture]['parameters']):
                largest_models[architecture] = model_info

        logger.info("Selected largest models for early exit generation:")
        for arch, model_info in largest_models.items():
            logger.info(f"  {arch}: {model_info['model_id']} ({model_info['parameters']:,} params)")

        return largest_models

    def save_early_exit_model(self, early_info: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
        """Save early exit model and metadata.

        Args:
            early_info: Early exit model information
            output_dir: Output directory

        Returns:
            Dictionary with saved file paths
        """
        model_id = early_info['model_id']
        model = early_info['model']
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save model state dict
        model_path = output_dir / f"{model_id}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': early_info,
            'config': self.config.to_dict()
        }, model_path)
        saved_files['model'] = str(model_path)

        # Save metadata
        import json
        metadata_path = output_dir / f"{model_id}_metadata.json"
        metadata = {k: v for k, v in early_info.items() if k != 'model'}  # Exclude model object
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_files['metadata'] = str(metadata_path)

        return saved_files


# Convenience functions
def generate_early_exits_from_model(config: PHAZEConfig, full_model: nn.Module,
                                   architecture: str, complexity: str,
                                   model_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate early exit variants from a single model.

    Args:
        config: PHAZE configuration
        full_model: Trained full model
        architecture: Model architecture
        complexity: Model complexity
        model_info: Model information

    Returns:
        List of early exit model information
    """
    generator = MultiExitGenerator(config)
    return generator.generate_early_exit_variants(
        full_model, architecture, complexity, model_info
    )


if __name__ == "__main__":
    # Example usage
    from .model_architectures import ModelComplexity, PHAZEModelFactory
    from .training_config import create_default_config

    config = create_default_config()
    generator = MultiExitGenerator(config)

    # Create a test model
    model = PHAZEModelFactory.create_full_model(
        "simple", ModelComplexity.MEDIUM, 784, 10
    )

    # Analyze model
    layer_info = generator.analyze_model_layers(model, "simple")
    print("Layer analysis:")
    for i, (layer, params) in enumerate(zip(layer_info['layers'], layer_info['layer_parameters'], strict=False)):
        print(f"  {i}: {layer} - {params:,} parameters")

    print(f"Total parameters: {layer_info['total_parameters']:,}")

    # Find truncation points
    truncation_points = generator.find_truncation_points(layer_info, [0.25, 0.5, 0.75])
    print("Truncation points:")
    for layer_idx, ratio in truncation_points:
        print(f"  Layer {layer_idx}: {ratio:.3f} of parameters")
