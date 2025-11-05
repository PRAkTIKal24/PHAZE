"""
RISC Zero code generation for PHAZE model architectures.

This module provides dynamic generation of RISC Zero guest programs based on
PyTorch model architectures from the PHAZE model factory.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model_architectures import ModelComplexity, PHAZEModelFactory

logger = logging.getLogger(__name__)


class RiscZeroArchitectureRegistry:
    """Registry for RISC Zero-compatible model architectures."""

    def __init__(self):
        self.architectures = {}
        self.templates_dir = (
            Path(__file__).parent.parent.parent / "rust_bindings" / "guest_templates"
        )
        self.guest_programs_dir = (
            Path(__file__).parent.parent.parent / "rust_bindings" / "guest_programs"
        )
        self.registry_file = self.guest_programs_dir / "architecture_registry.json"

        # Ensure directories exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.guest_programs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load_registry()

    def _load_registry(self):
        """Load the architecture registry from file."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    self.architectures = json.load(f)
                logger.info(
                    f"Loaded {len(self.architectures)} registered architectures"
                )
            except Exception as e:
                logger.warning(f"Failed to load architecture registry: {e}")
                self.architectures = {}
        else:
            logger.info("No existing architecture registry found, starting fresh")

    def _save_registry(self):
        """Save the architecture registry to file."""
        try:
            with open(self.registry_file, "w") as f:
                json.dump(self.architectures, f, indent=2)
            logger.debug("Architecture registry saved")
        except Exception as e:
            logger.error(f"Failed to save architecture registry: {e}")

    def register_architecture(
        self, architecture: str, complexity: ModelComplexity, model_info: Dict[str, Any]
    ) -> str:
        """Register a new architecture for RISC Zero generation.

        Args:
            architecture: Architecture name (e.g., 'simple', 'conv', 'multi_exit')
            complexity: Model complexity level
            model_info: Model information including layer sizes, parameters, etc.

        Returns:
            Unique architecture key for this configuration
        """
        arch_key = f"{architecture}_{complexity.value}"

        if arch_key in self.architectures:
            logger.debug(f"Architecture {arch_key} already registered")
            return arch_key

        # Extract relevant model structure information
        arch_config = {
            "architecture": architecture,
            "complexity": complexity.value,
            "input_size": model_info.get("input_size", 784),
            "output_size": model_info.get("output_size", 10),
            "layer_sizes": model_info.get("layer_sizes", []),
            "parameters": model_info.get("parameters", 0),
            "template_name": self._get_template_name(architecture),
            "guest_program_name": f"guest_{arch_key}",
        }

        self.architectures[arch_key] = arch_config
        self._save_registry()

        logger.info(f"Registered new architecture: {arch_key}")
        return arch_key

    def _get_template_name(self, architecture: str) -> str:
        """Get the template name for an architecture."""
        template_mapping = {
            "simple": "simple_model.rs.template",
            "conv": "conv_model.rs.template",
            "transformer": "transformer_model.rs.template",
            "multi_exit": "multi_exit_model.rs.template",
        }
        return template_mapping.get(architecture, "generic_model.rs.template")

    def get_architecture_info(self, arch_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered architecture."""
        return self.architectures.get(arch_key)

    def list_architectures(self) -> List[str]:
        """List all registered architecture keys."""
        return list(self.architectures.keys())

    def is_registered(self, architecture: str, complexity: ModelComplexity) -> bool:
        """Check if an architecture-complexity combination is registered."""
        arch_key = f"{architecture}_{complexity.value}"
        return arch_key in self.architectures

    def scan_model_factory(
        self, dataset_config: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, ModelComplexity, Dict[str, Any]]]:
        """Scan the model factory for all possible architectures and complexities.

        Args:
            dataset_config: Optional dataset configuration with input_size, output_size, etc.
                          If None, uses MNIST defaults for backward compatibility.

        Returns:
            List of (architecture, complexity, model_info) tuples
        """
        # TEMP: Only focus on multi-exit models for faster development
        architectures = ["multi_exit"]  # PHAZEModelFactory.get_available_architectures()
        complexities = PHAZEModelFactory.get_complexity_levels()
        
        logger.info(f"TEMP: Only using architectures: {architectures}")
        logger.info(f"TEMP: Using complexities: {[complexity.value for complexity in complexities]}")

        # Set dataset-specific defaults
        if dataset_config is None:
            # MNIST defaults for backward compatibility
            dataset_config = {
                "input_size": 784,  # 28*28 flattened
                "output_size": 10,  # 10 digit classes
                "input_channels": 1,  # Grayscale
                "spatial_size": 28,  # 28x28 images
                "dataset_name": "mnist",
            }

        logger.info(
            f"Scanning model factory for dataset: {dataset_config.get('dataset_name', 'custom')}"
        )
        logger.info(f"  Input size: {dataset_config['input_size']}")
        logger.info(f"  Output size: {dataset_config['output_size']}")

        model_configs = []

        for architecture in architectures:
            for complexity in complexities:
                try:
                    # TEMP: Only multi-exit models, so we always use flattened input
                    model = PHAZEModelFactory.create_full_model(
                        architecture=architecture,
                        complexity=complexity,
                        input_size=dataset_config["input_size"],
                        output_size=dataset_config["output_size"],
                    )

                    # Extract model information and add dataset info
                    model_info = model.get_model_info()
                    model_info.update(dataset_config)  # Include dataset configuration
                    model_configs.append((architecture, complexity, model_info))

                except Exception as e:
                    logger.warning(
                        f"Failed to create model {architecture}_{complexity.value}: {e}"
                    )
                    continue

        return model_configs

    def clear_registry(self):
        """Clear the current architecture registry."""
        self.architectures = {}
        self._save_registry()
        logger.info("Cleared architecture registry")

    def register_all_factory_models(
        self, dataset_config: Optional[Dict[str, Any]] = None
    ) -> int:
        """Register all models from the factory.

        Args:
            dataset_config: Optional dataset configuration for model creation

        Returns:
            Number of newly registered architectures
        """
        model_configs = self.scan_model_factory(dataset_config)
        newly_registered = 0

        for architecture, complexity, model_info in model_configs:
            arch_key = f"{architecture}_{complexity.value}"
            if arch_key not in self.architectures:
                self.register_architecture(architecture, complexity, model_info)
                newly_registered += 1

        dataset_name = (
            dataset_config.get("dataset_name", "default") if dataset_config else "mnist"
        )
        logger.info(
            f"Registered {newly_registered} new architectures from model factory for dataset: {dataset_name}"
        )
        return newly_registered

    def register_multi_exit_only(
        self, dataset_config: Optional[Dict[str, Any]] = None
    ) -> int:
        """Clear registry and register only multi-exit models.

        Args:
            dataset_config: Optional dataset configuration for model creation

        Returns:
            Number of registered architectures
        """
        # Clear existing registry
        self.clear_registry()
        
        # Register only multi-exit models
        return self.register_all_factory_models(dataset_config)


class RiscZeroTemplateEngine:
    """Template engine for generating RISC Zero guest programs."""

    def __init__(self, registry: RiscZeroArchitectureRegistry):
        self.registry = registry
        self.templates_dir = registry.templates_dir

    def create_templates(self):
        """Create initial template files for all supported architectures."""
        # TEMP: Only create multi-exit template for faster development
        templates = {
            # "simple_model.rs.template": self._generate_simple_template(),
            # "conv_model.rs.template": self._generate_conv_template(),
            # "transformer_model.rs.template": self._generate_transformer_template(),
            "multi_exit_model.rs.template": self._generate_multi_exit_template(),
            "generic_model.rs.template": self._generate_generic_template(),
        }

        for template_name, content in templates.items():
            template_path = self.templates_dir / template_name
            with open(template_path, "w") as f:
                f.write(content)
            logger.debug(f"Created template: {template_path}")

    def _generate_simple_template(self) -> str:
        """Generate template for simple feed-forward models."""
        return """#![no_std]
extern crate alloc;

use alloc::{vec, vec::Vec};
use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ModelWeights {
    fc1_weights: Vec<Vec<f32>>,
    fc1_bias: Vec<f32>,
    fc2_weights: Vec<Vec<f32>>,
    fc2_bias: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelInput {
    input_tensor: Vec<f32>,
    weights: ModelWeights,
}

#[derive(Serialize, Deserialize)]
pub struct ModelOutput {
    output_tensor: Vec<f32>,
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; weights.len()];

    for i in 0..weights.len() {
        let mut sum = bias[i];
        for j in 0..input.len() {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    output
}

fn forward(input: &[f32], weights: &ModelWeights) -> Vec<f32> {
    // First layer: Linear + ReLU
    let hidden = linear(input, &weights.fc1_weights, &weights.fc1_bias);
    let hidden_activated = hidden.iter().map(|&x| relu(x)).collect::<Vec<_>>();

    // Output layer: Linear
    let output = linear(&hidden_activated, &weights.fc2_weights, &weights.fc2_bias);

    output
}

fn main() {
    let input: ModelInput = env::read();
    let output_tensor = forward(&input.input_tensor, &input.weights);
    let output = ModelOutput { output_tensor };
    env::commit(&output);
}
"""

    def _generate_conv_template(self) -> str:
        """Generate template for convolutional models."""
        return """#![no_std]
extern crate alloc;

use alloc::{vec, vec::Vec};
use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct ConvWeights {
    conv_weights: Vec<Vec<Vec<Vec<f32>>>>, // [out_ch, in_ch, h, w]
    conv_bias: Vec<f32>,
    fc_weights: Vec<Vec<f32>>,
    fc_bias: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelInput {
    input_tensor: Vec<f32>,
    weights: ConvWeights,
    input_shape: (usize, usize, usize), // (channels, height, width)
}

#[derive(Serialize, Deserialize)]
pub struct ModelOutput {
    output_tensor: Vec<f32>,
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn conv2d(input: &[f32], weights: &[Vec<Vec<Vec<f32>>>], bias: &[f32],
          input_shape: (usize, usize, usize)) -> Vec<f32> {
    // Simplified convolution - this would be more complex in practice
    // For now, just apply a basic transformation
    let output_size = weights.len();
    let mut output = vec![0.0; output_size];

    for i in 0..output_size {
        output[i] = bias[i];
        for j in 0..input.len().min(weights[i][0][0].len()) {
            output[i] += input[j] * weights[i][0][0][j % weights[i][0][0].len()];
        }
        output[i] = relu(output[i]);
    }

    output
}

fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; weights.len()];

    for i in 0..weights.len() {
        let mut sum = bias[i];
        for j in 0..input.len() {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    output
}

fn forward(input: &[f32], weights: &ConvWeights, input_shape: (usize, usize, usize)) -> Vec<f32> {
    // Convolution layers
    let conv_output = conv2d(input, &weights.conv_weights, &weights.conv_bias, input_shape);

    // Flatten and apply fully connected
    let output = linear(&conv_output, &weights.fc_weights, &weights.fc_bias);

    output
}

fn main() {
    let input: ModelInput = env::read();
    let output_tensor = forward(&input.input_tensor, &input.weights, input.input_shape);
    let output = ModelOutput { output_tensor };
    env::commit(&output);
}
"""

    def _generate_transformer_template(self) -> str:
        """Generate template for transformer models."""
        return """#![no_std]
extern crate alloc;

use alloc::{vec, vec::Vec};
use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TransformerWeights {
    input_projection: Vec<Vec<f32>>,
    input_bias: Vec<f32>,
    attention_weights: Vec<Vec<f32>>,
    attention_bias: Vec<f32>,
    output_weights: Vec<Vec<f32>>,
    output_bias: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelInput {
    input_tensor: Vec<f32>,
    weights: TransformerWeights,
}

#[derive(Serialize, Deserialize)]
pub struct ModelOutput {
    output_tensor: Vec<f32>,
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; weights.len()];

    for i in 0..weights.len() {
        let mut sum = bias[i];
        for j in 0..input.len() {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    output
}

fn forward(input: &[f32], weights: &TransformerWeights) -> Vec<f32> {
    // Input projection
    let projected = linear(input, &weights.input_projection, &weights.input_bias);
    let projected_activated = projected.iter().map(|&x| relu(x)).collect::<Vec<_>>();

    // Simplified attention (just another linear layer for this template)
    let attended = linear(&projected_activated, &weights.attention_weights, &weights.attention_bias);
    let attended_activated = attended.iter().map(|&x| relu(x)).collect::<Vec<_>>();

    // Output projection
    let output = linear(&attended_activated, &weights.output_weights, &weights.output_bias);

    output
}

fn main() {
    let input: ModelInput = env::read();
    let output_tensor = forward(&input.input_tensor, &input.weights);
    let output = ModelOutput { output_tensor };
    env::commit(&output);
}
"""

    def _generate_multi_exit_template(self) -> str:
        """Generate template for multi-exit models."""
        return """#![no_std]
extern crate alloc;

use alloc::{vec, vec::Vec};
use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MultiExitWeights {
    backbone_weights: Vec<Vec<Vec<f32>>>, // Multiple layers
    backbone_bias: Vec<Vec<f32>>,
    exit_weights: Vec<Vec<Vec<f32>>>, // Multiple exit heads
    exit_bias: Vec<Vec<f32>>,
    exit_layer: usize, // Which exit to use
}

#[derive(Serialize, Deserialize)]
pub struct ModelInput {
    input_tensor: Vec<f32>,
    weights: MultiExitWeights,
}

#[derive(Serialize, Deserialize)]
pub struct ModelOutput {
    output_tensor: Vec<f32>,
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn linear(input: &[f32], weights: &[Vec<f32>], bias: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; weights.len()];

    for i in 0..weights.len() {
        let mut sum = bias[i];
        for j in 0..input.len() {
            sum += weights[i][j] * input[j];
        }
        output[i] = sum;
    }

    output
}

fn forward(input: &[f32], weights: &MultiExitWeights) -> Vec<f32> {
    let mut current = input.to_vec();

    // Forward through backbone layers up to exit point
    for layer_idx in 0..=weights.exit_layer {
        if layer_idx < weights.backbone_weights.len() {
            current = linear(&current, &weights.backbone_weights[layer_idx], &weights.backbone_bias[layer_idx]);
            current = current.iter().map(|&x| relu(x)).collect();
        }
    }

    // Apply exit head
    if weights.exit_layer < weights.exit_weights.len() {
        linear(&current, &weights.exit_weights[weights.exit_layer], &weights.exit_bias[weights.exit_layer])
    } else {
        current
    }
}

fn main() {
    let input: ModelInput = env::read();
    let output_tensor = forward(&input.input_tensor, &input.weights);
    let output = ModelOutput { output_tensor };
    env::commit(&output);
}
"""

    def _generate_generic_template(self) -> str:
        """Generate generic template for unknown architectures."""
        return self._generate_simple_template()  # Fallback to simple template

    def generate_guest_program(self, arch_key: str) -> Optional[Path]:
        """Generate a guest program for a specific architecture.

        Args:
            arch_key: Architecture key from registry

        Returns:
            Path to generated guest program directory
        """
        arch_info = self.registry.get_architecture_info(arch_key)
        if not arch_info:
            logger.error(f"Architecture {arch_key} not found in registry")
            return None

        template_name = arch_info["template_name"]
        template_path = self.templates_dir / template_name

        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            return None

        # Create guest program directory
        guest_program_name = arch_info["guest_program_name"]
        guest_dir = self.registry.guest_programs_dir / guest_program_name
        guest_dir.mkdir(parents=True, exist_ok=True)

        # Create Cargo.toml for the guest program
        cargo_toml = self._generate_cargo_toml(guest_program_name, arch_info)
        with open(guest_dir / "Cargo.toml", "w") as f:
            f.write(cargo_toml)

        # Create src directory
        src_dir = guest_dir / "src"
        src_dir.mkdir(exist_ok=True)

        # Copy and customize template
        with open(template_path, "r") as f:
            template_content = f.read()

        # Apply any architecture-specific customizations
        customized_content = self._customize_template(template_content, arch_info)

        # Write main.rs with the actual implementation
        # Extract and modify the template content to work as main.rs
        lines = customized_content.split('\n')
        
        # Build the main.rs content with proper attribute ordering
        main_rs_content = []
        main_rs_content.append("#![no_main]")
        
        # Add all the template content except the first #![no_std] line
        for line in lines:
            if line.strip() == "#![no_std]":
                main_rs_content.append("#![no_std]")  # Keep no_std but after no_main
            else:
                main_rs_content.append(line)
        
        main_rs_content.append("")
        main_rs_content.append("use risc0_zkvm::guest::entry;")
        main_rs_content.append("entry!(main);")
        
        with open(src_dir / "main.rs", "w") as f:
            f.write('\n'.join(main_rs_content))

        # Write lib.rs (empty for now, could be used for shared utilities)
        with open(src_dir / "lib.rs", "w") as f:
            f.write("// Shared utilities for the guest program\n")

        logger.info(f"Generated guest program: {guest_dir}")
        return guest_dir

    def _generate_cargo_toml(self, program_name: str, arch_info: Dict[str, Any]) -> str:
        """Generate Cargo.toml for a guest program."""
        return f'''[package]
name = "{program_name}"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "{program_name}"
path = "src/main.rs"

[dependencies]
risc0-zkvm = {{ version = "3.0.3", default-features = false, features = ["std"] }}
serde = {{ version = "1.0", features = ["derive"] }}

# Architecture: {arch_info["architecture"]}
# Complexity: {arch_info["complexity"]}
# Input size: {arch_info["input_size"]}
# Output size: {arch_info["output_size"]}
# Parameters: {arch_info["parameters"]}
'''

    def _customize_template(
        self, template_content: str, arch_info: Dict[str, Any]
    ) -> str:
        """Apply architecture-specific customizations to template."""
        # For now, just return the template as-is
        # Future: could replace placeholders with actual values
        return template_content


class RiscZeroBuildManager:
    """Manager for building RISC Zero guest programs."""

    def __init__(self, registry: RiscZeroArchitectureRegistry):
        self.registry = registry

    def build_all_guest_programs(self) -> Dict[str, bool]:
        """Build all registered guest programs.

        Returns:
            Dictionary mapping architecture keys to build success status
        """
        results = {}
        template_engine = RiscZeroTemplateEngine(self.registry)

        # Create templates first
        template_engine.create_templates()

        for arch_key in self.registry.list_architectures():
            logger.info(f"Building guest program for {arch_key}...")

            # Generate guest program
            guest_dir = template_engine.generate_guest_program(arch_key)
            if not guest_dir:
                results[arch_key] = False
                continue

            # Build the guest program
            success = self._build_guest_program(guest_dir)
            results[arch_key] = success

            if success:
                logger.info(f"✅ Successfully built {arch_key}")
            else:
                logger.error(f"❌ Failed to build {arch_key}")

        return results

    def _build_guest_program(self, guest_dir: Path) -> bool:
        """Build a single guest program using cargo risczero build.

        Args:
            guest_dir: Path to guest program directory

        Returns:
            True if build succeeded
        """
        try:
            # Check if cargo-risczero is available
            result = subprocess.run(
                ["cargo", "risczero", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(
                    "cargo-risczero not found. Please install RISC Zero toolchain."
                )
                return False

            # Build the guest program
            result = subprocess.run(
                ["cargo", "risczero", "build"],
                cwd=guest_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # Verify the ELF was created
                elf_path = (
                    guest_dir
                    / "target"
                    / "riscv32im-risc0-zkvm-elf"
                    / "release"
                    / guest_dir.name
                )
                if elf_path.exists():
                    logger.debug(f"ELF file created: {elf_path}")
                    return True
                else:
                    logger.warning(f"Build succeeded but ELF not found: {elf_path}")
                    return False
            else:
                logger.error(f"Build failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error building guest program: {e}")
            return False

    def get_guest_program_path(self, arch_key: str) -> Optional[Path]:
        """Get the path to a built guest program ELF.

        Args:
            arch_key: Architecture key

        Returns:
            Path to ELF file if it exists
        """
        arch_info = self.registry.get_architecture_info(arch_key)
        if not arch_info:
            return None

        guest_program_name = arch_info["guest_program_name"]
        guest_dir = self.registry.guest_programs_dir / guest_program_name
        elf_path = (
            guest_dir
            / "target"
            / "riscv32im-risc0-zkvm-elf"
            / "release"
            / guest_program_name
        )

        return elf_path if elf_path.exists() else None
