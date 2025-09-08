"""
zkML integration for the PHAZE framework.

This module provides integration between PyTorch models
and zero-knowledge machine learning
frameworks, with support for the new modular architecture.
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple  # noqa: F401

import numpy as np
import torch
import torch.nn as nn

from .model_architectures import ModelComplexity, PHAZEModelFactory

try:
    import ezkl

    EZKL_AVAILABLE = True
except ImportError:
    print("ezkl not found. Please install ezkl to enable full ZKML functionality.")
    EZKL_AVAILABLE = False


class SimpleFullModel(nn.Module):
    """A simple toy model to represent M_full."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class PHAZEFullModel(nn.Module):
    """Enhanced full model using the new architecture system."""

    def __init__(
        self,
        architecture: str = "simple",
        complexity: ModelComplexity = ModelComplexity.HEAVY,
        input_size: int = 10,
        output_size: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.base_model = PHAZEModelFactory.create_full_model(
            architecture, complexity, input_size, output_size, **kwargs
        )
        self.architecture = architecture
        self.complexity = complexity

    def forward(self, x):
        return self.base_model(x)

    def get_confidence(self, x):
        """Get confidence scores if the base model supports it."""
        if hasattr(self.base_model, "get_confidence"):
            return self.base_model.get_confidence(x)
        else:
            # Fallback: use softmax probabilities as confidence
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                return confidence.unsqueeze(-1)

    def get_model_info(self):
        """Get model information."""
        if hasattr(self.base_model, "get_model_info"):
            return self.base_model.get_model_info()
        else:
            return {
                "architecture": self.architecture,
                "complexity": self.complexity.value
                if hasattr(self.complexity, "value")
                else str(self.complexity),
                "parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            }


class ZKMLProverVerifier:
    """
    A class for integrating and testing ZK-SNARK based ZKML systems using ezkl.

    This class provides a high-level interface for:
    1. Compiling a PyTorch model to a zk-SNARK circuit.
    2. Generating a zero-knowledge proof of a model inference.
    3. Verifying the proof.
    """

    def __init__(self, model: nn.Module, zkml_system_name: str = "ezkl"):
        if not EZKL_AVAILABLE:
            print("Warning: ezkl is not available. Some functionality will be limited.")

        self.model = model
        self.zkml_system_name = zkml_system_name
        self.model_name = model.__class__.__name__
        self.onnx_path = f"/tmp/{self.model_name}.onnx"
        self.compiled_model_path = f"/tmp/{self.model_name}.compiled"
        self.pk_path = f"/tmp/{self.model_name}.pk"
        self.vk_path = f"/tmp/{self.model_name}.vk"
        self.settings_path = f"/tmp/{self.model_name}_settings.json"
        self.witness_path = f"/tmp/{self.model_name}.witness.json"
        self.proof_path = f"/tmp/{self.model_name}.proof"
        self.input_json_path = f"/tmp/{self.model_name}_input.json"
        self.output_json_path = f"/tmp/{self.model_name}_output.json"
        self.srs_path = os.path.join(
            os.path.expanduser("~"), ".ezkl", "srs", "kzg17.srs"
        )

    def _export_to_onnx(self, input_data: torch.Tensor):
        """Exports the PyTorch model to ONNX format."""
        # Handle different model types
        if hasattr(self.model, "base_model"):
            # PHAZEFullModel
            model_to_export = self.model.base_model
        else:
            model_to_export = self.model

        torch.onnx.export(
            model_to_export,
            input_data,
            self.onnx_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    async def _async_setup(self, input_data: torch.Tensor):
        """Asynchronous part of the setup process."""
        if not EZKL_AVAILABLE:
            raise ImportError("ezkl is not available")

        # Generate settings
        ezkl.gen_settings(self.onnx_path, self.settings_path)

        # Download SRS if not exists
        if not os.path.exists(self.srs_path):
            print(f"Downloading SRS to {self.srs_path}...")
            os.makedirs(os.path.dirname(self.srs_path), exist_ok=True)

            # The ezkl.get_srs function can take settings_path directly.
            # Ensure only keyword arguments are used for get_srs
            await ezkl.get_srs(srs_path=self.srs_path, settings_path=self.settings_path)

        # Compile the model
        ezkl.compile_circuit(
            self.onnx_path, self.compiled_model_path, self.settings_path
        )

        # Generate proving and verification keys
        ezkl.setup(
            self.compiled_model_path, self.vk_path, self.pk_path, srs_path=self.srs_path
        )

    def setup(self, input_data: torch.Tensor):
        """Sets up the ZKML system for the model."""
        self._export_to_onnx(input_data)
        # Run the async setup function in a new event loop
        asyncio.run(self._async_setup(input_data))

    async def generate_proof(self, input_data: torch.Tensor):
        """Generates a zero-knowledge proof for the model inference."""
        if not EZKL_AVAILABLE:
            # Mock proof for testing
            with torch.no_grad():
                output = self.model(input_data)
            mock_proof = {
                "proof": "mock_proof_data",
                "public_inputs": output.flatten().tolist()[
                    :3
                ],  # First 3 outputs as public
                "transcript_type": "EVM",
            }
            return mock_proof, output.numpy().tolist()

        # Prepare input data for ezkl
        input_array = (input_data.detach().numpy() * 2**15).astype(np.int64)
        data = dict(input_data=[input_array.flatten().tolist()])
        with open(self.input_json_path, "w") as f:
            json.dump(data, f)

        # Generate witness
        ezkl.gen_witness(
            self.input_json_path, self.compiled_model_path, self.witness_path
        )

        # Generate proof
        proof = ezkl.prove(
            self.witness_path,
            self.compiled_model_path,
            self.pk_path,
            self.proof_path,
            "single",
        )

        # Get the model output from the witness
        with open(self.witness_path, "r") as f:
            witness = json.load(f)
        output = witness["output_data"]

        return proof, output

    async def verify_proof(self, proof, input_data: torch.Tensor):
        """Verifies the zero-knowledge proof."""
        if not EZKL_AVAILABLE:
            # Mock verification
            return isinstance(proof, dict) and "proof" in proof

        # Prepare input data for ezkl (same as generate_proof)
        input_array = (input_data.detach().numpy() * 2**15).astype(np.int64)
        data = dict(input_data=[input_array.flatten().tolist()])
        with open(self.input_json_path, "w") as f:
            json.dump(data, f)

        # Verify proof
        verified = ezkl.verify(
            proof, self.settings_path, self.vk_path, srs_path=self.srs_path
        )

        return verified

    def cleanup(self):
        """Cleans up generated files."""
        for f in [
            self.onnx_path,
            self.compiled_model_path,
            self.pk_path,
            self.vk_path,
            self.settings_path,
            self.witness_path,
            self.proof_path,
            self.input_json_path,
            self.output_json_path,
        ]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass  # Ignore errors during cleanup


class PHAZEZKMLIntegration:
    """Enhanced zkML integration for PHAZE models."""

    def __init__(self):
        self.provers = {}
        self.model_registry = {}

    def register_model(self, name: str, model: nn.Module):
        """Register a model for zkML operations."""
        self.model_registry[name] = model
        self.provers[name] = ZKMLProverVerifier(model, name)

    def create_and_register_model(
        self,
        name: str,
        architecture: str = "simple",
        complexity: str = "medium",
        model_type: str = "full",
        **kwargs,
    ):
        """Create and register a new model."""
        complexity_map = {
            "minimal": ModelComplexity.MINIMAL,
            "light": ModelComplexity.LIGHT,
            "medium": ModelComplexity.MEDIUM,
            "heavy": ModelComplexity.HEAVY,
            "extreme": ModelComplexity.EXTREME,
        }

        complexity_enum = complexity_map.get(complexity, ModelComplexity.MEDIUM)

        if model_type == "full":
            model = PHAZEFullModel(architecture, complexity_enum, **kwargs)
        else:
            model = PHAZEModelFactory.create_early_exit_model(
                architecture, complexity_enum, **kwargs
            )

        self.register_model(name, model)
        return model

    async def setup_model(self, name: str, input_data: torch.Tensor):
        """Setup zkML for a registered model."""
        if name not in self.provers:
            raise ValueError(f"Model {name} not registered")

        await self.provers[name]._async_setup(input_data)

    async def prove_inference(self, name: str, input_data: torch.Tensor):
        """Generate proof for model inference."""
        if name not in self.provers:
            raise ValueError(f"Model {name} not registered")

        return await self.provers[name].generate_proof(input_data)

    async def verify_inference(self, name: str, proof: Any, input_data: torch.Tensor):
        """Verify proof for model inference."""
        if name not in self.provers:
            raise ValueError(f"Model {name} not registered")

        return await self.provers[name].verify_proof(proof, input_data)

    def cleanup_model(self, name: str):
        """Cleanup zkML artifacts for a model."""
        if name in self.provers:
            self.provers[name].cleanup()

    def cleanup_all(self):
        """Cleanup all zkML artifacts."""
        for prover in self.provers.values():
            prover.cleanup()

    def get_registered_models(self):
        """Get list of registered models."""
        return list(self.model_registry.keys())

    def get_model_info(self, name: str):
        """Get information about a registered model."""
        if name not in self.model_registry:
            raise ValueError(f"Model {name} not registered")

        model = self.model_registry[name]
        info = {
            "name": name,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "type": type(model).__name__,
        }

        if hasattr(model, "get_model_info"):
            info.update(model.get_model_info())

        return info


# Convenience functions for backward compatibility
def create_simple_full_model(input_size: int = 10, output_size: int = 5):
    """Create a simple full model."""
    return SimpleFullModel()


def create_zkml_prover_verifier(model: nn.Module, name: str):
    """Create a zkML prover-verifier for a model."""
    return ZKMLProverVerifier(model, name)
