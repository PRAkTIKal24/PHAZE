"""
Concrete implementations of zkML backends for different frameworks.
"""

import asyncio
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .zkml_framework_interface import ZKMLBackendInterface, ZKMLFramework

try:
    import ezkl

    EZKL_AVAILABLE = True
except ImportError:
    EZKL_AVAILABLE = False


class EZKLBackend(ZKMLBackendInterface):
    """EZKL framework backend implementation."""

    def __init__(self, model: nn.Module, name: str):
        super().__init__(ZKMLFramework.EZKL, model, name)

        if not EZKL_AVAILABLE:
            raise ImportError(
                "ezkl is not installed. Please install it to use this backend."
            )

        # File paths for EZKL artifacts
        self.onnx_path = f"/tmp/{self.name}.onnx"
        self.compiled_model_path = f"/tmp/{self.name}.compiled"
        self.pk_path = f"/tmp/{self.name}.pk"
        self.vk_path = f"/tmp/{self.name}.vk"
        self.settings_path = f"/tmp/{self.name}_settings.json"
        self.witness_path = f"/tmp/{self.name}.witness.json"
        self.proof_path = f"/tmp/{self.name}.proof"
        self.input_json_path = f"/tmp/{self.name}_input.json"
        self.output_json_path = f"/tmp/{self.name}_output.json"
        self.srs_path = os.path.join(
            os.path.expanduser("~"), ".ezkl", "srs", "kzg17.srs"
        )

    def _export_to_onnx(self, input_data: torch.Tensor):
        """Export the PyTorch model to ONNX format."""
        # Simple ONNX export for EZKL compatibility
        self.model.eval()
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                input_data,
                self.onnx_path,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                export_params=True,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                verbose=False
            )

    async def setup(self, input_data: torch.Tensor, **kwargs) -> None:
        """Setup the EZKL system."""
        self._export_to_onnx(input_data)

        # Generate settings
        ezkl.gen_settings(self.onnx_path, self.settings_path)

        # Download SRS if not exists
        if not os.path.exists(self.srs_path):
            print(f"Downloading SRS to {self.srs_path}...")
            os.makedirs(os.path.dirname(self.srs_path), exist_ok=True)
            await ezkl.get_srs(srs_path=self.srs_path, settings_path=self.settings_path)

        # Compile the model
        ezkl.compile_circuit(
            self.onnx_path, self.compiled_model_path, self.settings_path
        )

        # Generate proving and verification keys
        ezkl.setup(
            self.compiled_model_path, self.vk_path, self.pk_path, srs_path=self.srs_path
        )

        self.is_setup = True

    async def generate_proof(
        self, input_data: torch.Tensor, **kwargs
    ) -> Tuple[Any, Any]:
        """Generate a zero-knowledge proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

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
        output = witness["outputs"]

        return proof, output

    async def verify_proof(
        self, proof: Any, input_data: torch.Tensor, **kwargs
    ) -> bool:
        """Verify a zero-knowledge proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        # Prepare input data for ezkl
        input_array = (input_data.detach().numpy() * 2**15).astype(np.int64)
        data = dict(input_data=[input_array.flatten().tolist()])

        with open(self.input_json_path, "w") as f:
            json.dump(data, f)

        # Verify proof
        verified = ezkl.verify(
            self.proof_path, self.settings_path, self.vk_path, srs_path=self.srs_path
        )

        return verified

    def cleanup(self) -> None:
        """Clean up generated files."""
        files_to_remove = [
            self.onnx_path,
            self.compiled_model_path,
            self.pk_path,
            self.vk_path,
            self.settings_path,
            self.witness_path,
            self.proof_path,
            self.input_json_path,
            self.output_json_path,
        ]

        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)

    def get_framework_info(self) -> Dict[str, Any]:
        """Get EZKL framework information."""
        try:
            import ezkl

            return {
                "framework": "EZKL",
                "version": getattr(ezkl, "__version__", "unknown"),
                "backend": "KZG",
                "curve": "BN254",
            }
        except ImportError:
            return {"framework": "EZKL", "status": "not_available"}


class MockZKMLBackend(ZKMLBackendInterface):
    """Mock backend for testing and placeholder implementations."""

    def __init__(self, framework: ZKMLFramework, model: nn.Module, name: str):
        super().__init__(framework, model, name)
        self.mock_proof = {"proof": "mock_proof_data", "public_inputs": ["mock_input"]}

    async def setup(self, input_data: torch.Tensor, **kwargs) -> None:
        """Mock setup - just simulate some delay."""
        await asyncio.sleep(0.01)  # Simulate setup time
        self.is_setup = True

    async def generate_proof(
        self, input_data: torch.Tensor, **kwargs
    ) -> Tuple[Any, Any]:
        """Generate a mock proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        await asyncio.sleep(0.05)  # Simulate proving time

        # Mock output based on model forward pass
        with torch.no_grad():
            output = self.model(input_data)

        return self.mock_proof, output.numpy().tolist()

    async def verify_proof(
        self, proof: Any, input_data: torch.Tensor, **kwargs
    ) -> bool:
        """Verify a mock proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        await asyncio.sleep(0.01)  # Simulate verification time

        # Mock verification - always return True for valid mock proofs
        return proof == self.mock_proof

    def cleanup(self) -> None:
        """Mock cleanup - nothing to clean."""
        pass

    def get_framework_info(self) -> Dict[str, Any]:
        """Get mock framework information."""
        return {
            "framework": self.framework.value.upper(),
            "version": "mock_v1.0",
            "status": "placeholder_implementation",
        }


class ZKCNNBackend(MockZKMLBackend):
    """ZKCNN framework backend (placeholder implementation)."""

    def __init__(self, model: nn.Module, name: str):
        super().__init__(ZKMLFramework.ZKCNN, model, name)

    def get_framework_info(self) -> Dict[str, Any]:
        """Get ZKCNN framework information."""
        return {
            "framework": "ZKCNN",
            "version": "placeholder_v1.0",
            "backend": "CNN-optimized circuits",
            "status": "placeholder_implementation",
        }


class Groth16Backend(MockZKMLBackend):
    """Groth16 framework backend (placeholder implementation)."""

    def __init__(self, model: nn.Module, name: str):
        super().__init__(ZKMLFramework.GROTH16, model, name)

    def get_framework_info(self) -> Dict[str, Any]:
        """Get Groth16 framework information."""
        return {
            "framework": "Groth16",
            "version": "placeholder_v1.0",
            "backend": "Groth16 zk-SNARKs",
            "curve": "BN254",
            "status": "placeholder_implementation",
        }


class HaloBackend(MockZKMLBackend):
    """Halo framework backend (placeholder implementation)."""

    def __init__(self, model: nn.Module, name: str):
        super().__init__(ZKMLFramework.HALO, model, name)

    def get_framework_info(self) -> Dict[str, Any]:
        """Get Halo framework information."""
        return {
            "framework": "Halo",
            "version": "placeholder_v1.0",
            "backend": "Halo recursive proofs",
            "curve": "Pasta curves",
            "status": "placeholder_implementation",
        }


class PlonkyBackend(MockZKMLBackend):
    """Plonky framework backend (placeholder implementation)."""

    def __init__(self, model: nn.Module, name: str):
        super().__init__(ZKMLFramework.PLONKY, model, name)

    def get_framework_info(self) -> Dict[str, Any]:
        """Get Plonky framework information."""
        return {
            "framework": "Plonky",
            "version": "placeholder_v1.0",
            "backend": "PLONK with FRI",
            "field": "Goldilocks field",
            "status": "placeholder_implementation",
        }


class RiscZeroBackend(ZKMLBackendInterface):
    """RISC Zero framework backend implementation using Rust backend."""

    def __init__(self, model: nn.Module, name: str):
        super().__init__(ZKMLFramework.RISC_ZERO, model, name)
        from .rust_zkml_backend import RustRiscZeroBackend

        self.rust_backend = RustRiscZeroBackend()

    async def setup(self, input_data: torch.Tensor, **kwargs) -> None:
        """Setup the RISC Zero system."""
        # Extract model parameters for setup
        params = {
            "model_type": "neural_network",
            "input_size": str(input_data.numel()),
            "output_size": str(
                list(self.model.parameters())[-1].shape[0]
                if list(self.model.parameters())
                else "10"
            ),
        }

        # Setup the Rust backend
        self.rust_backend.setup(params)
        self.is_setup = True

    async def generate_proof(
        self, input_data: torch.Tensor, **kwargs
    ) -> Tuple[Any, Any]:
        """Generate a zero-knowledge proof using RISC Zero."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        # Get model weights
        model_weights = self.model.state_dict()

        # Generate proof using Rust backend
        proof_dict = self.rust_backend.prove(input_data, model_weights)

        # Run model to get output
        with torch.no_grad():
            output = self.model(input_data)

        return proof_dict, output.numpy().tolist()

    async def verify_proof(
        self, proof: Any, input_data: torch.Tensor, **kwargs
    ) -> bool:
        """Verify a zero-knowledge proof using RISC Zero."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        # Get expected outputs
        with torch.no_grad():
            expected_outputs = self.model(input_data)

        # Verify using Rust backend
        return self.rust_backend.verify(proof, expected_outputs)

    def cleanup(self) -> None:
        """Clean up RISC Zero backend resources."""
        # The Rust backend handles its own cleanup
        pass

    def get_framework_info(self) -> Dict[str, Any]:
        """Get RISC Zero framework information."""
        config = self.rust_backend.get_config()
        return {
            "framework": "RISC Zero",
            "version": "1.0.0",
            "backend": "RISC-V zkVM",
            "proof_system": "STARK",
            "status": "active",
            "config": config,
        }


class StarkBackend(MockZKMLBackend):
    """STARK framework backend (placeholder implementation)."""

    def __init__(self, model: nn.Module, name: str):
        super().__init__(ZKMLFramework.STARK, model, name)

    def get_framework_info(self) -> Dict[str, Any]:
        """Get STARK framework information."""
        return {
            "framework": "STARK",
            "version": "placeholder_v1.0",
            "backend": "STARK proofs",
            "field": "Prime field",
            "status": "placeholder_implementation",
        }


def create_backend(
    framework: ZKMLFramework, model: nn.Module, name: str
) -> ZKMLBackendInterface:
    """Factory function to create zkML backends."""
    backend_map = {
        ZKMLFramework.EZKL: EZKLBackend,
        ZKMLFramework.ZKCNN: ZKCNNBackend,
        ZKMLFramework.GROTH16: Groth16Backend,
        ZKMLFramework.HALO: HaloBackend,
        ZKMLFramework.PLONKY: PlonkyBackend,
        ZKMLFramework.RISC_ZERO: RiscZeroBackend,
        ZKMLFramework.STARK: StarkBackend,
    }

    if framework not in backend_map:
        raise ValueError(f"Unsupported framework: {framework}")

    backend_class = backend_map[framework]

    # Special case for EZKL which has different constructor signature
    if framework == ZKMLFramework.EZKL:
        return backend_class(model, name)
    else:
        return backend_class(model, name)
