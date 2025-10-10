"""
Enhanced Rust-based zkML backend with expanded functionality.
"""

import json  # noqa: F401
from typing import Any, Dict, List, Optional  # noqa: F401

import numpy as np  # noqa: F401
import torch

try:
    import rust_zkml_bindings
    _using_real_bindings = True
    _binding_info = rust_zkml_bindings.get_binding_info()
    print(f"✅ REAL Rust bindings loaded: {_binding_info}")
except ImportError:
    print("❌ rust_zkml_bindings not found. Using mock implementation for testing.")
    from . import mock_rust_zkml_bindings as rust_zkml_bindings
    _using_real_bindings = False
    _binding_info = {"implementation": "mock", "version": "0.1.0"}


class RustZKMLBackend:
    """Enhanced Rust-based backend for zkML operations."""

    def __init__(self):
        self.field_cache = {}
        self.proof_cache = {}

    @staticmethod
    def is_using_real_bindings() -> bool:
        """Check if we're using real Rust bindings or mock implementation."""
        return _using_real_bindings

    @staticmethod
    def get_binding_info() -> dict:
        """Get information about the current bindings."""
        return _binding_info.copy()

    def sha256_hash(self, data: bytes) -> str:
        """Compute SHA256 hash using Rust implementation."""
        return rust_zkml_bindings.sha256_hash(data)

    def keccak256_hash(self, data: bytes) -> str:
        """Compute Keccak256 hash using Rust implementation."""
        return rust_zkml_bindings.keccak256_hash(data)

    def create_finite_field(self, modulus: str) -> Any:
        """Create a finite field with the given modulus."""
        if modulus not in self.field_cache:
            self.field_cache[modulus] = rust_zkml_bindings.FiniteField(modulus)
        return self.field_cache[modulus]

    def field_add(self, field_modulus: str, a: str, b: str) -> str:
        """Add two field elements."""
        field = self.create_finite_field(field_modulus)
        return field.add(a, b)

    def field_multiply(self, field_modulus: str, a: str, b: str) -> str:
        """Multiply two field elements."""
        field = self.create_finite_field(field_modulus)
        return field.multiply(a, b)

    def field_power(self, field_modulus: str, base: str, exp: int) -> str:
        """Compute field exponentiation."""
        field = self.create_finite_field(field_modulus)
        return field.power(base, exp)

    def create_polynomial(self, coefficients: List[str], field_modulus: str) -> Any:
        """Create a polynomial over a finite field."""
        field = self.create_finite_field(field_modulus)
        return rust_zkml_bindings.Polynomial(coefficients, field)

    def evaluate_polynomial(
        self, coefficients: List[str], field_modulus: str, x: str
    ) -> str:
        """Evaluate a polynomial at a given point."""
        poly = self.create_polynomial(coefficients, field_modulus)
        return poly.evaluate(x)

    def generate_random_field_element(self, field_size: str) -> str:
        """Generate a random field element."""
        return rust_zkml_bindings.generate_random_field_element(field_size)

    def compute_merkle_root(self, leaves: List[str]) -> str:
        """Compute Merkle tree root from leaves."""
        return rust_zkml_bindings.compute_merkle_root(leaves)

    def benchmark_field_operations(
        self, field_size: str, num_operations: int = 1000
    ) -> Dict[str, float]:
        """Benchmark finite field operations."""
        return rust_zkml_bindings.benchmark_field_operations(field_size, num_operations)


class RustGroth16Backend:
    """Rust-based Groth16 zkML backend."""

    def __init__(self):
        self.groth16 = rust_zkml_bindings.MockGroth16()
        self.is_setup = False

    def setup(self, circuit_size: int) -> str:
        """Setup Groth16 proving and verification keys."""
        result = self.groth16.setup(circuit_size)
        self.is_setup = True
        return result

    def prove(self, witness: List[str]) -> Dict[str, Any]:
        """Generate a Groth16 proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        proof = self.groth16.prove(witness)
        return {
            "proof_data": proof.proof_data,
            "public_inputs": proof.public_inputs,
            "framework": proof.framework,
            "verification_key_hash": proof.verification_key_hash,
        }

    def verify(self, proof_dict: Dict[str, Any]) -> bool:
        """Verify a Groth16 proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        proof = rust_zkml_bindings.ZKMLProof(
            proof_dict["proof_data"],
            proof_dict["public_inputs"],
            proof_dict["framework"],
        )
        return self.groth16.verify(proof)

    def get_setup_info(self) -> Dict[str, str]:
        """Get setup information."""
        return self.groth16.get_setup_info()


class RustPlonkyBackend:
    """Rust-based Plonky zkML backend."""

    def __init__(self):
        self.plonky = rust_zkml_bindings.MockPlonky()
        self.is_setup = False

    def setup(self, degree_bound: int = 1024) -> str:
        """Setup Plonky parameters."""
        result = self.plonky.setup(degree_bound)
        self.is_setup = True
        return result

    def prove(self, witness: List[str]) -> Dict[str, Any]:
        """Generate a Plonky proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        proof = self.plonky.prove(witness)
        return {
            "proof_data": proof.proof_data,
            "public_inputs": proof.public_inputs,
            "framework": proof.framework,
            "verification_key_hash": proof.verification_key_hash,
        }

    def verify(self, proof_dict: Dict[str, Any]) -> bool:
        """Verify a Plonky proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        proof = rust_zkml_bindings.ZKMLProof(
            proof_dict["proof_data"],
            proof_dict["public_inputs"],
            proof_dict["framework"],
        )
        return self.plonky.verify(proof)

    def get_field_info(self) -> Dict[str, str]:
        """Get field information."""
        return self.plonky.get_field_info()


class RustHaloBackend:
    """Rust-based Halo zkML backend."""

    def __init__(self):
        self.halo = rust_zkml_bindings.MockHalo()
        self.is_setup = False

    def setup(self, circuit_depth: int = 10) -> str:
        """Setup Halo parameters."""
        result = self.halo.setup(circuit_depth)
        self.is_setup = True
        return result

    def prove(self, witness: List[str]) -> Dict[str, Any]:
        """Generate a Halo proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        proof = self.halo.prove(witness)
        return {
            "proof_data": proof.proof_data,
            "public_inputs": proof.public_inputs,
            "framework": proof.framework,
            "verification_key_hash": proof.verification_key_hash,
        }

    def verify(self, proof_dict: Dict[str, Any]) -> bool:
        """Verify a Halo proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        proof = rust_zkml_bindings.ZKMLProof(
            proof_dict["proof_data"],
            proof_dict["public_inputs"],
            proof_dict["framework"],
        )
        return self.halo.verify(proof)

    def get_curve_info(self) -> Dict[str, str]:
        """Get curve information."""
        return self.halo.get_curve_info()


class RustRiscZeroBackend:
    """Rust-based RISC Zero zkML backend."""

    def __init__(self):
        self.risc_zero = rust_zkml_bindings.RiscZeroBackend()
        self.is_setup = False

    def setup(self, params: Dict[str, str] = None) -> str:
        """Setup RISC Zero parameters."""
        if params is None:
            params = {
                "model_type": "simple",
                "input_size": "10",
                "output_size": "5",
            }

        result = self.risc_zero.setup(params)
        self.is_setup = True
        return result

    def prove(
        self, input_tensor: torch.Tensor, model_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Generate a RISC Zero proof.

        Args:
            input_tensor: Input tensor for model inference
            model_weights: Model weights as a dictionary of tensors (state_dict format)

        Returns:
            Dictionary containing the proof and related information
        """
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        # Convert PyTorch tensors to flat float lists
        input_data = input_tensor.flatten().tolist()

        # Convert model weights to a flat list
        # In a real implementation, this would need to match
        # the expected format in the guest program
        flattened_weights = []
        for weight in model_weights.values():
            flattened_weights.extend(weight.flatten().tolist())

        proof = self.risc_zero.prove(input_data, flattened_weights)
        return {
            "proof_data": proof.proof_data,
            "public_inputs": proof.public_inputs,
            "framework": proof.framework,
            "verification_key_hash": proof.verification_key_hash,
        }

    def verify(
        self, proof_dict: Dict[str, Any], expected_outputs: torch.Tensor = None
    ) -> bool:
        """Verify a RISC Zero proof."""
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        proof = rust_zkml_bindings.ZKMLProof(
            proof_dict["proof_data"],
            proof_dict["public_inputs"],
            proof_dict["framework"],
        )

        # Convert expected outputs to a flat float list, if provided
        expected_outputs_list = []
        if expected_outputs is not None:
            expected_outputs_list = expected_outputs.flatten().tolist()

        return self.risc_zero.verify(proof, expected_outputs_list)

    def get_config(self) -> Dict[str, str]:
        """Get RISC Zero configuration."""
        return self.risc_zero.get_config()


class RustZKMLFrameworkManager:
    """Manager for different Rust-based zkML frameworks."""

    def __init__(self):
        self.backends = {
            "groth16": RustGroth16Backend(),
            "plonky": RustPlonkyBackend(),
            "halo": RustHaloBackend(),
            "risc_zero": RustRiscZeroBackend(),
        }
        self.base_backend = RustZKMLBackend()

    def get_backend(self, framework: str):
        """Get a specific zkML framework backend."""
        if framework not in self.backends:
            raise ValueError(f"Unsupported framework: {framework}")
        return self.backends[framework]

    def list_frameworks(self) -> List[str]:
        """List available frameworks."""
        return list(self.backends.keys())

    def benchmark_all_frameworks(
        self, witness: List[str], circuit_params: Dict[str, int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark all available frameworks."""
        if circuit_params is None:
            circuit_params = {
                "groth16": {"circuit_size": 1000},
                "plonky": {"degree_bound": 1024},
                "halo": {"circuit_depth": 10},
                "risc_zero": {},  # No specific parameters for RISC Zero
            }

        results = {}

        for framework_name, backend in self.backends.items():
            try:
                # Setup
                if framework_name == "groth16":
                    setup_result = backend.setup(
                        circuit_params[framework_name]["circuit_size"]
                    )
                elif framework_name == "plonky":
                    setup_result = backend.setup(
                        circuit_params[framework_name]["degree_bound"]
                    )
                elif framework_name == "halo":
                    setup_result = backend.setup(
                        circuit_params[framework_name]["circuit_depth"]
                    )
                elif framework_name == "risc_zero":
                    setup_result = backend.setup({"witness_size": str(len(witness))})

                # Prove
                if framework_name == "risc_zero":
                    # For RISC Zero, we need to convert the witness format
                    # In a real implementation, this would be proper tensor input
                    input_tensor = torch.tensor(
                        [float(w) for w in witness[:10]]
                    ).reshape(1, -1)
                    model_weights = {
                        "weights": torch.ones(10, 5)  # Dummy weights
                    }
                    proof = backend.prove(input_tensor, model_weights)
                else:
                    proof = backend.prove(witness)

                # Verify
                is_valid = backend.verify(proof)

                results[framework_name] = {
                    "setup_result": setup_result,
                    "proof": proof,
                    "verification_result": is_valid,
                    "success": True,
                }

            except Exception as e:
                results[framework_name] = {"error": str(e), "success": False}

        return results

    def get_framework_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get a comparison of framework capabilities."""
        comparison = {}

        for framework_name, backend in self.backends.items():
            try:
                if framework_name == "groth16":
                    backend.setup(100)  # Small circuit for comparison
                    info = backend.get_setup_info()
                elif framework_name == "plonky":
                    backend.setup(256)
                    info = backend.get_field_info()
                elif framework_name == "halo":
                    backend.setup(5)
                    info = backend.get_curve_info()
                elif framework_name == "risc_zero":
                    backend.setup()
                    info = backend.get_config()

                comparison[framework_name] = {"info": info, "available": True}

            except Exception as e:
                comparison[framework_name] = {"error": str(e), "available": False}

        return comparison
