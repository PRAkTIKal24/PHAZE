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
    # Try to get binding info, fall back to basic info if function doesn't exist
    try:
        _binding_info = rust_zkml_bindings.get_binding_info()
    except AttributeError:
        _binding_info = {"implementation": "real_rust_bindings", "version": "unknown"}
except ImportError:
    print("❌ rust_zkml_bindings not found. Using mock implementation for testing.")
    from . import mock_rust_zkml_bindings as rust_zkml_bindings

    _using_real_bindings = False
    try:
        _binding_info = rust_zkml_bindings.get_binding_info()
    except AttributeError:
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
    def get_binding_info() -> Dict[str, Any]:
        """Get information about the current binding implementation."""
        return _binding_info

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

        print(f"DEBUG: RustRiscZeroBackend.setup() called with params: {list(params.keys())}")
        
        # Store guest program path for later use if provided
        if "guest_program_path" in params:
            self._guest_program_path = params["guest_program_path"]
            print(f"DEBUG: Storing guest program path: {self._guest_program_path}")
        
        # Also try alternative path keys
        if "guest_elf_path" in params:
            self._guest_program_path = params["guest_elf_path"]
            print(f"DEBUG: Storing guest ELF path: {self._guest_program_path}")
        
        # Check if we have any path stored
        if not hasattr(self, '_guest_program_path'):
            print("DEBUG: No guest program path provided in setup params")

        result = self.risc_zero.setup(params)
        self.is_setup = True
        return result

    def prove(
        self, input_tensor: torch.Tensor, model_weights: Dict[str, any]
    ) -> Dict[str, Any]:
        """Generate a RISC Zero proof.

        Args:
            input_tensor: Input tensor for model inference
            model_weights: Model weights - can be state_dict (Dict[str, torch.Tensor]) or 
                          structured weights (Dict[str, List]) for multi-exit models

        Returns:
            Dictionary containing the proof and related information
        """
        if not self.is_setup:
            raise RuntimeError("Backend not setup. Call setup() first.")

        # Convert PyTorch tensors to flat float lists
        input_data = input_tensor.flatten().tolist()

        # Check if this is structured weights (multi-exit) or standard state_dict
        if isinstance(model_weights, dict) and any(
            key in model_weights for key in ["backbone_weights", "exit_weights", "backbone_bias", "exit_bias"]
        ):
            # This is structured weights format for multi-exit models
            # The guest program expects a ModelInput struct with this exact structure
            
            # Check if we're using real RISC Zero bindings or mock
            # Import here to avoid circular imports
            from . import rust_zkml_backend
            use_real_bindings = rust_zkml_backend.RustZKMLBackend.is_using_real_bindings()
            print(f"DEBUG: Using real RISC Zero bindings: {use_real_bindings}")
            
            if use_real_bindings:
                # Real implementation: create proper ModelInput structure
                model_input = {
                    "input_tensor": input_data,
                    "weights": {
                        "backbone_weights": model_weights["backbone_weights"],
                        "backbone_bias": model_weights["backbone_bias"], 
                        "exit_weights": model_weights["exit_weights"],
                        "exit_bias": model_weights["exit_bias"],
                        "exit_layer": model_weights["exit_layer"]
                    }
                }
                
                # Serialize the complete ModelInput for the guest program
                import json
                serialized_input = json.dumps(model_input)
                print(f"DEBUG: Attempting structured input: {len(serialized_input)} bytes")
                
                # Serialize weights separately for the backend
                serialized_weights = json.dumps(model_weights).encode('utf-8')
                print(f"DEBUG: Serialized weights to {len(serialized_weights)} bytes")
                
                # For real RISC Zero backend, we need to match the expected API signature
                try:
                    # Check if we have a stored guest program path from setup
                    if hasattr(self, '_guest_program_path') and self._guest_program_path:
                        print(f"DEBUG: Using stored guest program path: {self._guest_program_path}")
                        # Try to pass the ELF path directly to the prove method
                        if hasattr(self.risc_zero, 'prove_with_elf'):
                            print("DEBUG: Using prove_with_elf method")
                            proof = self.risc_zero.prove_with_elf(input_data, serialized_weights, self._guest_program_path)
                        else:
                            print("DEBUG: Setting working directory to rust_bindings and using standard prove method")
                            # Try changing working directory to rust_bindings root as suggested by error message
                            import os
                            from pathlib import Path
                            original_cwd = os.getcwd()
                            try:
                                # The error message suggests: "cd rust_bindings && ./build_guest.sh"
                                # So the backend expects to be run from rust_bindings directory
                                # Path: /Users/.../rust_bindings/guest_programs/.../docker/guest_multi_exit_minimal
                                # We need to find the rust_bindings ancestor directory
                                from pathlib import Path
                                current_path = Path(self._guest_program_path)
                                rust_bindings_dir = None
                                
                                # Walk up the path until we find rust_bindings
                                for parent in current_path.parents:
                                    if parent.name == 'rust_bindings':
                                        rust_bindings_dir = parent
                                        break
                                
                                if rust_bindings_dir and rust_bindings_dir.exists():
                                    print(f"DEBUG: Found rust_bindings directory: {rust_bindings_dir}")
                                    os.chdir(str(rust_bindings_dir))
                                    
                                    # Check if the legacy risc0_guest build exists (what build_guest.sh creates)
                                    legacy_elf_path = rust_bindings_dir / "risc0_guest" / "target" / "riscv32im-risc0-zkvm-elf" / "release" / "risc0_guest"
                                    if legacy_elf_path.exists():
                                        print(f"DEBUG: Found legacy ELF, using: {legacy_elf_path}")
                                        os.environ['RISC0_GUEST_PATH'] = str(legacy_elf_path)
                                        os.environ['RISC0_ELF_PATH'] = str(legacy_elf_path)
                                    else:
                                        print(f"DEBUG: No legacy ELF found at {legacy_elf_path}")
                                        # Try the dynamic guest program path
                                        os.environ['RISC0_GUEST_PATH'] = str(self._guest_program_path)
                                        os.environ['RISC0_ELF_PATH'] = str(self._guest_program_path)
                                    
                                    print(f"DEBUG: Set environment variables for ELF path")
                                else:
                                    print(f"DEBUG: Could not find rust_bindings directory in path: {self._guest_program_path}")
                                    
                                proof = self.risc_zero.prove(input_data, serialized_weights)
                            finally:
                                os.chdir(original_cwd)
                    elif hasattr(self.risc_zero, 'prove_structured'):
                        print("DEBUG: Using prove_structured method")
                        proof = self.risc_zero.prove_structured(serialized_input.encode('utf-8'))
                    else:
                        print("DEBUG: Using standard prove method with structured data")
                        # Real backend expects: prove(input_data, model_weights)
                        # But model_weights must be a Sequence, not dict
                        # Convert structured dict to JSON bytes for the backend
                        proof = self.risc_zero.prove(input_data, serialized_weights)
                except Exception as e:
                    print(f"DEBUG: Real backend failed: {e}, falling back to mock approach")
                    # If real backend fails, fall back to flattened approach
                    flattened_weights = self._flatten_structured_weights(model_weights)
                    proof = self.risc_zero.prove(input_data, flattened_weights)
            else:
                # Mock implementation: flatten everything for backward compatibility
                flattened_weights = self._flatten_structured_weights(model_weights)
                proof = self.risc_zero.prove(input_data, flattened_weights)
        else:
            # Standard state_dict format with PyTorch tensors - flatten for backward compatibility
            flattened_weights = []
            for weight in model_weights.values():
                if hasattr(weight, 'flatten'):
                    flattened_weights.extend(weight.flatten().tolist())
                else:
                    # Handle case where weight might already be a list
                    if isinstance(weight, (list, tuple)):
                        flattened_weights.extend(weight)
                    else:
                        flattened_weights.append(float(weight))

            proof = self.risc_zero.prove(input_data, flattened_weights)
            
        return {
            "proof_data": proof.proof_data,
            "public_inputs": proof.public_inputs,
            "framework": proof.framework,
            "verification_key_hash": proof.verification_key_hash,
        }

    def _flatten_structured_weights(self, model_weights: Dict[str, any]) -> List[float]:
        """Flatten structured weights for mock backend compatibility."""
        def recursive_flatten(item):
            """Recursively flatten any nested structure to a flat list of floats."""
            if hasattr(item, 'flatten'):
                # PyTorch tensor
                return item.flatten().tolist()
            elif isinstance(item, (list, tuple)):
                # Nested list/tuple, recursively flatten
                result = []
                for sub_item in item:
                    result.extend(recursive_flatten(sub_item))
                return result
            else:
                # Single value, convert to float
                return [float(item)]
        
        flattened_weights = []
        for key, weight_data in model_weights.items():
            if key == "exit_layer":
                # exit_layer is typically an integer, skip it for weight flattening in mock
                continue
            flattened_weights.extend(recursive_flatten(weight_data))
        
        return flattened_weights

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
