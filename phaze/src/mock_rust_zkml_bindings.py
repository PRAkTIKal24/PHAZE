"""
Mock implementation of rust_zkml_bindings for testing when Rust bindings are not available.
"""

import hashlib
import json
import random
import time
from typing import Any, Dict, List


class MockZKMLProof:
    """Mock ZKML proof object."""
    
    def __init__(self, proof_data: str, public_inputs: List[str], framework: str):
        self.proof_data = proof_data
        self.public_inputs = public_inputs
        self.framework = framework
        self.verification_key_hash = "mock_verification_key_hash"


class MockFiniteField:
    """Mock finite field implementation."""
    
    def __init__(self, modulus: str):
        self.modulus = int(modulus)
    
    def add(self, a: str, b: str) -> str:
        return str((int(a) + int(b)) % self.modulus)
    
    def multiply(self, a: str, b: str) -> str:
        return str((int(a) * int(b)) % self.modulus)
    
    def power(self, base: str, exp: int) -> str:
        return str(pow(int(base), exp, self.modulus))


class MockPolynomial:
    """Mock polynomial implementation."""
    
    def __init__(self, coefficients: List[str], field: MockFiniteField):
        self.coefficients = [int(c) for c in coefficients]
        self.field = field
    
    def evaluate(self, x: str) -> str:
        x_val = int(x)
        result = 0
        for i, coeff in enumerate(self.coefficients):
            result = (result + coeff * pow(x_val, i, self.field.modulus)) % self.field.modulus
        return str(result)


class MockGroth16:
    """Mock Groth16 backend."""
    
    def __init__(self):
        self.setup_done = False
        self.circuit_size = 0
    
    def setup(self, circuit_size: int) -> str:
        self.circuit_size = circuit_size
        self.setup_done = True
        return f"Groth16 setup completed for circuit size {circuit_size}"
    
    def prove(self, witness: List[str]) -> MockZKMLProof:
        if not self.setup_done:
            raise RuntimeError("Setup not completed")
        
        # Simulate proof generation
        time.sleep(0.01)  # Small delay to simulate computation
        proof_data = hashlib.sha256(f"groth16_proof_{witness}".encode()).hexdigest()
        return MockZKMLProof(proof_data, witness, "groth16")
    
    def verify(self, proof: MockZKMLProof) -> bool:
        # Mock verification always returns True for valid structure
        return proof.framework == "groth16" and len(proof.proof_data) == 64
    
    def get_setup_info(self) -> Dict[str, str]:
        return {
            "framework": "groth16",
            "circuit_size": str(self.circuit_size),
            "curve": "BN254",
            "setup_completed": str(self.setup_done)
        }


class MockPlonky:
    """Mock Plonky backend."""
    
    def __init__(self):
        self.setup_done = False
        self.degree_bound = 0
    
    def setup(self, degree_bound: int) -> str:
        self.degree_bound = degree_bound
        self.setup_done = True
        return f"Plonky setup completed with degree bound {degree_bound}"
    
    def prove(self, witness: List[str]) -> MockZKMLProof:
        if not self.setup_done:
            raise RuntimeError("Setup not completed")
        
        time.sleep(0.02)  # Small delay to simulate computation
        proof_data = hashlib.sha256(f"plonky_proof_{witness}".encode()).hexdigest()
        return MockZKMLProof(proof_data, witness, "plonky")
    
    def verify(self, proof: MockZKMLProof) -> bool:
        return proof.framework == "plonky" and len(proof.proof_data) == 64
    
    def get_field_info(self) -> Dict[str, str]:
        return {
            "framework": "plonky",
            "field": "GoldilocksField",
            "degree_bound": str(self.degree_bound),
            "setup_completed": str(self.setup_done)
        }


class MockHalo:
    """Mock Halo backend."""
    
    def __init__(self):
        self.setup_done = False
        self.circuit_depth = 0
    
    def setup(self, circuit_depth: int) -> str:
        self.circuit_depth = circuit_depth
        self.setup_done = True
        return f"Halo setup completed with circuit depth {circuit_depth}"
    
    def prove(self, witness: List[str]) -> MockZKMLProof:
        if not self.setup_done:
            raise RuntimeError("Setup not completed")
        
        time.sleep(0.015)  # Small delay to simulate computation
        proof_data = hashlib.sha256(f"halo_proof_{witness}".encode()).hexdigest()
        return MockZKMLProof(proof_data, witness, "halo")
    
    def verify(self, proof: MockZKMLProof) -> bool:
        return proof.framework == "halo" and len(proof.proof_data) == 64
    
    def get_curve_info(self) -> Dict[str, str]:
        return {
            "framework": "halo",
            "curve": "Pasta",
            "circuit_depth": str(self.circuit_depth),
            "setup_completed": str(self.setup_done)
        }


class MockRiscZeroBackend:
    """Mock RISC Zero backend."""
    
    def __init__(self):
        self.setup_done = False
        self.config = {}
    
    def setup(self, params: Dict[str, str] = None) -> str:
        if params is None:
            params = {}
        self.config = params
        self.setup_done = True
        return f"RISC Zero setup completed with params: {json.dumps(params)}"
    
    def prove(self, input_data: List[float], model_weights: List[float]) -> MockZKMLProof:
        if not self.setup_done:
            raise RuntimeError("Setup not completed")
        
        time.sleep(0.03)  # Small delay to simulate computation
        proof_data = hashlib.sha256(f"risc_zero_proof_{input_data}_{model_weights}".encode()).hexdigest()
        # For RISC Zero, public inputs might include the computation result
        public_inputs = [str(sum(input_data[:5]))]  # Mock computation result
        return MockZKMLProof(proof_data, public_inputs, "risc_zero")
    
    def verify(self, proof: MockZKMLProof, expected_outputs: List[float] = None) -> bool:
        # Mock verification with optional output checking
        if proof.framework != "risc_zero":
            return False
        if expected_outputs and proof.public_inputs:
            # Simple check that public inputs are reasonable
            try:
                computed = float(proof.public_inputs[0])
                expected = sum(expected_outputs[:1]) if expected_outputs else 0
                # Allow some tolerance in mock verification
                return abs(computed - expected) < 10.0
            except (ValueError, IndexError):
                return False
        return len(proof.proof_data) == 64
    
    def get_config(self) -> Dict[str, str]:
        return {
            "framework": "risc_zero",
            "config": json.dumps(self.config),
            "setup_completed": str(self.setup_done)
        }


# Mock module-level functions
def sha256_hash(data: bytes) -> str:
    """Mock SHA256 hash function."""
    return hashlib.sha256(data).hexdigest()


def keccak256_hash(data: bytes) -> str:
    """Mock Keccak256 hash function."""
    # For mock purposes, we'll use SHA3-256 which is similar
    return hashlib.sha3_256(data).hexdigest()


def FiniteField(modulus: str) -> MockFiniteField:
    """Create a mock finite field."""
    return MockFiniteField(modulus)


def Polynomial(coefficients: List[str], field: MockFiniteField) -> MockPolynomial:
    """Create a mock polynomial."""
    return MockPolynomial(coefficients, field)


def generate_random_field_element(field_size: str) -> str:
    """Generate a mock random field element."""
    size = int(field_size)
    return str(random.randint(0, size - 1))


def compute_merkle_root(leaves: List[str]) -> str:
    """Mock Merkle root computation."""
    # Simple mock: hash all leaves together
    combined = "".join(leaves)
    return hashlib.sha256(combined.encode()).hexdigest()


def benchmark_field_operations(field_size: str, num_operations: int = 1000) -> Dict[str, float]:
    """Mock field operations benchmark."""
    # Return mock timing data
    base_time = 0.001  # 1ms base time
    size_factor = len(field_size) / 10.0  # Larger fields take longer
    
    return {
        "addition_avg_time": base_time * size_factor,
        "multiplication_avg_time": base_time * size_factor * 2,
        "exponentiation_avg_time": base_time * size_factor * 10,
        "total_operations": num_operations,
        "field_size": field_size
    }


def ZKMLProof(proof_data: str, public_inputs: List[str], framework: str) -> MockZKMLProof:
    """Create a mock ZKML proof."""
    return MockZKMLProof(proof_data, public_inputs, framework)


def MockGroth16() -> MockGroth16:
    """Create a mock Groth16 instance."""
    return MockGroth16()


def MockPlonky() -> MockPlonky:
    """Create a mock Plonky instance."""
    return MockPlonky()


def MockHalo() -> MockHalo:
    """Create a mock Halo instance."""
    return MockHalo()


def RiscZeroBackend() -> MockRiscZeroBackend:
    """Create a mock RISC Zero backend."""
    return MockRiscZeroBackend()