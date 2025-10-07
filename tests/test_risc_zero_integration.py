"""
Integration tests for the RISC Zero zkML workflow.
"""

import pytest
import torch

from phaze.src.model_architectures import PHAZEModelFactory
from phaze.src.rust_zkml_backend import RustRiscZeroBackend
from phaze.src.zkml_backends import create_backend
from phaze.src.zkml_framework_interface import ZKMLBackendInterface, ZKMLFramework
from phaze.src.zkml_integration import PHAZEZKMLIntegration


@pytest.fixture
def simple_model_and_input():
    """Create a simple early exit model and sample input."""
    model = PHAZEModelFactory.create_early_exit_model("simple", complexity="light")
    input_data = torch.randn(1, 10)  # Batch size 1, 10 features
    return model, input_data


@pytest.mark.skip(
    reason="Full RISC Zero integration requires actual RISC Zero dependency"
)
@pytest.mark.asyncio
async def test_risc_zero_end_to_end(simple_model_and_input):
    """Test the end-to-end RISC Zero workflow."""
    model, input_data = simple_model_and_input

    # 1. Create RISC Zero backend
    backend = RustRiscZeroBackend()

    # 2. Set up the backend
    params = {
        "model_type": "simple_early_exit",
        "input_size": "10",
        "output_size": "5",
    }
    setup_result = backend.setup(params)
    assert "RISC Zero setup completed" in setup_result

    # 3. Run PyTorch inference to get ground truth
    with torch.no_grad():
        expected_output = model(input_data)

    # 4. Prove using RISC Zero backend
    proof = backend.prove(input_data, model.state_dict())

    # 5. Verify proof structure
    assert "proof_data" in proof
    assert "public_inputs" in proof
    assert "framework" in proof
    assert proof["framework"] == "RISC0"

    # 6. Verify the proof
    verification_result = backend.verify(proof, expected_output)
    assert verification_result is True


@pytest.mark.asyncio
async def test_zkml_integration_with_risc_zero():
    """Test that RISC Zero integrates with PHAZEZKMLIntegration."""
    # 1. Create and register a model
    integration = PHAZEZKMLIntegration()
    model = integration.create_and_register_model(
        "risc_zero_test_model",
        architecture="simple",
        complexity="light",
        model_type="early_exit",
    )

    # 2. Create sample input
    input_data = torch.randn(1, 10)

    # 3. Create RISC Zero backend through the backend factory
    backend = create_backend(ZKMLFramework.RISC_ZERO, model, "risc_zero_test")
    assert isinstance(backend, ZKMLBackendInterface)
    assert backend.framework == ZKMLFramework.RISC_ZERO

    # 4. Setup the backend
    await backend.setup(input_data)
    assert backend.is_setup

    # 5. Generate a proof (this uses the mock implementation)
    proof, output = await backend.generate_proof(input_data)
    assert proof is not None

    # 6. Verify the proof
    is_valid = await backend.verify_proof(proof, input_data)
    assert is_valid

    # 7. Cleanup
    backend.cleanup()


def test_rust_risc_zero_backend_integration():
    """Test integration between Python and Rust for RISC Zero."""
    # 1. Create the RustRiscZeroBackend
    backend = RustRiscZeroBackend()

    # 2. Set up the backend
    setup_result = backend.setup()
    assert setup_result is not None
    assert backend.is_setup

    # 3. Create dummy inputs
    input_tensor = torch.randn(1, 10)
    model_weights = {
        "fc1.weight": torch.randn(50, 10),
        "fc1.bias": torch.randn(50),
        "fc2.weight": torch.randn(5, 50),
        "fc2.bias": torch.randn(5),
    }

    # 4. Generate a proof
    proof = backend.prove(input_tensor, model_weights)
    assert "proof_data" in proof
    assert "framework" in proof
    assert proof["framework"] == "RISC0"

    # 5. Verify the proof
    expected_outputs = torch.randn(1, 5)  # Mock expected outputs
    verification_result = backend.verify(proof, expected_outputs)
    assert verification_result is True

    # 6. Get and check configuration
    config = backend.get_config()
    assert "proof_system" in config
    assert config["proof_system"] == "STARK"
