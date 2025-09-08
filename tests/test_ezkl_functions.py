import pytest
import torch
import os
from phaze.zkml_integration import ZKMLProverVerifier, SimpleFullModel

# Fixture for a dummy model and input
@pytest.fixture
def dummy_model_input():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    input_data = torch.randn(1, 10)
    return model, input_data

@pytest.mark.asyncio
async def test_ezkl_setup_and_prove(dummy_model_input):
    model, input_data = dummy_model_input
    pv = ZKMLProverVerifier(model, "test_model")
    
    # Ensure cleanup is called even if tests fail
    try:
        await pv.setup(input_data)
        proof, output_data = await pv.generate_proof(input_data)
        assert proof is not None
        assert isinstance(proof, dict)
        assert "proof" in proof
        assert "public_inputs" in proof
        assert "transcript_type" in proof
        assert output_data is not None

        # Verify the proof
        is_valid = await pv.verify_proof(proof, input_data)
        assert is_valid
    finally:
        pv.cleanup()

@pytest.mark.asyncio
async def test_ezkl_proof_verification_failure(dummy_model_input):
    model, input_data = dummy_model_input
    pv = ZKMLProverVerifier(model, "test_model")
    
    try:
        await pv.setup(input_data)
        proof, _ = await pv.generate_proof(input_data)
        
        # Tamper with the proof to make verification fail
        proof["public_inputs"] = [str(float(proof["public_inputs"][0]) + 1.0)]

        is_valid = await pv.verify_proof(proof, input_data)
        assert not is_valid
    finally:
        pv.cleanup()

@pytest.mark.asyncio
async def test_ezkl_cleanup(dummy_model_input):
    model, input_data = dummy_model_input
    pv = ZKMLProverVerifier(model, "test_model")
    await pv.setup(input_data)
    pv.cleanup()
    # Check if files are removed (this is a heuristic, actual check depends on ezkl internals)
    assert not os.path.exists("test_model.onnx")
    assert not os.path.exists("test_model.compiled")
    assert not os.path.exists("test_model.pk")
    assert not os.path.exists("test_model.vk")
    assert not os.path.exists("test_model_input.json")
    assert not os.path.exists("test_model_output.json")


