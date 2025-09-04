import pytest
import torch
import torch.nn as nn
import os
from phaze.zkml_integration import ZKMLProverVerifier, SimpleFullModel

@pytest.fixture(scope="module")
def toy_full_model():
    return SimpleFullModel()

@pytest.fixture(scope="module")
def dummy_input():
    return torch.randn(1, 10)

@pytest.fixture(scope="module")
def zkml_pv(toy_full_model):
    pv = ZKMLProverVerifier(toy_full_model, "mock")
    yield pv
    pv.cleanup() # Clean up ONNX file after tests

def test_zkml_prover_verifier_init(toy_full_model):
    pv = ZKMLProverVerifier(toy_full_model, "mock")
    assert pv.model == toy_full_model
    assert pv.zkml_system_name == "mock"
    assert pv.onnx_path.endswith(f"{toy_full_model.__class__.__name__}.onnx")

def test_export_to_onnx(zkml_pv, dummy_input):
    zkml_pv._export_to_onnx(dummy_input)
    assert os.path.exists(zkml_pv.onnx_path)
    # Basic ONNX model check
    model = onnx.load(zkml_pv.onnx_path)
    onnx.checker.check_model(model)

def test_generate_proof(zkml_pv, dummy_input):
    proof = zkml_pv.generate_proof(dummy_input)
    assert "model_output" in proof
    assert "proof_string" in proof
    assert isinstance(proof["model_output"], torch.Tensor)
    assert isinstance(proof["proof_string"], str)
    assert "Proof_for_model" in proof["proof_string"]

def test_verify_proof_valid(zkml_pv, dummy_input):
    proof = zkml_pv.generate_proof(dummy_input)
    is_valid = zkml_pv.verify_proof(proof)
    assert is_valid is True

def test_verify_proof_invalid(zkml_pv):
    invalid_proof = {"model_output": torch.randn(1, 2), "
        
        # Simulate a tampered proof string
        "proof_string": "Tampered_proof_string"
    }
    is_valid = zkml_pv.verify_proof(invalid_proof)
    assert is_valid is False

def test_cleanup(toy_full_model, dummy_input):
    pv = ZKMLProverVerifier(toy_full_model, "mock")
    pv.generate_proof(dummy_input) # This creates the ONNX file
    assert os.path.exists(pv.onnx_path)
    pv.cleanup()
    assert not os.path.exists(pv.onnx_path)


