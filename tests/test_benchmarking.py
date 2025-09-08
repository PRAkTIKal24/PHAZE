import torch
import numpy as np
import asyncio
import pytest
from phaze.early_exit_models import SimpleEarlyExitModel
from phaze.crypto_primitives import RabinFingerprint
from phaze.zkml_integration import ZKMLProverVerifier, SimpleFullModel
from phaze.benchmarking import run_early_exit_benchmark, run_hashing_benchmark, run_zkml_benchmark, run_full_pipeline_benchmark

@pytest.mark.asyncio
async def test_run_early_exit_benchmark():
    model = SimpleEarlyExitModel()
    input_data = torch.randn(1, 10)
    avg_time = await run_early_exit_benchmark(model, input_data, num_iterations=5)
    assert isinstance(avg_time, float)
    assert avg_time >= 0

@pytest.mark.asyncio
async def test_run_hashing_benchmark():
    fingerprinter = RabinFingerprint(field_size=2**31 - 1, degree=10)
    data_vector = [np.random.randint(0, 100) for _ in range(10)]
    avg_time = await run_hashing_benchmark(fingerprinter, data_vector, num_iterations=5)
    assert isinstance(avg_time, float)
    assert avg_time >= 0

@pytest.mark.asyncio
async def test_run_zkml_benchmark():
    # This test requires ezkl to be installed and configured correctly
    # For now, we\"ll mock the ZKMLProverVerifier to avoid actual ezkl calls
    class MockZKMLProverVerifier:
        def __init__(self, model, name):
            pass
        async def setup(self, input_data):
            pass
        async def generate_proof(self, input_data):
            return "mock_proof", None
        async def verify_proof(self, proof, input_data):
            return True
        def cleanup(self):
            pass

    full_model = SimpleFullModel()
    zkml_input = torch.randn(1, 10)
    zkml_pv = MockZKMLProverVerifier(full_model, "mock")
    
    proving_time, verification_time = await run_zkml_benchmark(zkml_pv, zkml_input, num_iterations=1)
    assert isinstance(proving_time, float)
    assert proving_time >= 0
    assert isinstance(verification_time, float)
    assert verification_time >= 0

@pytest.mark.asyncio
async def test_run_full_pipeline_benchmark():
    # This test will run the actual benchmarks, so it might take some time
    # and requires ezkl to be functional.
    # For CI/CD, consider mocking or skipping this test if ezkl setup is complex.
    results = await run_full_pipeline_benchmark(num_iterations=1)
    assert "m_early_inference_ms" in results
    assert "hashing_ms" in results
    assert "zkml_proving_ms" in results
    assert "zkml_verification_ms" in results
    for key, value in results.items():
        assert isinstance(value, float)
        assert value >= 0


