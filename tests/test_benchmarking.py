import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path for legacy imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from phaze import (
    RabinFingerprint,
    SimpleEarlyExitModel,
    SimpleFullModel,
)
from legacy.src.benchmarking import (
    run_early_exit_benchmark,
    run_hashing_benchmark,
    run_zkml_benchmark,
)


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

    proving_time, verification_time = await run_zkml_benchmark(
        zkml_pv, zkml_input, num_iterations=1
    )
    assert isinstance(proving_time, float)
    assert proving_time >= 0
    assert isinstance(verification_time, float)
    assert verification_time >= 0


# Removed async benchmarking test due to event loop conflicts
