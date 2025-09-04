import pytest
import torch
import numpy as np
from phaze.early_exit_models import SimpleEarlyExitModel
from phaze.crypto_primitives import RabinFingerprint
from phaze.zkml_integration import ZKMLProverVerifier, SimpleFullModel
from phaze.benchmarking import (
    run_early_exit_benchmark,
    run_hashing_benchmark,
    run_zkml_benchmark,
    run_full_pipeline_benchmark
)

@pytest.fixture(scope="module")
def early_model():
    return SimpleEarlyExitModel()

@pytest.fixture(scope="module")
def early_input():
    return torch.randn(1, 10)

@pytest.fixture(scope="module")
def fingerprinter():
    return RabinFingerprint(field_size=2**64 - 59, degree=10)

@pytest.fixture(scope="module")
def hash_data():
    return [np.random.randint(0, 100) for _ in range(10)]

@pytest.fixture(scope="module")
def full_model():
    return SimpleFullModel()

@pytest.fixture(scope="module")
def zkml_input():
    return torch.randn(1, 10)

@pytest.fixture(scope="module")
def zkml_pv(full_model):
    pv = ZKMLProverVerifier(full_model, "mock")
    yield pv
    pv.cleanup()

def test_run_early_exit_benchmark(early_model, early_input):
    avg_time = run_early_exit_benchmark(early_model, early_input, num_iterations=5)
    assert isinstance(avg_time, float)
    assert avg_time >= 0

def test_run_hashing_benchmark(fingerprinter, hash_data):
    avg_time = run_hashing_benchmark(fingerprinter, hash_data, num_iterations=5)
    assert isinstance(avg_time, float)
    assert avg_time >= 0

def test_run_zkml_benchmark(zkml_pv, zkml_input):
    avg_proving_time, avg_verification_time = run_zkml_benchmark(zkml_pv, zkml_input, num_iterations=2)
    assert isinstance(avg_proving_time, float)
    assert avg_proving_time >= 0
    assert isinstance(avg_verification_time, float)
    assert avg_verification_time >= 0

def test_run_full_pipeline_benchmark():
    results = run_full_pipeline_benchmark(num_iterations=2)
    assert isinstance(results, dict)
    assert "m_early_inference_ms" in results
    assert "hashing_ms" in results
    assert "zkml_proving_ms" in results
    assert "zkml_verification_ms" in results
    for key, value in results.items():
        assert isinstance(value, float)
        assert value >= 0


