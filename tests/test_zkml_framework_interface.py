import pytest
import torch
import asyncio
from phaze.zkml_framework_interface import (
    ZKMLFramework, ZKMLBenchmarkRunner, BenchmarkMetrics, 
    CryptographicPrimitiveBenchmark
)
from phaze.zkml_backends import MockZKMLBackend, create_backend
from phaze.early_exit_models import SimpleEarlyExitModel


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    return SimpleEarlyExitModel()


@pytest.fixture
def test_input():
    """Create test input data."""
    return torch.randn(1, 10)


def test_benchmark_metrics():
    """Test BenchmarkMetrics class."""
    metrics = BenchmarkMetrics()
    
    # Test initial state
    assert not metrics.success
    assert metrics.setup_time_ms is None
    assert metrics.proving_time_ms is None
    assert metrics.verification_time_ms is None
    
    # Test setting values
    metrics.setup_time_ms = 100.0
    metrics.proving_time_ms = 500.0
    metrics.verification_time_ms = 50.0
    metrics.success = True
    
    # Test to_dict conversion
    result_dict = metrics.to_dict()
    assert result_dict["setup_time_ms"] == 100.0
    assert result_dict["proving_time_ms"] == 500.0
    assert result_dict["verification_time_ms"] == 50.0
    assert result_dict["success"] is True


@pytest.mark.asyncio
async def test_mock_zkml_backend(dummy_model, test_input):
    """Test MockZKMLBackend functionality."""
    backend = MockZKMLBackend(ZKMLFramework.GROTH16, dummy_model, "test_mock")
    
    # Test initial state
    assert backend.framework == ZKMLFramework.GROTH16
    assert backend.name == "test_mock"
    assert not backend.is_setup
    
    # Test setup
    await backend.setup(test_input)
    assert backend.is_setup
    
    # Test proof generation
    proof, output = await backend.generate_proof(test_input)
    assert proof == backend.mock_proof
    assert output is not None
    
    # Test proof verification
    is_valid = await backend.verify_proof(proof, test_input)
    assert is_valid
    
    # Test invalid proof verification
    invalid_proof = {"proof": "invalid", "public_inputs": ["invalid"]}
    is_valid = await backend.verify_proof(invalid_proof, test_input)
    assert not is_valid
    
    # Test framework info
    info = backend.get_framework_info()
    assert info["framework"] == "GROTH16"
    assert "status" in info
    
    # Test cleanup
    backend.cleanup()  # Should not raise any errors


@pytest.mark.asyncio
async def test_zkml_benchmark_runner(dummy_model, test_input):
    """Test ZKMLBenchmarkRunner functionality."""
    runner = ZKMLBenchmarkRunner()
    
    # Register a mock backend
    backend = MockZKMLBackend(ZKMLFramework.HALO, dummy_model, "test_halo")
    runner.register_backend(backend)
    
    # Test single framework benchmark
    metrics = await runner.benchmark_framework(ZKMLFramework.HALO, test_input, num_iterations=2)
    
    assert metrics.success
    assert metrics.setup_time_ms is not None
    assert metrics.proving_time_ms is not None
    assert metrics.verification_time_ms is not None
    assert metrics.framework_info is not None
    
    # Test benchmark for unregistered framework
    metrics = await runner.benchmark_framework(ZKMLFramework.PLONKY, test_input)
    assert not metrics.success
    assert "not registered" in metrics.error_message
    
    # Test benchmark all frameworks
    results = await runner.benchmark_all_frameworks(test_input, num_iterations=1)
    assert ZKMLFramework.HALO.value in results
    assert results[ZKMLFramework.HALO.value].success


def test_cryptographic_primitive_benchmark():
    """Test CryptographicPrimitiveBenchmark functionality."""
    benchmark = CryptographicPrimitiveBenchmark()
    
    # Register a simple primitive
    def simple_hash(data):
        return sum(data) % 1000
    
    test_data = [1, 2, 3, 4, 5]
    benchmark.register_primitive("simple_hash", simple_hash, test_data)
    
    # Test single primitive benchmark
    results = benchmark.benchmark_primitive("simple_hash", num_iterations=100)
    
    assert "total_time_ms" in results
    assert "avg_time_ms" in results
    assert "iterations" in results
    assert results["iterations"] == 100
    assert results["total_time_ms"] > 0
    assert results["avg_time_ms"] > 0
    
    # Test benchmark all primitives
    all_results = benchmark.benchmark_all_primitives(num_iterations=50)
    assert "simple_hash" in all_results
    assert all_results["simple_hash"]["iterations"] == 50
    
    # Test unregistered primitive
    with pytest.raises(ValueError, match="not registered"):
        benchmark.benchmark_primitive("nonexistent_primitive")


def test_create_backend_factory(dummy_model):
    """Test the create_backend factory function."""
    # Test EZKL backend creation (might fail if ezkl not available)
    try:
        backend = create_backend(ZKMLFramework.EZKL, dummy_model, "test_ezkl")
        assert backend.framework == ZKMLFramework.EZKL
    except ImportError:
        # EZKL not available, skip this test
        pass
    
    # Test mock backend creation
    backend = create_backend(ZKMLFramework.GROTH16, dummy_model, "test_groth16")
    assert backend.framework == ZKMLFramework.GROTH16
    assert backend.name == "test_groth16"
    
    backend = create_backend(ZKMLFramework.HALO, dummy_model, "test_halo")
    assert backend.framework == ZKMLFramework.HALO
    
    backend = create_backend(ZKMLFramework.PLONKY, dummy_model, "test_plonky")
    assert backend.framework == ZKMLFramework.PLONKY
    
    # Test that all supported frameworks work
    for framework in ZKMLFramework:
        backend = create_backend(framework, dummy_model, f"test_{framework.value}")
        assert backend.framework == framework


@pytest.mark.asyncio
async def test_benchmark_runner_report_generation(dummy_model, test_input):
    """Test benchmark report generation."""
    runner = ZKMLBenchmarkRunner()
    
    # Register multiple backends
    backend1 = MockZKMLBackend(ZKMLFramework.GROTH16, dummy_model, "test_groth16")
    backend2 = MockZKMLBackend(ZKMLFramework.HALO, dummy_model, "test_halo")
    
    runner.register_backend(backend1)
    runner.register_backend(backend2)
    
    # Run benchmarks
    results = await runner.benchmark_all_frameworks(test_input, num_iterations=1)
    
    # Generate report
    report = runner.generate_comparison_report(results)
    
    assert "PHAZE zkML Framework Benchmark Report" in report
    assert "Successful Benchmarks:" in report
    assert "GROTH16:" in report
    assert "HALO:" in report
    assert "Setup Time:" in report
    assert "Proving Time:" in report
    assert "Verification Time:" in report


def test_zkml_framework_enum():
    """Test ZKMLFramework enum values."""
    assert ZKMLFramework.EZKL.value == "ezkl"
    assert ZKMLFramework.GROTH16.value == "groth16"
    assert ZKMLFramework.HALO.value == "halo"
    assert ZKMLFramework.PLONKY.value == "plonky"
    assert ZKMLFramework.RISC_ZERO.value == "risc_zero"
    assert ZKMLFramework.STARK.value == "stark"
    
    # Test that all frameworks are unique
    frameworks = list(ZKMLFramework)
    framework_values = [f.value for f in frameworks]
    assert len(framework_values) == len(set(framework_values))

