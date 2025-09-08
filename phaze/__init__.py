"""
PHAZE: Privacy-preserving High-energy physics Analysis with Zero-knowledge proofs and Early-exit models

A framework for cryptographic and ZKML-based low latency inference at LHC.
"""

from .early_exit_models import SimpleEarlyExitModel
from .crypto_primitives import RabinFingerprint
from .zkml_integration import ZKMLProverVerifier, SimpleFullModel
from .rust_zkml_backend import RustZKMLBackend
from .benchmarking import (
    run_early_exit_benchmark, 
    run_hashing_benchmark, 
    run_zkml_benchmark, 
    run_full_pipeline_benchmark
)

# New modular architecture
from .zkml_framework_interface import (
    ZKMLFramework,
    ZKMLBackendInterface,
    BenchmarkMetrics,
    ZKMLBenchmarkRunner,
    CryptographicPrimitiveBenchmark
)
from .zkml_backends import (
    EZKLBackend,
    MockZKMLBackend,
    ZKCNNBackend,
    Groth16Backend,
    HaloBackend,
    PlonkyBackend,
    RiscZeroBackend,
    StarkBackend,
    create_backend
)
from .phaze_benchmark_suite import (
    PHAZEBenchmarkConfig,
    PHAZEBenchmarkResults,
    PHAZEBenchmarkSuite
)

__version__ = "0.3.2"
__author__ = "Pratik Jawahar"
__email__ = "pratik.jawahar@cern.ch"

__all__ = [
    # Core models
    "SimpleEarlyExitModel",
    "SimpleFullModel",
    
    # Cryptographic primitives
    "RabinFingerprint",
    "RustZKMLBackend",
    
    # zkML integration
    "ZKMLProverVerifier",
    
    # Legacy benchmarking
    "run_early_exit_benchmark",
    "run_hashing_benchmark", 
    "run_zkml_benchmark",
    "run_full_pipeline_benchmark",
    
    # New modular architecture
    "ZKMLFramework",
    "ZKMLBackendInterface",
    "BenchmarkMetrics",
    "ZKMLBenchmarkRunner",
    "CryptographicPrimitiveBenchmark",
    
    # zkML backends
    "EZKLBackend",
    "MockZKMLBackend",
    "ZKCNNBackend",
    "Groth16Backend",
    "HaloBackend",
    "PlonkyBackend",
    "RiscZeroBackend",
    "StarkBackend",
    "create_backend",
    
    # Comprehensive benchmarking
    "PHAZEBenchmarkConfig",
    "PHAZEBenchmarkResults",
    "PHAZEBenchmarkSuite",
]

