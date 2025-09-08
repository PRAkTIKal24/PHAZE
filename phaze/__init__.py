"""
PHAZE: Privacy-preserving High-energy physics Analysis with Zero-knowledge proofs and Early-exit models

A framework for cryptographic and ZKML-based low latency inference at LHC.
"""

from .src.benchmarking import (
    run_early_exit_benchmark,
    run_full_pipeline_benchmark,
    run_hashing_benchmark,
    run_zkml_benchmark,
)
from .src.crypto_primitives import RabinFingerprint
from .src.early_exit_models import SimpleEarlyExitModel
from .src.phaze_benchmark_suite import (
    PHAZEBenchmarkConfig,
    PHAZEBenchmarkResults,
    PHAZEBenchmarkSuite,
)
from .src.rust_zkml_backend import RustZKMLBackend
from .src.zkml_backends import (
    EZKLBackend,
    Groth16Backend,
    HaloBackend,
    MockZKMLBackend,
    PlonkyBackend,
    RiscZeroBackend,
    StarkBackend,
    ZKCNNBackend,
    create_backend,
)

# New modular architecture
from .src.zkml_framework_interface import (
    BenchmarkMetrics,
    CryptographicPrimitiveBenchmark,
    ZKMLBackendInterface,
    ZKMLBenchmarkRunner,
    ZKMLFramework,
)
from .src.zkml_integration import SimpleFullModel, ZKMLProverVerifier

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
