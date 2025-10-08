"""
PHAZE: Privacy-preserving High-energy physics Analysis
with Zero-knowledge proofs and Early-exit models

A framework for cryptographic and ZKML-based low latency inference at LHC.
"""

from .src.benchmarking import (
    run_early_exit_benchmark,
    run_full_pipeline_benchmark,
    run_hashing_benchmark,
    run_zkml_benchmark,
)
from .src.comprehensive_benchmark import (
    ComprehensiveBenchmarkSuite,
    run_phaze_benchmarks,
)
from .src.crypto_primitives import RabinFingerprint
from .src.early_exit_models import SimpleEarlyExitModel as LegacySimpleEarlyExitModel
from .src.model_architectures import (
    ConvolutionalEarlyExitModel,
    ModelComplexity,
    MultiExitModel,
    PHAZEModelFactory,
    SimpleEarlyExitModel,
    TransformerEarlyExitModel,
    create_simple_early_exit_model,
    create_simple_full_model,
)
from .src.phaze_benchmark_suite import (
    PHAZEBenchmarkConfig,
    PHAZEBenchmarkResults,
    PHAZEBenchmarkSuite,
)
from .src.plotter_registry import register_all_plotters

# Plotting system
from .src.plotting import PHAZEPlotSuite, PlotConfig, PlotStyle
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

# Auto-register plotters when package is imported
register_all_plotters()

# Get version dynamically from package metadata
try:
    from importlib.metadata import version

    __version__ = version("phaze")
except ImportError:
    # Fallback for Python < 3.8
    try:
        from importlib_metadata import version

        __version__ = version("phaze")
    except ImportError:
        __version__ = "unknown"

__author__ = "Pratik Jawahar"
__email__ = "pratik.jawahar@cern.ch"

__all__ = [
    # Core models
    "SimpleEarlyExitModel",
    "LegacySimpleEarlyExitModel",
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
    # Comprehensive benchmarking
    "ComprehensiveBenchmarkSuite",
    "run_phaze_benchmarks",
    # Model architectures
    "PHAZEModelFactory",
    "ModelComplexity",
    "ConvolutionalEarlyExitModel",
    "TransformerEarlyExitModel",
    "MultiExitModel",
    "create_simple_early_exit_model",
    "create_simple_full_model",
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
    # Plotting system
    "PHAZEPlotSuite",
    "PlotConfig",
    "PlotStyle",
]
