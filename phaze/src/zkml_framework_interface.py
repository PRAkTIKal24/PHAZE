"""
Modular interface for different zkML frameworks in the PHAZE system.

This module provides abstract base classes and concrete implementations for various
zkML frameworks, allowing for easy benchmarking and comparison of different systems.
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class ZKMLFramework(Enum):
    """Enumeration of supported zkML frameworks."""

    EZKL = "ezkl"
    ZKCNN = "zkcnn"
    GROTH16 = "groth16"
    HALO = "halo"
    PLONKY = "plonky"
    RISC_ZERO = "risc_zero"
    STARK = "stark"


class ZKMLBackendInterface(ABC):
    """Abstract base class for zkML backend implementations."""

    def __init__(self, framework: ZKMLFramework, model: nn.Module, name: str):
        self.framework = framework
        self.model = model
        self.name = name
        self.is_setup = False

    @abstractmethod
    async def setup(self, input_data: torch.Tensor, **kwargs) -> None:
        """Setup the zkML system for the given model and input."""
        pass

    @abstractmethod
    async def generate_proof(
        self, input_data: torch.Tensor, **kwargs
    ) -> Tuple[Any, Any]:
        """Generate a zero-knowledge proof for the model inference."""
        pass

    @abstractmethod
    async def verify_proof(
        self, proof: Any, input_data: torch.Tensor, **kwargs
    ) -> bool:
        """Verify a zero-knowledge proof."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any temporary files or resources."""
        pass

    @abstractmethod
    def get_framework_info(self) -> Dict[str, Any]:
        """Get information about the framework (version, capabilities, etc.)."""
        pass


class BenchmarkMetrics:
    """Container for benchmark metrics."""

    def __init__(self):
        self.setup_time_ms: Optional[float] = None
        self.proving_time_ms: Optional[float] = None
        self.verification_time_ms: Optional[float] = None
        self.proof_size_bytes: Optional[int] = None
        self.memory_usage_mb: Optional[float] = None
        self.framework_info: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
        self.success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "setup_time_ms": self.setup_time_ms,
            "proving_time_ms": self.proving_time_ms,
            "verification_time_ms": self.verification_time_ms,
            "proof_size_bytes": self.proof_size_bytes,
            "memory_usage_mb": self.memory_usage_mb,
            "framework_info": self.framework_info,
            "error_message": self.error_message,
            "success": self.success,
        }


class ZKMLBenchmarkRunner:
    """Orchestrates benchmarking across different zkML frameworks."""

    def __init__(self):
        self.backends: Dict[ZKMLFramework, ZKMLBackendInterface] = {}

    def register_backend(self, backend: ZKMLBackendInterface) -> None:
        """Register a zkML backend for benchmarking."""
        self.backends[backend.framework] = backend

    async def benchmark_framework(
        self,
        framework: ZKMLFramework,
        input_data: torch.Tensor,
        num_iterations: int = 10,
    ) -> BenchmarkMetrics:
        """Benchmark a specific zkML framework."""
        metrics = BenchmarkMetrics()

        if framework not in self.backends:
            metrics.error_message = f"Framework {framework.value} not registered"
            return metrics

        backend = self.backends[framework]

        try:
            # Setup phase
            start_time = time.perf_counter()
            await backend.setup(input_data)
            setup_time = (time.perf_counter() - start_time) * 1000
            metrics.setup_time_ms = setup_time

            # Proving phase (multiple iterations)
            proving_times = []
            verification_times = []
            proof_sizes = []

            for _ in range(num_iterations):
                # Generate proof
                start_time = time.perf_counter()
                proof, output = await backend.generate_proof(input_data)
                proving_time = (time.perf_counter() - start_time) * 1000
                proving_times.append(proving_time)

                # Calculate proof size if possible
                if hasattr(proof, "__len__"):
                    proof_sizes.append(len(str(proof).encode("utf-8")))

                # Verify proof
                start_time = time.perf_counter()
                is_valid = await backend.verify_proof(proof, input_data)
                verification_time = (time.perf_counter() - start_time) * 1000
                verification_times.append(verification_time)

                if not is_valid:
                    metrics.error_message = (
                        f"Proof verification failed on iteration {_}"
                    )
                    return metrics

            # Calculate average metrics
            metrics.proving_time_ms = sum(proving_times) / len(proving_times)
            metrics.verification_time_ms = sum(verification_times) / len(
                verification_times
            )

            if proof_sizes:
                metrics.proof_size_bytes = sum(proof_sizes) // len(proof_sizes)

            metrics.framework_info = backend.get_framework_info()
            metrics.success = True

        except Exception as e:
            metrics.error_message = str(e)
        finally:
            backend.cleanup()

        return metrics

    async def benchmark_all_frameworks(
        self, input_data: torch.Tensor, num_iterations: int = 10
    ) -> Dict[str, BenchmarkMetrics]:
        """Benchmark all registered frameworks."""
        results = {}

        for framework in self.backends.keys():
            print(f"Benchmarking {framework.value}...")
            metrics = await self.benchmark_framework(
                framework, input_data, num_iterations
            )
            results[framework.value] = metrics

        return results

    def generate_comparison_report(
        self, benchmark_results: Dict[str, BenchmarkMetrics]
    ) -> str:
        """Generate a comparison report from benchmark results."""
        report = "PHAZE zkML Framework Benchmark Report\n"
        report += "=" * 50 + "\n\n"

        successful_results = {k: v for k, v in benchmark_results.items() if v.success}
        failed_results = {k: v for k, v in benchmark_results.items() if not v.success}

        if successful_results:
            report += "Successful Benchmarks:\n"
            report += "-" * 25 + "\n"

            # Sort by proving time
            sorted_results = sorted(
                successful_results.items(),
                key=lambda x: x[1].proving_time_ms or float("inf"),
            )

            for framework_name, metrics in sorted_results:
                report += f"\n{framework_name.upper()}:\n"
                report += f"  Setup Time: {metrics.setup_time_ms:.2f} ms\n"
                report += f"  Proving Time: {metrics.proving_time_ms:.2f} ms\n"
                report += (
                    f"  Verification Time: {metrics.verification_time_ms:.2f} ms\n"
                )
                if metrics.proof_size_bytes:
                    report += f"  Proof Size: {metrics.proof_size_bytes} bytes\n"
                if metrics.framework_info:
                    report += f"  Framework Info: {metrics.framework_info}\n"

        if failed_results:
            report += "\n\nFailed Benchmarks:\n"
            report += "-" * 20 + "\n"

            for framework_name, metrics in failed_results.items():
                report += f"\n{framework_name.upper()}:\n"
                report += f"  Error: {metrics.error_message}\n"

        return report


class CryptographicPrimitiveBenchmark:
    """Benchmarking suite for cryptographic primitives used in PHAZE."""

    def __init__(self):
        self.primitives = {}

    def register_primitive(self, name: str, primitive_func, *args, **kwargs):
        """Register a cryptographic primitive for benchmarking."""
        self.primitives[name] = (primitive_func, args, kwargs)

    def benchmark_primitive(
        self, name: str, num_iterations: int = 1000
    ) -> Dict[str, float]:
        """Benchmark a specific cryptographic primitive."""
        if name not in self.primitives:
            raise ValueError(f"Primitive {name} not registered")

        primitive_func, args, kwargs = self.primitives[name]

        # Warm-up
        for _ in range(10):
            primitive_func(*args, **kwargs)

        # Actual benchmark
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            primitive_func(*args, **kwargs)
        end_time = time.perf_counter()

        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / num_iterations

        return {
            "total_time_ms": total_time_ms,
            "avg_time_ms": avg_time_ms,
            "iterations": num_iterations,
        }

    def benchmark_all_primitives(
        self, num_iterations: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark all registered primitives."""
        results = {}

        for name in self.primitives.keys():
            print(f"Benchmarking {name}...")
            results[name] = self.benchmark_primitive(name, num_iterations)

        return results
