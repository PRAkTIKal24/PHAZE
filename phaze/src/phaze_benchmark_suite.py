"""
Comprehensive benchmarking suite for the PHAZE framework.

This module provides a complete benchmarking system for all components of PHAZE:
- Early exit models (M_early)
- Full models (M_full)
- Cryptographic primitives (hashing, fingerprinting)
- zkML frameworks (EZKL, Groth16, Halo, Plonky, etc.)
- End-to-end pipeline performance
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from .crypto_primitives import RabinFingerprint
from .early_exit_models import SimpleEarlyExitModel
from .zkml_backends import create_backend
from .zkml_framework_interface import (
    BenchmarkMetrics,
    CryptographicPrimitiveBenchmark,
    ZKMLBenchmarkRunner,
    ZKMLFramework,
)
from .zkml_integration import SimpleFullModel


@dataclass
class PHAZEBenchmarkConfig:
    """Configuration for PHAZE benchmarking."""

    num_iterations: int = 10
    zkml_iterations: int = 5  # Fewer iterations for expensive zkML operations
    crypto_iterations: int = 1000
    input_size: int = 10
    batch_size: int = 1
    frameworks_to_test: List[ZKMLFramework] = None
    save_results: bool = True
    results_dir: str = "/tmp/phaze_benchmark_results"

    def __post_init__(self):
        if self.frameworks_to_test is None:
            self.frameworks_to_test = [
                ZKMLFramework.EZKL,
                ZKMLFramework.ZKCNN,
                ZKMLFramework.GROTH16,
                ZKMLFramework.HALO,
                ZKMLFramework.PLONKY,
            ]


@dataclass
class PHAZEBenchmarkResults:
    """Complete benchmark results for PHAZE system."""

    config: PHAZEBenchmarkConfig
    m_early_results: Dict[str, float]
    m_full_results: Dict[str, float]
    crypto_results: Dict[str, Dict[str, float]]
    zkml_results: Dict[str, BenchmarkMetrics]
    pipeline_results: Dict[str, float]
    total_time_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            "config": asdict(self.config),
            "m_early_results": self.m_early_results,
            "m_full_results": self.m_full_results,
            "crypto_results": self.crypto_results,
            "zkml_results": {k: v.to_dict() for k, v in self.zkml_results.items()},
            "pipeline_results": self.pipeline_results,
            "total_time_ms": self.total_time_ms,
            "timestamp": self.timestamp,
        }

    def save_to_file(self, filepath: str) -> None:
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class PHAZEBenchmarkSuite:
    """Comprehensive benchmarking suite for PHAZE framework."""

    def __init__(self, config: PHAZEBenchmarkConfig = None):
        self.config = config or PHAZEBenchmarkConfig()
        self.zkml_runner = ZKMLBenchmarkRunner()
        self.crypto_benchmark = CryptographicPrimitiveBenchmark()

        # Initialize models
        self.m_early = SimpleEarlyExitModel()
        self.m_full = SimpleFullModel()

        # Setup test input
        self.test_input = torch.randn(self.config.batch_size, self.config.input_size)

        # Register zkML backends
        self._register_zkml_backends()

        # Register cryptographic primitives
        self._register_crypto_primitives()

    def _register_zkml_backends(self):
        """Register all zkML backends for benchmarking."""
        for framework in self.config.frameworks_to_test:
            try:
                backend = create_backend(
                    framework, self.m_full, f"benchmark_{framework.value}"
                )
                self.zkml_runner.register_backend(backend)
                print(f"Registered {framework.value} backend")
            except Exception as e:
                print(f"Failed to register {framework.value} backend: {e}")

    def _register_crypto_primitives(self):
        """Register cryptographic primitives for benchmarking."""
        # Rabin fingerprinting
        fingerprinter = RabinFingerprint(field_size=2**31 - 1, degree=10)
        test_data = [np.random.randint(0, 100) for _ in range(10)]

        self.crypto_benchmark.register_primitive(
            "rabin_fingerprint", fingerprinter.compute_hash, test_data
        )

        # Add more cryptographic primitives here as they are implemented
        # For example:
        # - Merkle tree operations
        # - Hash functions (SHA256, Keccak256)
        # - Digital signatures
        # - Commitment schemes

    async def benchmark_early_exit_model(self) -> Dict[str, float]:
        """Benchmark the early exit model (M_early)."""
        print("Benchmarking M_early model...")

        # Warm-up
        for _ in range(10):
            _ = self.m_early(self.test_input)

        # Benchmark inference time
        start_time = time.perf_counter()
        for _ in range(self.config.num_iterations):
            with torch.no_grad():
                _ = self.m_early(self.test_input)
        end_time = time.perf_counter()

        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / self.config.num_iterations

        # Benchmark memory usage (approximate)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return {
            "avg_inference_time_ms": avg_time_ms,
            "total_time_ms": total_time_ms,
            "iterations": self.config.num_iterations,
            "throughput_inferences_per_sec": 1000 / avg_time_ms
            if avg_time_ms > 0
            else 0,
        }

    async def benchmark_full_model(self) -> Dict[str, float]:
        """Benchmark the full model (M_full)."""
        print("Benchmarking M_full model...")

        # Warm-up
        for _ in range(10):
            _ = self.m_full(self.test_input)

        # Benchmark inference time
        start_time = time.perf_counter()
        for _ in range(self.config.num_iterations):
            with torch.no_grad():
                _ = self.m_full(self.test_input)
        end_time = time.perf_counter()

        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / self.config.num_iterations

        return {
            "avg_inference_time_ms": avg_time_ms,
            "total_time_ms": total_time_ms,
            "iterations": self.config.num_iterations,
            "throughput_inferences_per_sec": 1000 / avg_time_ms
            if avg_time_ms > 0
            else 0,
        }

    async def benchmark_crypto_primitives(self) -> Dict[str, Dict[str, float]]:
        """Benchmark cryptographic primitives."""
        print("Benchmarking cryptographic primitives...")
        return self.crypto_benchmark.benchmark_all_primitives(
            self.config.crypto_iterations
        )

    async def benchmark_zkml_frameworks(self) -> Dict[str, BenchmarkMetrics]:
        """Benchmark all registered zkML frameworks."""
        print("Benchmarking zkML frameworks...")
        return await self.zkml_runner.benchmark_all_frameworks(
            self.test_input, self.config.zkml_iterations
        )

    async def benchmark_end_to_end_pipeline(self) -> Dict[str, float]:
        """Benchmark the complete PHAZE pipeline."""
        print("Benchmarking end-to-end pipeline...")

        pipeline_times = []

        for _ in range(self.config.num_iterations):
            start_time = time.perf_counter()

            # Step 1: M_early inference
            with torch.no_grad():
                early_output = self.m_early(self.test_input)

            # Step 2: Decision logic (simplified)
            # In real PHAZE, this would involve confidence thresholding
            use_full_model = torch.rand(1).item() > 0.5  # Random decision for benchmark

            if use_full_model:
                # Step 3: Cryptographic hashing of early output
                fingerprinter = RabinFingerprint(field_size=2**31 - 1, degree=5)
                early_data = early_output.flatten().numpy().astype(int)[:5].tolist()
                hash_result = fingerprinter.compute_hash(early_data)

                # Step 4: M_full inference (would be done with zkML in practice)
                with torch.no_grad():
                    full_output = self.m_full(self.test_input)

            end_time = time.perf_counter()
            pipeline_times.append((end_time - start_time) * 1000)

        avg_pipeline_time = np.mean(pipeline_times)

        return {
            "avg_pipeline_time_ms": avg_pipeline_time,
            "min_pipeline_time_ms": np.min(pipeline_times),
            "max_pipeline_time_ms": np.max(pipeline_times),
            "std_pipeline_time_ms": np.std(pipeline_times),
            "iterations": self.config.num_iterations,
        }

    async def run_full_benchmark(self) -> PHAZEBenchmarkResults:
        """Run the complete PHAZE benchmark suite."""
        print("Starting PHAZE comprehensive benchmark...")
        start_time = time.perf_counter()

        # Run all benchmarks
        m_early_results = await self.benchmark_early_exit_model()
        m_full_results = await self.benchmark_full_model()
        crypto_results = await self.benchmark_crypto_primitives()
        zkml_results = await self.benchmark_zkml_frameworks()
        pipeline_results = await self.benchmark_end_to_end_pipeline()

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Create results object
        results = PHAZEBenchmarkResults(
            config=self.config,
            m_early_results=m_early_results,
            m_full_results=m_full_results,
            crypto_results=crypto_results,
            zkml_results=zkml_results,
            pipeline_results=pipeline_results,
            total_time_ms=total_time_ms,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Save results if configured
        if self.config.save_results:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                self.config.results_dir, f"phaze_benchmark_{timestamp_str}.json"
            )
            results.save_to_file(filepath)
            print(f"Results saved to: {filepath}")

        return results

    def generate_report(self, results: PHAZEBenchmarkResults) -> str:
        """Generate a comprehensive benchmark report."""
        report = "PHAZE Framework Comprehensive Benchmark Report\n"
        report += "=" * 60 + "\n\n"

        report += f"Timestamp: {results.timestamp}\n"
        report += f"Total Benchmark Time: {results.total_time_ms:.2f} ms\n\n"

        # Configuration
        report += "Configuration:\n"
        report += "-" * 15 + "\n"
        report += f"  Iterations: {results.config.num_iterations}\n"
        report += f"  zkML Iterations: {results.config.zkml_iterations}\n"
        report += f"  Crypto Iterations: {results.config.crypto_iterations}\n"
        report += f"  Input Size: {results.config.input_size}\n"
        report += f"  Batch Size: {results.config.batch_size}\n\n"

        # M_early results
        report += "M_early Model Performance:\n"
        report += "-" * 30 + "\n"
        for key, value in results.m_early_results.items():
            report += f"  {key}: {value:.4f}\n"
        report += "\n"

        # M_full results
        report += "M_full Model Performance:\n"
        report += "-" * 29 + "\n"
        for key, value in results.m_full_results.items():
            report += f"  {key}: {value:.4f}\n"
        report += "\n"

        # Cryptographic primitives
        report += "Cryptographic Primitives Performance:\n"
        report += "-" * 40 + "\n"
        for primitive, metrics in results.crypto_results.items():
            report += f"  {primitive}:\n"
            for key, value in metrics.items():
                report += f"    {key}: {value:.4f}\n"
        report += "\n"

        # zkML frameworks
        report += "zkML Frameworks Performance:\n"
        report += "-" * 32 + "\n"
        successful_zkml = {k: v for k, v in results.zkml_results.items() if v.success}
        failed_zkml = {k: v for k, v in results.zkml_results.items() if not v.success}

        if successful_zkml:
            for framework, metrics in successful_zkml.items():
                report += f"  {framework.upper()}:\n"
                report += f"    Setup Time: {metrics.setup_time_ms:.2f} ms\n"
                report += f"    Proving Time: {metrics.proving_time_ms:.2f} ms\n"
                report += (
                    f"    Verification Time: {metrics.verification_time_ms:.2f} ms\n"
                )
                if metrics.proof_size_bytes:
                    report += f"    Proof Size: {metrics.proof_size_bytes} bytes\n"
                report += "\n"

        if failed_zkml:
            report += "  Failed Frameworks:\n"
            for framework, metrics in failed_zkml.items():
                report += f"    {framework.upper()}: {metrics.error_message}\n"
            report += "\n"

        # Pipeline results
        report += "End-to-End Pipeline Performance:\n"
        report += "-" * 35 + "\n"
        for key, value in results.pipeline_results.items():
            report += f"  {key}: {value:.4f}\n"
        report += "\n"

        # Performance summary
        report += "Performance Summary:\n"
        report += "-" * 20 + "\n"

        m_early_throughput = results.m_early_results.get(
            "throughput_inferences_per_sec", 0
        )
        m_full_throughput = results.m_full_results.get(
            "throughput_inferences_per_sec", 0
        )

        report += f"  M_early Throughput: {m_early_throughput:.2f} inferences/sec\n"
        report += f"  M_full Throughput: {m_full_throughput:.2f} inferences/sec\n"

        if successful_zkml:
            fastest_zkml = min(
                successful_zkml.items(), key=lambda x: x[1].proving_time_ms
            )
            report += f"  Fastest zkML Framework: {fastest_zkml[0]} ({fastest_zkml[1].proving_time_ms:.2f} ms proving)\n"

        avg_pipeline = results.pipeline_results.get("avg_pipeline_time_ms", 0)
        if avg_pipeline > 0:
            pipeline_throughput = 1000 / avg_pipeline
            report += (
                f"  Pipeline Throughput: {pipeline_throughput:.2f} pipelines/sec\n"
            )

        return report


async def main():
    """Main function for running PHAZE benchmarks."""
    config = PHAZEBenchmarkConfig(
        num_iterations=50, zkml_iterations=3, crypto_iterations=1000
    )

    suite = PHAZEBenchmarkSuite(config)
    results = await suite.run_full_benchmark()

    print("\n" + "=" * 60)
    print(suite.generate_report(results))


if __name__ == "__main__":
    asyncio.run(main())
