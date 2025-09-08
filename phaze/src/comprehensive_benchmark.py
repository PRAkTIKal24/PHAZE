"""
Comprehensive benchmarking system for the PHAZE framework.

This module provides a complete benchmarking suite that can evaluate:
1. Different zkML frameworks (ezkl, groth16, plonky, halo)
2. Various model architectures and complexities
3. Cryptographic primitive performance
4. End-to-end PHAZE workflow performance
"""

import asyncio
import json
import logging
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# import seaborn as sns
import torch

from .crypto_primitives import RabinFingerprint, ShamirSecretSharing
from .model_architectures import ModelComplexity, PHAZEModelFactory
from .rust_zkml_backend import RustZKMLFrameworkManager
from .zkml_integration import PHAZEZKMLIntegration, ZKMLProverVerifier


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results."""

    test_name: str
    framework: str
    architecture: str
    complexity: str
    input_size: int
    output_size: int
    setup_time: float
    proof_time: float
    verification_time: float
    total_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    accuracy: Optional[float] = None
    model_parameters: Optional[int] = None
    proof_size_bytes: Optional[int] = None
    additional_metrics: Optional[Dict[str, Any]] = None


@dataclass
class CryptoBenchmarkResult:
    """Data class for cryptographic primitive benchmark results."""

    primitive_name: str
    operation: str
    input_size: int
    execution_time: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    success: bool
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Monitor system performance during benchmarks."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_cpu_percent = None
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start performance monitoring."""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu_percent = self.process.cpu_percent()

    def stop_monitoring(self) -> Tuple[float, float, float]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu_percent = self.process.cpu_percent()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = end_time - self.start_time
        memory_usage = max(end_memory - self.start_memory, peak / 1024 / 1024)
        cpu_usage = max(end_cpu_percent, self.start_cpu_percent)

        return execution_time, memory_usage, cpu_usage


class ZKMLFrameworkBenchmark:
    """Benchmark different zkML frameworks."""

    def __init__(self):
        self.zkml_integration = PHAZEZKMLIntegration()
        self.rust_manager = RustZKMLFrameworkManager()
        self.results = []
        self.logger = logging.getLogger(__name__)

    async def benchmark_framework(
        self,
        framework_name: str,
        architecture: str = "simple",
        complexity: str = "light",
        input_size: int = 10,
        output_size: int = 5,
        num_trials: int = 3,
    ) -> List[BenchmarkResult]:
        """Benchmark a specific framework configuration."""
        results = []

        for trial in range(num_trials):
            monitor = PerformanceMonitor()
            result = BenchmarkResult(
                test_name=f"{framework_name}_{architecture}_{complexity}_trial_{trial}",
                framework=framework_name,
                architecture=architecture,
                complexity=complexity,
                input_size=input_size,
                output_size=output_size,
                setup_time=0.0,
                proof_time=0.0,
                verification_time=0.0,
                total_time=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
            )

            try:
                monitor.start_monitoring()

                # Create model
                model_name = f"test_model_{trial}"
                model = self.zkml_integration.create_and_register_model(
                    model_name,
                    architecture,
                    complexity,
                    "full",
                    input_size=input_size,
                    output_size=output_size,
                )

                result.model_parameters = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )

                # Generate test data
                input_data = torch.randn(1, input_size)

                # Setup phase
                setup_start = time.time()
                if framework_name == "ezkl":
                    await self.zkml_integration.setup_model(model_name, input_data)
                elif framework_name in ["groth16", "plonky", "halo"]:
                    backend = self.rust_manager.get_backend(framework_name)
                    if framework_name == "groth16":
                        backend.setup(1000)
                    elif framework_name == "plonky":
                        backend.setup(512)
                    elif framework_name == "halo":
                        backend.setup(8)
                setup_end = time.time()
                result.setup_time = setup_end - setup_start

                # Proof generation phase
                proof_start = time.time()
                if framework_name == "ezkl":
                    proof, output = await self.zkml_integration.prove_inference(
                        model_name, input_data
                    )
                elif framework_name in ["groth16", "plonky", "halo"]:
                    backend = self.rust_manager.get_backend(framework_name)
                    witness = [str(float(x)) for x in input_data.flatten()]
                    proof_dict = backend.prove(witness)
                    proof = proof_dict
                    output = proof_dict.get("public_inputs", [])
                proof_end = time.time()
                result.proof_time = proof_end - proof_start

                # Verification phase
                verify_start = time.time()
                if framework_name == "ezkl":
                    is_valid = await self.zkml_integration.verify_inference(
                        model_name, proof, input_data
                    )
                elif framework_name in ["groth16", "plonky", "halo"]:
                    backend = self.rust_manager.get_backend(framework_name)
                    is_valid = backend.verify(proof)
                verify_end = time.time()
                result.verification_time = verify_end - verify_start

                # Calculate total metrics
                total_time, memory_usage, cpu_usage = monitor.stop_monitoring()
                result.total_time = total_time
                result.memory_usage_mb = memory_usage
                result.cpu_usage_percent = cpu_usage
                result.success = is_valid

                # Estimate proof size (rough approximation)
                if isinstance(proof, dict):
                    result.proof_size_bytes = len(json.dumps(proof).encode())
                else:
                    result.proof_size_bytes = len(str(proof).encode())

                # Cleanup
                self.zkml_integration.cleanup_model(model_name)

            except Exception as e:
                total_time, memory_usage, cpu_usage = monitor.stop_monitoring()
                result.total_time = total_time
                result.memory_usage_mb = memory_usage
                result.cpu_usage_percent = cpu_usage
                result.success = False
                result.error_message = str(e)
                self.logger.error(f"Benchmark failed for {framework_name}: {e}")

            results.append(result)

        return results

    async def benchmark_all_frameworks(
        self,
        architectures: List[str] = None,
        complexities: List[str] = None,
        input_sizes: List[int] = None,
        num_trials: int = 3,
    ) -> List[BenchmarkResult]:
        """Benchmark all available frameworks with different configurations."""
        if architectures is None:
            architectures = ["simple", "multi_exit"]
        if complexities is None:
            complexities = ["light", "medium"]
        if input_sizes is None:
            input_sizes = [10, 50]

        frameworks = ["ezkl", "groth16", "plonky", "halo"]
        all_results = []

        for framework in frameworks:
            for architecture in architectures:
                for complexity in complexities:
                    for input_size in input_sizes:
                        self.logger.info(
                            f"Benchmarking {framework} with {architecture}/{complexity}, input_size={input_size}"
                        )

                        results = await self.benchmark_framework(
                            framework,
                            architecture,
                            complexity,
                            input_size,
                            5,
                            num_trials,
                        )
                        all_results.extend(results)

        self.results.extend(all_results)
        return all_results


class CryptographicPrimitiveBenchmark:
    """Benchmark cryptographic primitives."""

    def __init__(self):
        self.results = []
        self.logger = logging.getLogger(__name__)

    def benchmark_rabin_fingerprint(
        self, input_sizes: List[int] = None, num_trials: int = 100
    ) -> List[CryptoBenchmarkResult]:
        """Benchmark Rabin fingerprint operations."""
        if input_sizes is None:
            input_sizes = [64, 256, 1024, 4096]

        results = []

        for input_size in input_sizes:
            # Test hash computation
            rabin = RabinFingerprint()
            test_data = np.random.bytes(input_size)

            monitor = PerformanceMonitor()
            monitor.start_monitoring()

            start_time = time.time()
            for _ in range(num_trials):
                try:
                    hash_result = rabin.compute_hash(test_data)
                except Exception as e:
                    self.logger.error(f"Rabin fingerprint failed: {e}")
                    continue
            end_time = time.time()

            execution_time, memory_usage, _ = monitor.stop_monitoring()
            avg_time_per_op = (end_time - start_time) / num_trials
            throughput = 1.0 / avg_time_per_op if avg_time_per_op > 0 else 0

            result = CryptoBenchmarkResult(
                primitive_name="RabinFingerprint",
                operation="compute_hash",
                input_size=input_size,
                execution_time=avg_time_per_op,
                memory_usage_mb=memory_usage,
                throughput_ops_per_sec=throughput,
                success=True,
            )
            results.append(result)

        self.results.extend(results)
        return results

    def benchmark_shamir_secret_sharing(
        self, secret_sizes: List[int] = None, num_trials: int = 50
    ) -> List[CryptoBenchmarkResult]:
        """Benchmark Shamir secret sharing operations."""
        if secret_sizes is None:
            secret_sizes = [32, 64, 128, 256]

        results = []

        for secret_size in secret_sizes:
            sss = ShamirSecretSharing(threshold=3, num_shares=5)
            test_secret = np.random.randint(
                0, 256, secret_size, dtype=np.uint8
            ).tobytes()

            # Benchmark share generation
            monitor = PerformanceMonitor()
            monitor.start_monitoring()

            start_time = time.time()
            for _ in range(num_trials):
                try:
                    shares = sss.generate_shares(test_secret)
                except Exception as e:
                    self.logger.error(f"Share generation failed: {e}")
                    continue
            end_time = time.time()

            execution_time, memory_usage, _ = monitor.stop_monitoring()
            avg_time_per_op = (end_time - start_time) / num_trials
            throughput = 1.0 / avg_time_per_op if avg_time_per_op > 0 else 0

            result = CryptoBenchmarkResult(
                primitive_name="ShamirSecretSharing",
                operation="generate_shares",
                input_size=secret_size,
                execution_time=avg_time_per_op,
                memory_usage_mb=memory_usage,
                throughput_ops_per_sec=throughput,
                success=True,
            )
            results.append(result)

            # Benchmark secret reconstruction
            if "shares" in locals():
                monitor = PerformanceMonitor()
                monitor.start_monitoring()

                start_time = time.time()
                for _ in range(num_trials):
                    try:
                        reconstructed = sss.reconstruct_secret(
                            shares[:3]
                        )  # Use threshold number of shares
                    except Exception as e:
                        self.logger.error(f"Secret reconstruction failed: {e}")
                        continue
                end_time = time.time()

                execution_time, memory_usage, _ = monitor.stop_monitoring()
                avg_time_per_op = (end_time - start_time) / num_trials
                throughput = 1.0 / avg_time_per_op if avg_time_per_op > 0 else 0

                result = CryptoBenchmarkResult(
                    primitive_name="ShamirSecretSharing",
                    operation="reconstruct_secret",
                    input_size=secret_size,
                    execution_time=avg_time_per_op,
                    memory_usage_mb=memory_usage,
                    throughput_ops_per_sec=throughput,
                    success=True,
                )
                results.append(result)

        self.results.extend(results)
        return results

    def benchmark_rust_primitives(self) -> List[CryptoBenchmarkResult]:
        """Benchmark Rust-based cryptographic primitives."""
        results = []
        rust_manager = RustZKMLFrameworkManager()

        # Benchmark field operations
        field_sizes = ["17", "101", "2147483647"]  # Small, medium, large primes

        for field_size in field_sizes:
            try:
                monitor = PerformanceMonitor()
                monitor.start_monitoring()

                benchmark_results = (
                    rust_manager.base_backend.benchmark_field_operations(
                        field_size, 1000
                    )
                )

                execution_time, memory_usage, _ = monitor.stop_monitoring()

                for operation, time_ms in benchmark_results.items():
                    result = CryptoBenchmarkResult(
                        primitive_name="RustFieldOperations",
                        operation=f"{operation}_field_{field_size}",
                        input_size=int(field_size),
                        execution_time=time_ms / 1000.0,  # Convert to seconds
                        memory_usage_mb=memory_usage,
                        throughput_ops_per_sec=1000.0 / (time_ms / 1000.0)
                        if time_ms > 0
                        else 0,
                        success=True,
                    )
                    results.append(result)

            except Exception as e:
                self.logger.error(f"Rust field operations benchmark failed: {e}")

        # Benchmark hash functions
        data_sizes = [64, 256, 1024, 4096]

        for data_size in data_sizes:
            test_data = np.random.bytes(data_size)

            # SHA256
            try:
                monitor = PerformanceMonitor()
                monitor.start_monitoring()

                start_time = time.time()
                for _ in range(100):
                    hash_result = rust_manager.base_backend.sha256_hash(test_data)
                end_time = time.time()

                execution_time, memory_usage, _ = monitor.stop_monitoring()
                avg_time = (end_time - start_time) / 100

                result = CryptoBenchmarkResult(
                    primitive_name="RustSHA256",
                    operation="hash",
                    input_size=data_size,
                    execution_time=avg_time,
                    memory_usage_mb=memory_usage,
                    throughput_ops_per_sec=1.0 / avg_time if avg_time > 0 else 0,
                    success=True,
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"Rust SHA256 benchmark failed: {e}")

            # Keccak256
            try:
                monitor = PerformanceMonitor()
                monitor.start_monitoring()

                start_time = time.time()
                for _ in range(100):
                    hash_result = rust_manager.base_backend.keccak256_hash(test_data)
                end_time = time.time()

                execution_time, memory_usage, _ = monitor.stop_monitoring()
                avg_time = (end_time - start_time) / 100

                result = CryptoBenchmarkResult(
                    primitive_name="RustKeccak256",
                    operation="hash",
                    input_size=data_size,
                    execution_time=avg_time,
                    memory_usage_mb=memory_usage,
                    throughput_ops_per_sec=1.0 / avg_time if avg_time > 0 else 0,
                    success=True,
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"Rust Keccak256 benchmark failed: {e}")

        self.results.extend(results)
        return results


class ComprehensiveBenchmarkSuite:
    """Main benchmark suite that orchestrates all benchmarks."""

    def __init__(self, output_dir: str = "/tmp/phaze_benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.zkml_benchmark = ZKMLFrameworkBenchmark()
        self.crypto_benchmark = CryptographicPrimitiveBenchmark()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "benchmark.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    async def run_full_benchmark_suite(
        self, zkml_config: Dict[str, Any] = None, crypto_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        self.logger.info("Starting comprehensive PHAZE benchmark suite")

        results = {
            "zkml_results": [],
            "crypto_results": [],
            "summary": {},
            "timestamp": time.time(),
        }

        # Default configurations
        if zkml_config is None:
            zkml_config = {
                "architectures": ["simple", "multi_exit"],
                "complexities": ["light", "medium"],
                "input_sizes": [10, 50],
                "num_trials": 2,
            }

        if crypto_config is None:
            crypto_config = {
                "rabin_input_sizes": [64, 256, 1024],
                "shamir_secret_sizes": [32, 64, 128],
                "num_trials": 50,
            }

        try:
            # Run zkML framework benchmarks
            self.logger.info("Running zkML framework benchmarks")
            zkml_results = await self.zkml_benchmark.benchmark_all_frameworks(
                **zkml_config
            )
            results["zkml_results"] = [asdict(r) for r in zkml_results]

            # Run cryptographic primitive benchmarks
            self.logger.info("Running cryptographic primitive benchmarks")

            rabin_results = self.crypto_benchmark.benchmark_rabin_fingerprint(
                crypto_config["rabin_input_sizes"], crypto_config["num_trials"]
            )

            shamir_results = self.crypto_benchmark.benchmark_shamir_secret_sharing(
                crypto_config["shamir_secret_sizes"], crypto_config["num_trials"]
            )

            rust_results = self.crypto_benchmark.benchmark_rust_primitives()

            all_crypto_results = rabin_results + shamir_results + rust_results
            results["crypto_results"] = [asdict(r) for r in all_crypto_results]

            # Generate summary statistics
            results["summary"] = self._generate_summary(
                zkml_results, all_crypto_results
            )

            # Save results
            self._save_results(results)

            # Generate visualizations
            self._generate_visualizations(zkml_results, all_crypto_results)

            self.logger.info("Benchmark suite completed successfully")

        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            results["error"] = str(e)

        return results

    def _generate_summary(
        self,
        zkml_results: List[BenchmarkResult],
        crypto_results: List[CryptoBenchmarkResult],
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {"zkml_summary": {}, "crypto_summary": {}, "overall_summary": {}}

        # zkML summary
        if zkml_results:
            successful_zkml = [r for r in zkml_results if r.success]

            summary["zkml_summary"] = {
                "total_tests": len(zkml_results),
                "successful_tests": len(successful_zkml),
                "success_rate": len(successful_zkml) / len(zkml_results)
                if zkml_results
                else 0,
                "avg_setup_time": np.mean([r.setup_time for r in successful_zkml])
                if successful_zkml
                else 0,
                "avg_proof_time": np.mean([r.proof_time for r in successful_zkml])
                if successful_zkml
                else 0,
                "avg_verification_time": np.mean(
                    [r.verification_time for r in successful_zkml]
                )
                if successful_zkml
                else 0,
                "avg_total_time": np.mean([r.total_time for r in successful_zkml])
                if successful_zkml
                else 0,
                "avg_memory_usage": np.mean(
                    [r.memory_usage_mb for r in successful_zkml]
                )
                if successful_zkml
                else 0,
                "frameworks_tested": list(set(r.framework for r in zkml_results)),
                "architectures_tested": list(set(r.architecture for r in zkml_results)),
            }

        # Crypto summary
        if crypto_results:
            successful_crypto = [r for r in crypto_results if r.success]

            summary["crypto_summary"] = {
                "total_tests": len(crypto_results),
                "successful_tests": len(successful_crypto),
                "success_rate": len(successful_crypto) / len(crypto_results)
                if crypto_results
                else 0,
                "avg_execution_time": np.mean(
                    [r.execution_time for r in successful_crypto]
                )
                if successful_crypto
                else 0,
                "avg_throughput": np.mean(
                    [r.throughput_ops_per_sec for r in successful_crypto]
                )
                if successful_crypto
                else 0,
                "primitives_tested": list(
                    set(r.primitive_name for r in crypto_results)
                ),
                "operations_tested": list(set(r.operation for r in crypto_results)),
            }

        # Overall summary
        total_tests = len(zkml_results) + len(crypto_results)
        successful_tests = len([r for r in zkml_results if r.success]) + len(
            [r for r in crypto_results if r.success]
        )

        summary["overall_summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "overall_success_rate": successful_tests / total_tests
            if total_tests > 0
            else 0,
            "test_categories": ["zkML Frameworks", "Cryptographic Primitives"],
        }

        return summary

    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to files."""
        # Save JSON results
        with open(self.output_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV results for zkML
        if results["zkml_results"]:
            zkml_df = pd.DataFrame(results["zkml_results"])
            zkml_df.to_csv(self.output_dir / "zkml_benchmark_results.csv", index=False)

        # Save CSV results for crypto
        if results["crypto_results"]:
            crypto_df = pd.DataFrame(results["crypto_results"])
            crypto_df.to_csv(
                self.output_dir / "crypto_benchmark_results.csv", index=False
            )

        self.logger.info(f"Results saved to {self.output_dir}")

    def _generate_visualizations(
        self,
        zkml_results: List[BenchmarkResult],
        crypto_results: List[CryptoBenchmarkResult],
    ):
        """Generate visualization plots."""
        plt.style.use("default")

        # zkML visualizations
        if zkml_results:
            successful_zkml = [r for r in zkml_results if r.success]

            if successful_zkml:
                # Framework comparison
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                # Proof time by framework
                frameworks = [r.framework for r in successful_zkml]
                proof_times = [r.proof_time for r in successful_zkml]

                axes[0, 0].bar(frameworks, proof_times)
                axes[0, 0].set_title("Proof Generation Time by Framework")
                axes[0, 0].set_ylabel("Time (seconds)")
                axes[0, 0].tick_params(axis="x", rotation=45)

                # Memory usage by complexity
                complexities = [r.complexity for r in successful_zkml]
                memory_usage = [r.memory_usage_mb for r in successful_zkml]

                axes[0, 1].scatter(complexities, memory_usage)
                axes[0, 1].set_title("Memory Usage by Model Complexity")
                axes[0, 1].set_ylabel("Memory (MB)")

                # Total time vs input size
                input_sizes = [r.input_size for r in successful_zkml]
                total_times = [r.total_time for r in successful_zkml]

                axes[1, 0].scatter(input_sizes, total_times)
                axes[1, 0].set_title("Total Time vs Input Size")
                axes[1, 0].set_xlabel("Input Size")
                axes[1, 0].set_ylabel("Total Time (seconds)")

                # Success rate by framework
                framework_success = {}
                for r in zkml_results:
                    if r.framework not in framework_success:
                        framework_success[r.framework] = {"total": 0, "success": 0}
                    framework_success[r.framework]["total"] += 1
                    if r.success:
                        framework_success[r.framework]["success"] += 1

                frameworks = list(framework_success.keys())
                success_rates = [
                    framework_success[f]["success"] / framework_success[f]["total"]
                    for f in frameworks
                ]

                axes[1, 1].bar(frameworks, success_rates)
                axes[1, 1].set_title("Success Rate by Framework")
                axes[1, 1].set_ylabel("Success Rate")
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].tick_params(axis="x", rotation=45)

                plt.tight_layout()
                plt.savefig(
                    self.output_dir / "zkml_benchmark_plots.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        # Crypto visualizations
        if crypto_results:
            successful_crypto = [r for r in crypto_results if r.success]

            if successful_crypto:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))

                # Throughput by primitive
                primitives = [r.primitive_name for r in successful_crypto]
                throughputs = [r.throughput_ops_per_sec for r in successful_crypto]

                axes[0, 0].bar(primitives, throughputs)
                axes[0, 0].set_title("Throughput by Cryptographic Primitive")
                axes[0, 0].set_ylabel("Operations per Second")
                axes[0, 0].tick_params(axis="x", rotation=45)

                # Execution time vs input size
                input_sizes = [r.input_size for r in successful_crypto]
                exec_times = [r.execution_time for r in successful_crypto]

                axes[0, 1].scatter(input_sizes, exec_times)
                axes[0, 1].set_title("Execution Time vs Input Size")
                axes[0, 1].set_xlabel("Input Size (bytes)")
                axes[0, 1].set_ylabel("Execution Time (seconds)")
                axes[0, 1].set_xscale("log")
                axes[0, 1].set_yscale("log")

                # Memory usage by operation
                operations = [r.operation for r in successful_crypto]
                memory_usage = [r.memory_usage_mb for r in successful_crypto]

                axes[1, 0].scatter(operations, memory_usage)
                axes[1, 0].set_title("Memory Usage by Operation")
                axes[1, 0].set_ylabel("Memory (MB)")
                axes[1, 0].tick_params(axis="x", rotation=45)

                # Performance comparison
                primitive_perf = {}
                for r in successful_crypto:
                    if r.primitive_name not in primitive_perf:
                        primitive_perf[r.primitive_name] = []
                    primitive_perf[r.primitive_name].append(r.throughput_ops_per_sec)

                primitives = list(primitive_perf.keys())
                avg_throughputs = [np.mean(primitive_perf[p]) for p in primitives]

                axes[1, 1].bar(primitives, avg_throughputs)
                axes[1, 1].set_title("Average Throughput by Primitive")
                axes[1, 1].set_ylabel("Average Ops/sec")
                axes[1, 1].tick_params(axis="x", rotation=45)

                plt.tight_layout()
                plt.savefig(
                    self.output_dir / "crypto_benchmark_plots.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

        self.logger.info(f"Visualizations saved to {self.output_dir}")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report."""
        report_lines = [
            "# PHAZE Framework Comprehensive Benchmark Report",
            f"Generated at: {time.ctime(results['timestamp'])}",
            "",
            "## Executive Summary",
            "",
        ]

        summary = results.get("summary", {})
        overall = summary.get("overall_summary", {})

        report_lines.extend(
            [
                f"- Total tests executed: {overall.get('total_tests', 0)}",
                f"- Successful tests: {overall.get('successful_tests', 0)}",
                f"- Overall success rate: {overall.get('overall_success_rate', 0):.2%}",
                "",
            ]
        )

        # zkML section
        zkml_summary = summary.get("zkml_summary", {})
        if zkml_summary:
            report_lines.extend(
                [
                    "## zkML Framework Benchmarks",
                    "",
                    f"- Frameworks tested: {', '.join(zkml_summary.get('frameworks_tested', []))}",
                    f"- Architectures tested: {', '.join(zkml_summary.get('architectures_tested', []))}",
                    f"- Success rate: {zkml_summary.get('success_rate', 0):.2%}",
                    f"- Average setup time: {zkml_summary.get('avg_setup_time', 0):.3f}s",
                    f"- Average proof time: {zkml_summary.get('avg_proof_time', 0):.3f}s",
                    f"- Average verification time: {zkml_summary.get('avg_verification_time', 0):.3f}s",
                    f"- Average memory usage: {zkml_summary.get('avg_memory_usage', 0):.2f}MB",
                    "",
                ]
            )

        # Crypto section
        crypto_summary = summary.get("crypto_summary", {})
        if crypto_summary:
            report_lines.extend(
                [
                    "## Cryptographic Primitive Benchmarks",
                    "",
                    f"- Primitives tested: {', '.join(crypto_summary.get('primitives_tested', []))}",
                    f"- Operations tested: {', '.join(crypto_summary.get('operations_tested', []))}",
                    f"- Success rate: {crypto_summary.get('success_rate', 0):.2%}",
                    f"- Average execution time: {crypto_summary.get('avg_execution_time', 0):.6f}s",
                    f"- Average throughput: {crypto_summary.get('avg_throughput', 0):.2f} ops/sec",
                    "",
                ]
            )

        report_lines.extend(
            [
                "## Recommendations",
                "",
                "Based on the benchmark results:",
                "1. Consider the trade-offs between proof generation time and verification time",
                "2. Monitor memory usage for large-scale deployments",
                "3. Choose appropriate model complexity based on performance requirements",
                "4. Evaluate cryptographic primitives based on throughput needs",
                "",
            ]
        )

        report_content = "\n".join(report_lines)

        # Save report
        with open(self.output_dir / "benchmark_report.md", "w") as f:
            f.write(report_content)

        return report_content


# Convenience function for running benchmarks
async def run_phaze_benchmarks(
    output_dir: str = "/tmp/phaze_benchmarks", quick_mode: bool = False
) -> Dict[str, Any]:
    """Run PHAZE benchmarks with optional quick mode."""
    suite = ComprehensiveBenchmarkSuite(output_dir)

    if quick_mode:
        zkml_config = {
            "architectures": ["simple"],
            "complexities": ["light"],
            "input_sizes": [10],
            "num_trials": 1,
        }
        crypto_config = {
            "rabin_input_sizes": [64],
            "shamir_secret_sizes": [32],
            "num_trials": 10,
        }
    else:
        zkml_config = None  # Use defaults
        crypto_config = None  # Use defaults

    results = await suite.run_full_benchmark_suite(zkml_config, crypto_config)
    report = suite.generate_report(results)

    return {"results": results, "report": report, "output_directory": output_dir}
